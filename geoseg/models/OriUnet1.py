import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from backbones.resnet_for_bes import FCNHead, UP_Conv, resnet18

# 下采 加权上采 MSF模块
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 比 1 反卷上采样后加了两次卷积
class WeightedUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # else:
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                                DoubleConv(in_channels // 2, out_channels))
        self.fusion = FusionBlock(in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        if x1.shape[2] != x2.shape[2]:
            x1 = self.up(x1)
        x2 = self.fusion(x1, x2)
        return x2


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ConvBnReLU(nn.Sequential):
    def __init__(
            self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch))

        if relu:
            self.add_module("relu", nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class FusionBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, mid_ch):
        super(FusionBlock, self).__init__()
        self.input1 = ConvBnReLU(in_ch1, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.input2 = ConvBnReLU(in_ch2, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)

        self.fusion = nn.Sequential(
            ConvBnReLU(mid_ch * 2, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Sigmoid(),
            GlobalAvgPool2d()
        )

    def forward(self, f1, f2):
        # 先判断channel是否相同 不相同先统一
        if f1.shape[1] != f2.shape[1]:
            f1 = self.input1(f1)
            f2 = self.input2(f2)

        w = torch.cat((f1, f2), dim=1)
        w = self.fusion(w)
        w = w.unsqueeze(2).unsqueeze(3).expand_as(f1)  # 把w1变成f5的size
        m = (1 - w) * f1 + w * f2

        return m


class MSF_Module(nn.Module):
    def __init__(self, in_ch, mid_ch1, cat_ch, mid_ch2, out_ch):
        super(MSF_Module, self).__init__()

        self.input1 = ConvBnReLU(in_ch[0], mid_ch1, kernel_size=1, stride=2, padding=0, dilation=1)
        self.input2 = ConvBnReLU(in_ch[1], mid_ch1, kernel_size=1, stride=1, padding=0, dilation=1)
        self.input3 = ConvBnReLU(in_ch[2], mid_ch1, kernel_size=1, stride=1, padding=0, dilation=1)

        self.fusion1 = nn.Sequential(
            ConvBnReLU(cat_ch, mid_ch2, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.Conv2d(mid_ch2, mid_ch2, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Sigmoid(),
            GlobalAvgPool2d()
        )

        self.fusion2 = nn.Sequential(
            ConvBnReLU(cat_ch, mid_ch2, kernel_size=1, stride=1, padding=0, dilation=1),
            nn.Conv2d(mid_ch2, out_ch, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Sigmoid(),
            GlobalAvgPool2d()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, l3, l4, l5):
        f3 = self.input1(l3)
        f4 = self.input2(l4)
        f5 = self.input3(l5)

        f5 = F.interpolate(f5, f4.size()[2:], mode='bilinear', align_corners=True)  # f5上采至f4相同size

        w1 = torch.cat((f4, f5), dim=1)
        w1 = self.fusion1(w1).unsqueeze(2).unsqueeze(3).expand_as(f5)
        m1 = (1 - w1) * f4 + w1 * f5

        w2 = torch.cat((m1, f3), dim=1)
        w2 = self.fusion2(w2).unsqueeze(2).unsqueeze(3).expand_as(f5)
        m2 = (1 - w2) * f3 + w2 * m1

        return m2


def Upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


class MyUNet(nn.Module):
    def __init__(self, n_classes, backbone='resnet18',bilinear=False):
        super(MyUNet, self).__init__()

        self.backbone = timm.create_model(backbone, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)
        self.Fusion = MSF_Module([128, 256, 512], 128, 256, 128, 128)  # in_ch(元组), mid_ch1, cat_ch, mid_ch2, out_ch
        self.down = nn.Conv2d(256, 64, kernel_size=1)

        self.n_classes = n_classes

        self.wup1 = WeightedUp(512, 256)
        self.wup = WeightedUp(256, 128)
        self.wup2 = WeightedUp(128, 128)
        self.wup3 = WeightedUp(128, 64)
        self.wup4 = WeightedUp(128, 64)
        self.wup5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.backbone(x)
        f = self.Fusion(c2, c3, c4)  # 4,128,64,64

        x = self.wup1(c4, c3)
        x = self.wup(x, f)
        x = self.wup2(x, c2)
        x = self.wup3(x, c1)  # 4,64,256,256
        x = self.wup5(x)
        #
        # x = self.wup4(x)
        logits = self.outc(x)
        x = Upsample(logits, imsize)
        return x
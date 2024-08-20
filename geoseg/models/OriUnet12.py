import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

# 下采 加权上采 每次上采后加两次卷积 加CAB
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


# 看之后可不可以改成bes中的上采样
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2),
                                DoubleConv(in_channels // 2, out_channels))
        self.fusion = FusionBlock(in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        # x1 = self.up(x1)
        # # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # x = torch.cat([x2, x1], dim=1)
        # return self.conv(x)
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


def Upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


class RRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res  = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)


class CAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAB, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low
        x = torch.cat([x1,x2],dim=1)
        x = self.global_pooling(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class SAB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x1, x2 = x  # high, low
        x = torch.cat([x1,x2],dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sigmod(x)
        x2 = x * x2
        res = x2 + x1
        return res


class MAFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MAFB, self).__init__()
        self.cab = CAB(in_channels, out_channels)
        self.sab = SAB(in_channels, out_channels)

    def forward(self, x):
        x = self.cab(x)
        x = self.sab(x)
        return x


class MyUNet(nn.Module):
    def __init__(self, n_classes, backbone='resnet18'):
        super(MyUNet, self).__init__()

        self.backbone = timm.create_model(backbone, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=True)

        self.down = nn.Conv2d(64, 64, kernel_size=2, stride=2)

        self.n_classes = n_classes

        self.rrb1 = RRB(64, 64)
        self.rrb2 = RRB(128, 128)
        self.rrb3 = RRB(256, 256)
        self.rrb4 = RRB(512, 512)

        self.mafb1 = SAB(128, 64)
        self.mafb2 = SAB(256, 128)
        self.mafb3 = CAB(512, 256)
        self.mafb4 = CAB(1024, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)

        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        imsize = x.size()[2:]
        f1, f2, f3, f4 = self.backbone(x)
        r1 = self.rrb1(f1)
        r2 = self.rrb2(f2)
        r3 = self.rrb3(f3)
        r4 = self.rrb4(f4)

        m1 = self.mafb1([f1, r1])
        m2 = self.mafb2([f2, r2])
        m3 = self.mafb3([f3, r3])
        m4 = self.mafb4([f4, r4])

        x = self.up1(m4, m3)
        x = self.up2(x, m2)
        x = self.up3(x, m1)
        x = self.up4(x)
        logits = self.outc(x)
        x = Upsample(logits, imsize)
        return x
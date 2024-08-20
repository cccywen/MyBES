import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.resnet_for_bes import FCNHead, UP_Conv, resnet18

# 下采 加权上采
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

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
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


# ASF module
class ScaleFeatureSelection(nn.Module):
    def __init__(self, in_channels, inter_channels, out_features_num=4, attention_type='scale_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.type = attention_type
        self.enhanced_attention = ScaleSpatialAttention(inter_channels, inter_channels//4, out_features_num)

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, concat_x, features_list):
        # N×C×H×W
        concat_x = self.conv(concat_x)  # 3×3卷积生成中间feature
        # C×H×W
        score = self.enhanced_attention(concat_x)  # 算attention weight

        assert len(features_list) == self.out_features_num
        if self.type not in ['scale_channel_spatial', 'scale_spatial']:
            shape = features_list[0].shape[2:]
            score = F.interpolate(score, size=shape, mode='bilinear')
        x = []
        for i in range(self.out_features_num):
            x.append(score[:, i:i+1] * features_list[i])  # feature × 对应的weight
        return torch.cat(x, dim=1)


class ScaleSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleSpatialAttention, self).__init__()
        self.spatial_wise = nn.Sequential(
            # Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        global_x = torch.mean(x, dim=1, keepdim=True)
        global_x = self.spatial_wise(global_x) + x
        global_x = self.attention_wise(global_x)
        return global_x


class MyUNet(nn.Module):
    def __init__(self, n_classes, backbone='resnet18',bilinear=False):
        super(MyUNet, self).__init__()

        resnet = eval(backbone)(pretrained=True)
        in_channels = [64, 128, 256, 512]
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # 用于进FusionBlock前统一channel##############################
        inner_channels = 256
        self.in3 = nn.Conv2d(in_channels[3], inner_channels, 1)
        self.in2 = nn.Conv2d(in_channels[2], inner_channels, 1)
        self.in1 = nn.Conv2d(in_channels[1], inner_channels, 1)
        self.in0 = nn.Conv2d(in_channels[0], inner_channels, 1)
        self.out3 = nn.Sequential(
            nn.Conv2d(in_channels[3], inner_channels // 8, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Sequential(
            nn.Conv2d(in_channels[2], inner_channels // 8, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out1 = nn.Sequential(
            nn.Conv2d(in_channels[1], inner_channels // 8, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out0 = nn.Conv2d(inner_channels, inner_channels // 8, 3, padding=1)
        #############################################################################

        self.fusion1 = FusionBlock(64, 64, 64)
        self.fusion2 = FusionBlock(64, 128, 64)
        self.fusion3 = FusionBlock(64, 256, 128)
        self.fusion4 = FusionBlock(128, 512, 256)

        self.down = nn.Conv2d(64, 64, kernel_size=2, stride=2)

        self.n_classes = n_classes
        self.bilinear = bilinear



        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        # 要先统一channel（后续优化 减少代码）
        self.asf = ScaleFeatureSelection(128, inner_channels // 4)  # 有几个feature就除几

    def forward(self, x):
        imsize = x.size()[2:]
        c0 = x = self.layer0(x)
        c1 = x = self.layer1(x)
        c2 = x = self.layer2(x)
        c3 = x = self.layer3(x)
        c4 = x = self.layer4(x)

        # 为了asf
        in3 = self.in3(c3)  # 统一channel
        in2 = self.in2(c2)
        in1 = self.in1(c1)
        in0 = self.in0(c0)
        out4 = in3 + in2  # 1/16
        out3 = out4 + in1  # 1/8
        out2 = self.up3(out3) + in0  # 1/4
        p5 = self.out5(in3)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)

        fuse = torch.cat((p5, p4, p3, p2), 1)  # 全部 4，64，256，256
        fuse = self.asf(fuse, [p5, p4, p3, p2])  # output4，64，256，256

        # 对c5 aspp与fuse融合后上采
        x = self.up1(fuse, c4)
        x = self.up1(c4, c3)
        x = self.up2(x, c2)
        x = self.up3(x, c1)
        # x = self.up4(x, c0)
        logits = self.outc(x)
        x = Upsample(logits, imsize)
        return x
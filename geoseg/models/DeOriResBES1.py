import logging
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from backbones.resnet_for_bes import FCNHead, UP_Conv, resnet18

# 尺寸不匹配时用上采样 BE模块不加f4高级特征
class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
        logging.info("Global Average Pooling Initialized")

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


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


def Upsample(x, size):
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=True)


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True))
    return block


# ASPP将全局context纳入模型，最后一个feature map用GAP
class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AsppPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return Upsample(pool, (h, w))  # GAP后的分辨率1×1，要上采样才能concat


class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)  # tuple元组
        self.b0 = nn.Sequential(    # 第一个feature map 1×1
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)  # 3×3
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = AsppPooling(in_channels, out_channels)  # 最后一个feature map GAP

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return y


class BE_Module(nn.Module):
    def __init__(self, in_ch1, in_ch2, mid_ch, out_ch, n_class):
        super(BE_Module, self).__init__()

        self.convb_1 = ConvBnReLU(in_ch1, mid_ch, kernel_size=1, stride=2, padding=0, dilation=1)  # 用来统一channel
        self.convb_2 = ConvBnReLU(in_ch2, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.convbloss = nn.Conv2d(mid_ch, n_class, kernel_size=1, bias=False)
        boundary_ch = 3 * mid_ch
        self.boundaryconv = ConvBnReLU(boundary_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)

                nn.init.constant_(m.bias, 0)

    def forward(self, l1, l2):
        # 统一channel,F5需要上采
        l1_b = self.convb_1(l1)
        l2_b = self.convb_2(l2)

        l1_bl = self.convbloss(l1_b)  # l1的boundary
        l2_bl = self.convbloss(l2_b)


        b = torch.cat((l1_b, l2_b, ), dim=1)  # 输出的boundary feature Fb
        b = self.boundaryconv(b)

        c_boundaryloss = l1_bl + l2_bl

        return b, c_boundaryloss


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


class BES_Module(nn.Module):
    def __init__(self, f5_in, mul_ch):
        super(BES_Module, self).__init__()
        aspp_out = 5 * f5_in // 8
        self.aspp = ASPP_Module(f5_in, atrous_rates=[12, 24, 36])
        self.f5_out = ConvBnReLU(aspp_out, mul_ch, kernel_size=3, stride=1, padding=1, dilation=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, f5, fb, ff):
        aspp = self.aspp(f5)
        f5 = self.f5_out(aspp)
        f5 = F.interpolate(f5, fb.size()[2:], mode='bilinear', align_corners=True)
        f5_guide = torch.mul(f5, fb)  # 对应位置相乘
        ff = F.interpolate(ff, fb.size()[2:], mode='bilinear', align_corners=True)
        ff_guide = torch.mul(ff, fb)
        fe = ff + ff_guide + f5_guide

        return fe


class BESNet(nn.Module):
    def __init__(self, nclass, backbone='resnet18', aux=True, norm_layer=nn.BatchNorm2d, pretrained=True):
        super(BESNet, self).__init__()

        self.aux = aux  # 辅助loss
        self.backbone = timm.create_model(backbone, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        filters = self.backbone.feature_info.channels()

        self.BoundaryExtraction = BE_Module(64, 128, 64, 128, 1)  # in_ch1, in_ch2, in_ch5, mid_ch, out_ch, n_class

        self.Fusion = MSF_Module([128, 256, 512], 128, 256, 128, 128)  # in_ch(元组), mid_ch1, cat_ch, mid_ch2, out_ch

        self.up = UP_Conv(512, 512)  # ch_in, ch_out

        self.Enhance = BES_Module(512, 128)  # f4_in, mul_ch

        self.head = FCNHead(128, nclass, norm_layer)  # in_channels, out_channels, norm_layer=nn.BatchNorm2d
        if self.aux:
            self.auxlayer = FCNHead(256, nclass, norm_layer)

    def forward(self, x):

        imsize = x.size()[2:]

        c1, c2, c3, c4 = self.backbone(x)

        b, c_boundaryloss = self.BoundaryExtraction(c1, c2)

        f = self.Fusion(c2, c3, c4)
        c4 = self.up(c4)

        x = self.Enhance(c4, b, f)

        x = self.head(x)
        x = Upsample(x, imsize)
        return x

        # outputs = [x]
        # if self.aux:
        #     auxout = self.auxlayer(c3)
        #     auxout = Upsample(auxout, imsize)
        #     outputs.append(auxout)
        #
        # if self.training and self.aux:
        #     outputs.append(c_boundaryloss)
        #     return tuple(outputs)
        # return x, c_boundaryloss
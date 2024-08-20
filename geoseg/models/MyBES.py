import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.resnet_for_bes import FCNHead, UP_Conv, resnet18

# f4+(f3+(f2+f1))
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
    def __init__(self, in_ch1, in_ch2, in_ch5, mid_ch, out_ch, n_class):
        super(BE_Module, self).__init__()

        self.convb_1 = ConvBnReLU(in_ch1, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)  # 用来统一channel
        self.convb_2 = ConvBnReLU(in_ch2, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
        self.convb_5 = ConvBnReLU(in_ch5, mid_ch, kernel_size=1, stride=1, padding=0, dilation=1)
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

    def forward(self, l1, l2, l5):
        # 统一channel,F5需要上采
        l1_b = self.convb_1(l1)
        l2_b = self.convb_2(l2)
        l5_b = self.convb_5(l5)
        l5_b = F.interpolate(l5_b, l1.size()[2:], mode='bilinear', align_corners=True)

        l1_bl = self.convbloss(l1_b)  # l1的boundary
        l2_bl = self.convbloss(l2_b)

        l5_bl = self.convbloss(l5_b)

        b = torch.cat((l1_b, l2_b, l5_b), dim=1)  # 输出的boundary feature Fb
        b = self.boundaryconv(b)

        c_boundaryloss = l1_bl + l2_bl + l5_bl

        return b, c_boundaryloss


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
        a = f1.shape[1]
        a = f2.shape[1]
        if f1.shape[1] != f2.shape[1]:
            f1 = self.input1(f1)
            f2 = self.input2(f2)

        w = torch.cat((f1, f2), dim=1)
        w = self.fusion(w)
        w = w.unsqueeze(2).unsqueeze(3).expand_as(f1)  # 把w1变成f5的size
        m = (1 - w) * f1 + w * f2

        return m


class BESNet(nn.Module):
    def __init__(self, nclass, backbone='resnet18', aux=True, norm_layer=nn.BatchNorm2d, pretrained=True):
        super(BESNet, self).__init__()

        self.aux = aux  # 辅助loss
        resnet = eval(backbone)(pretrained=pretrained)

        self.fusion1 = FusionBlock(64, 64, 64)
        self.fusion2 = FusionBlock(64, 128, 64)
        self.fusion3 = FusionBlock(64, 256, 128)
        self.fusion4 = FusionBlock(128, 512, 256)


        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.down = nn.Conv2d(64, 64, kernel_size=2, stride=2)

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        # self.BoundaryExtraction = BE_Module(64, 64, 512, 64, 128, 1)  # in_ch1, in_ch2, in_ch5, mid_ch, out_ch, n_class
        # self.Fusion = MSF_Module([128, 256, 512], 128, 256, 128, 128)  # in_ch(元组), mid_ch1, cat_ch, mid_ch2, out_ch
        # self.up = UP_Conv(128, 128)  # ch_in, ch_out
        # self.Enhance = BES_Module(512, 128)  # f5_in, mul_ch
        self.head = FCNHead(256, nclass, norm_layer)  # in_channels, out_channels, norm_layer=nn.BatchNorm2d


        if self.aux:
            self.auxlayer = FCNHead(256, nclass, norm_layer)

    def forward(self, x):

        imsize = x.size()[2:]

        c0 = x = self.layer0(x)
        c1 = x = self.layer1(x)
        c2 = x = self.layer2(x)
        c3 = x = self.layer3(x)
        c4 = x = self.layer4(x)

        # b, c_boundaryloss = self.BoundaryExtraction(c0, c1, c4)

        f1 = self.fusion1(c0, c1)
        # 给F1下采样
        f1 = self.down(f1)
        f2 = self.fusion2(f1, c2)
        f3 = self.fusion3(f2, c3)
        f4 = self.fusion4(f3, c4)


        # x = self.Enhance(c4, b, f)

        x = self.head(f4)
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
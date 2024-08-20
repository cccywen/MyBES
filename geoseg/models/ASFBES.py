import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones.resnet_for_bes import FCNHead, UP_Conv, resnet18

# 返回一个tuple 含x 和与boundary GT做loss的bl 先修改了只输出一个 需要的话要改回
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
        ff_guide = torch.mul(ff, fb)
        fe = ff + ff_guide + f5_guide

        return fe


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


class BESNet(nn.Module):
    def __init__(self, nclass, inner_channels=256, backbone='resnet18', aux=True, norm_layer=nn.BatchNorm2d, pretrained=True):
        super(BESNet, self).__init__()

        self.aux = aux  # 辅助loss
        resnet = eval(backbone)(pretrained=pretrained)
        in_channels = [64, 128, 256, 512]


        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.out5 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 8, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 8, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 8, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(inner_channels, inner_channels // 8, 3, padding=1)

        self.in5 = nn.Conv2d(in_channels[-1], inner_channels, 1)
        self.in4 = nn.Conv2d(in_channels[-2], inner_channels, 1)
        self.in3 = nn.Conv2d(in_channels[-3], inner_channels, 1)
        self.in2 = nn.Conv2d(in_channels[-4], inner_channels, 1)

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

        self.BoundaryExtraction = BE_Module(64, 64, 512, 64, 128, 1)  # in_ch1, in_ch2, in_ch5, mid_ch, out_ch, n_class

        self.up = UP_Conv(128, 128)  # ch_in, ch_out

        self.asf = ScaleFeatureSelection(128, inner_channels//4)
        self.visi_output = nn.Conv2d(128, 3, kernel_size=1, stride=1)
        self.Enhance = BES_Module(512, 128)  # f5_in, mul_ch

        self.head = FCNHead(128, nclass, norm_layer)  # in_channels, out_channels, norm_layer=nn.BatchNorm2d
        if self.aux:
            self.auxlayer = FCNHead(256, nclass, norm_layer)

    def forward(self, x):

        imsize = x.size()[2:]

        c0 = x = self.layer0(x)
        c1 = x = self.layer1(x)
        c2 = x = self.layer2(x)
        c3 = x = self.layer3(x)
        c4 = x = self.layer4(x)

        in5 = self.in5(c4)  # 统一channel
        in4 = self.in4(c3)
        in3 = self.in3(c2)
        in2 = self.in2(c1)
        out4 = in5 + in4  # 1/16
        out3 = out4 + in3  # 1/8
        out2 = self.up3(out3) + in2  # 1/4
        p5 = self.out5(in5)
        p4 = self.out4(out4)
        p3 = self.out3(out3)
        p2 = self.out2(out2)


        fuse = torch.cat((p5, p4, p3, p2), 1)  # 全部 4，64，256，256
        fuse = self.asf(fuse, [p5, p4, p3, p2])  # output4，64，256，256

        b, c_boundaryloss = self.BoundaryExtraction(c0, c1, c4)


        x= self.Enhance(c4, b, fuse)
        x_v = self.visi_output(x)  # 输出3 channel
        x = self.head(x)
        x = Upsample(x, imsize)
        return x, x_v
        # return x

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
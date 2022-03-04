import math
from functools import partial
from math import log2

import torch
import torch.nn as nn
from utils.tensor_ops import upsample_add, upsample_cat
from torch.nn import functional as F
from module.BaseBlocks import PCConv2d, PSELayer, CFConv2d, ISELayer
from config import arg_config


class PCNet_ISE(nn.Module):
    def __init__(self, channel=arg_config["channel"]):
        super(PCNet_ISE, self).__init__()
        self.w, self.h = arg_config["input_size"], arg_config["input_size"]
        self.downsample1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.downsample4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downconv1 = PCConv2d(4, channel, kernel_size=3, stride=1, padding=1)
        self.DPSE1 = PSELayer(in_chnls=channel)
        self.DISE1 = ISELayer(in_chnls=channel, loc="begin")

        self.downconv2 = PCConv2d(channel, 2*channel, kernel_size=3, stride=1, padding=1)
        channel *= 2
        self.DPSE2 = PSELayer(in_chnls=channel)
        self.DISE2 = ISELayer(in_chnls=channel, loc="middle")

        self.downconv3 = PCConv2d(channel, 2*channel, kernel_size=3, stride=1, padding=1)
        channel *= 2
        self.DPSE3 = PSELayer(in_chnls=channel)
        self.DISE3 = ISELayer(in_chnls=channel, loc="middle")

        self.downconv4 = PCConv2d(channel, 2*channel, kernel_size=3, stride=1, padding=1)
        channel *= 2
        self.DPSE4 = PSELayer(in_chnls=channel)
        self.DISE4 = ISELayer(in_chnls=channel, loc="middle")

        self.downconv5 = PCConv2d(channel, 2*channel, kernel_size=3, stride=1, padding=1)
        channel *= 2
        self.DPSE5 = PSELayer(in_chnls=channel)
        self.DISE5 = ISELayer(in_chnls=channel, loc="end")

        """
        weight
        """
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_a = nn.Linear(channel, 384)
        self.fc_b = nn.Linear(channel, 384)

        self.fc_weight = nn.Linear(384, 1)
        self.fc_weight_sigmoid = nn.Sigmoid()

        self.CF5 = CFModule(channel, channel//2)
        channel = channel // 2

        self.CF4_V = AVModule(channel, channel // 2)
        self.CF4_A = AVModule(channel, channel // 2)

        self.CF4 = CFModule(channel, channel//2)

        self.UpConv4_V = PCConv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
        self.UPSE4_V = PSELayer(channel//2)
        self.UpConv4_A = PCConv2d(channel, channel // 2, kernel_size=3, stride=1, padding=1)
        self.UPSE4_A = PSELayer(channel // 2)

        channel = channel // 2
        self.CF3_V = AVModule(channel, channel//2)
        self.CF3_A = AVModule(channel, channel//2)
        self.CF3 = CFModule(channel, channel//2)
        self.UpConv3_V = PCConv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
        self.UPSE3_V = PSELayer(channel//2)
        self.UpConv3_A = PCConv2d(channel, channel//2, kernel_size=3, stride=1, padding=1)
        self.UPSE3_A = PSELayer(channel//2)

        channel = channel // 2
        self.CF2 = CFModule(channel, channel//2)
        self.CF2_V = AVModule(channel, channel//2)
        self.CF2_A = AVModule(channel, channel//2)

        channel = channel // 2
        self.Con_V = PCConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.PSE_V = PSELayer(channel)
        self.Con_A = PCConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.PSE_A = PSELayer(channel)


        self.classifier = nn.Conv2d(channel, 1, 1)
        self.classifier_v = nn.Conv2d(channel, 1, 1)
        self.classifier_a = nn.Conv2d(channel, 1, 1)

    def forward(self, in_data, in_data_norm):
        in_data = torch.cat((in_data, in_data_norm), dim=1)
        in_data_1 = self.downconv1(in_data)
        in_data_1 = self.DPSE1(in_data_1)

        in_data_down_1 = self.downsample1(in_data_1)
        in_data_2 = self.downconv2(in_data_down_1)
        in_data_2 = self.DPSE2(in_data_2)

        in_data_down_2 = self.downsample2(in_data_2)
        in_data_3 = self.downconv3(in_data_down_2)
        in_data_3 = self.DPSE3(in_data_3)

        in_data_down_3 = self.downsample3(in_data_3)
        in_data_4 = self.downconv4(in_data_down_3)
        in_data_4 = self.DPSE4(in_data_4)

        in_data_down_4 = self.downsample3(in_data_4)
        in_data_5 = self.downconv5(in_data_down_4)
        in_data_5 = self.DPSE5(in_data_5)


        in_data_1 = self.DISE1([in_data_1, in_data_2])
        in_data_2 = self.DISE2([in_data_2, in_data_1, in_data_3])
        in_data_3 = self.DISE3([in_data_3, in_data_2, in_data_4])
        in_data_4 = self.DISE4([in_data_4, in_data_3, in_data_5])
        in_data_5 = self.DISE5([in_data_5, in_data_4])

        if self.training and arg_config["AFM"]:
            bs = int(in_data_5.shape[0] / 2)
            x_part1 = in_data_5[:bs, :, :, :]
            x_part2 = in_data_5[bs:, :, :, :]

            in_x = self.avgpool(in_data_5)
            in_x = torch.flatten(in_x, 1)

            part1_x = in_x[:bs, :]
            part2_x = in_x[bs:, :]

            fc_a = F.relu(self.fc_a(part1_x), inplace=True)
            fc_b = F.relu(self.fc_b(part2_x), inplace=True)
            fc_weight = self.fc_weight(fc_a + fc_b)
            weight = self.fc_weight_sigmoid(fc_weight)
            weight = weight.view(weight.shape[0], 1, 1, 1)

            mixup_x = weight * x_part1 + (1 - weight) * x_part2
            in_data_5 = torch.cat([mixup_x, mixup_x], dim=0)

        out_data_5 = self.CF5(in_data_5, in_data_4)

        out_data_4 = self.CF4(out_data_5, in_data_3)

        out_data_4_v = self.CF4_V(out_data_5, in_data_3)
        out_data_4_a = self.CF4_A(out_data_5, in_data_3)
        out_data_4_v = self.UpConv4_V(torch.cat((out_data_4_v, out_data_4), dim=1))
        out_data_4_v = self.UPSE4_V(out_data_4_v)
        out_data_4_a = self.UpConv4_A(torch.cat((out_data_4_a, out_data_4), dim=1))
        out_data_4_a = self.UPSE4_A(out_data_4_a)

        out_data_3 = self.CF3(out_data_4, in_data_2)
        out_data_3_v = self.CF3_V(out_data_4_v, in_data_2)
        out_data_3_a = self.CF3_A(out_data_4_a, in_data_2)
        out_data_3_v = self.UpConv3_V(torch.cat((out_data_3_v, out_data_3), dim=1))
        out_data_3_v = self.UPSE3_V(out_data_3_v)
        out_data_3_a = self.UpConv3_A(torch.cat((out_data_3_a, out_data_3), dim=1))
        out_data_3_a = self.UPSE3_A(out_data_3_a)

        out_data_2 = self.CF2(out_data_3, in_data_1)
        out_data_2_v = self.CF2_V(out_data_3_v, in_data_1)
        out_data_2_a = self.CF2_A(out_data_3_a, in_data_1)

        out_data_v = self.Con_V(out_data_2_v)
        out_data_a = self.Con_A(out_data_2_a)

        out_data = self.classifier(out_data_2)
        out_data_v = self.classifier_v(out_data_v)
        out_data_a = self.classifier_a(out_data_a)
        return out_data_v, out_data_a, out_data


class CFModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CFModule, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_add = upsample_add
        self.upsample_cat = UpCat(out_channel)
        self.upconv = PCConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.UPSE = PSELayer(out_channel)
        self.CF = CFConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv = PCConv2d(2*out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.PSE = PSELayer(out_channel)

    def forward(self, in_data_1, in_data_2):
        out_data = self.upsample(in_data_1)
        out_data_up = self.upconv(out_data)
        # out_data_up = self.upsample_add(out_data_up, in_data_2)
        out_data_up = self.upsample_cat(out_data_up, in_data_2)
        out_data_up = self.UPSE(out_data_up)
        out_data = self.CF(out_data)
        out_data = in_data_2 - out_data
        out_data = torch.cat((out_data_up, out_data), dim=1)
        out_data = self.conv(out_data)
        out_data = self.PSE(out_data)

        return out_data


class AVModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AVModule, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.upsample_add = upsample_add
        self.upsample_cat = UpCat(out_channel)
        self.upconv = PCConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.UPSE = PSELayer(out_channel)
        self.conv = PCConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.PSE = PSELayer(out_channel)

    def forward(self, in_data_1, in_data_2):
        out_data = self.upsample(in_data_1)
        out_data_up = self.upconv(out_data)
        # out_data_up = self.upsample_add(out_data_up, in_data_2)
        out_data_up = self.upsample_cat(out_data_up, in_data_2)
        out_data_up = self.UPSE(out_data_up)
        out_data = self.conv(out_data_up)
        out_data = self.PSE(out_data)

        return out_data


class UpCat(nn.Module):
    def __init__(self, out_channel):
        super(UpCat, self).__init__()
        self.upsample_cat = upsample_cat
        self.upconv = CFConv2d(2*out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, in_data_1, in_data_2):
        out_data = self.upsample_cat(in_data_1, in_data_2)
        out_data = self.upconv(out_data)

        return out_data


if __name__ == "__main__":
    in_data = torch.randn((2, 3, 256, 256))
    in_data_norm = torch.randn((2, 1, 256, 256))
    net = PCNet_ISE()
    print('# PCNet:', sum(param.numel() for param in net.parameters()))
    # print('# parameters:', sum(param.numel() for param in net.parameters()))
    # for name, param in net.named_parameters():
    #     print(name, '      ', param.size())
    # net_2 = VGG_19()
    # print('# VGG_19-2 parameters:', sum(param.numel() for param in net_2.parameters()))
    # net_3 = MRNet_Res34_AV()
    # print('# parameters:', sum(param.numel() for param in net_3.parameters()))
    # net_4 = MRNet_VGG16_AV()
    # print('# parameters:', sum(param.numel() for param in net_4.parameters()))
    print(net(in_data, in_data_norm)[0].size())

    # in_data_A = torch.randn((1, 3, 560, 560))
    # in_data_V = torch.randn((1, 3, 560, 560))
    # dis_net = Discriminator_E2E_AV(input_nc=6)
    # dis = Dricriminator_Normal(562)
    # print(dis_net(in_data_A, in_data_V)[0].size())
    # print(dis(in_data_A)[0].size(), dis(in_data_A)[1].size())

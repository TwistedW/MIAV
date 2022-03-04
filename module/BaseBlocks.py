# -*- coding: utf-8 -*-
# @Time    : 2019
# @Author  : Lart Pang
# @FileName: BaseBlocks.py
# @GitHub  : https://github.com/lartpang

import torch.nn as nn
import torch
# from network.DCT import get_dct_weight


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(BasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.5)
        )

    def forward(self, x):
        return self.basicconv(x)


class PCConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(PCConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(
                out_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.basicconv(x)


class CFConv2d(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(CFConv2d, self).__init__()

        self.cfconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.cfconv(x)


class PSELayer(nn.Module):
    def __init__(self, in_chnls, ratio=16):
        # ratio = 8
        super(PSELayer, self).__init__()
        self.squeeze_1 = nn.AdaptiveAvgPool2d((1, 1))
        self.compress_1 = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation_1 = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

        self.squeeze_2 = nn.AdaptiveAvgPool2d((2, 2))
        self.compress_2 = nn.Conv2d(in_chnls, in_chnls // ratio, 2, 1, 0)
        self.excitation_2 = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

        self.squeeze_3 = nn.AdaptiveAvgPool2d((3, 3))
        self.compress_3 = nn.Conv2d(in_chnls, in_chnls // ratio, 3, 1, 0)
        self.excitation_3 = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

        self.merge = nn.Conv2d(3*in_chnls, in_chnls, 1, 1, 0)
        self.merge_BN = nn.BatchNorm2d(in_chnls)

    def forward(self, x):
        out_1 = self.squeeze_1(x)
        out_1 = self.compress_1(out_1)
        out_1 = torch.relu(out_1)
        out_1 = self.excitation_1(out_1)
        out_1 = x * torch.sigmoid(out_1)

        out_2 = self.squeeze_2(x)
        out_2 = self.compress_2(out_2)
        out_2 = torch.relu(out_2)
        out_2 = self.excitation_2(out_2)
        out_2 = x * torch.sigmoid(out_2)

        out_3 = self.squeeze_3(x)
        out_3 = self.compress_3(out_3)
        out_3 = torch.relu(out_3)
        out_3 = self.excitation_2(out_3)
        out_3 = x * torch.sigmoid(out_3)

        out = torch.cat((out_1, out_2), dim=1)
        out = torch.cat((out, out_3), dim=1)

        out = self.merge(out)
        out = self.merge_BN(out)

        return out


class ISELayer(nn.Module):
    def __init__(self, in_chnls, loc="middle", ratio=16):
        super(ISELayer, self).__init__()
        # ratio = 8  # 仅在hrf数据集下进行的测试，请及时清除
        self.loc = loc
        self.squeeze_m = nn.AdaptiveAvgPool2d((1, 1))
        self.compress_m = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation_m = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

        if self.loc == "middle":
            self.squeeze_l = nn.AdaptiveAvgPool2d((1, 1))
            self.compress_l = nn.Conv2d(in_chnls//2, in_chnls // (2 * ratio), 1, 1, 0)
            self.excitation_l = nn.Conv2d(in_chnls // (2 * ratio), in_chnls//2, 1, 1, 0)

            self.squeeze_h = nn.AdaptiveAvgPool2d((1, 1))
            self.compress_h = nn.Conv2d(2*in_chnls, 2*in_chnls // ratio, 1, 1, 0)
            self.excitation_h = nn.Conv2d(2*in_chnls // ratio, 2*in_chnls, 1, 1, 0)

            self.ml = nn.Conv2d(in_chnls, in_chnls//2, 1, 1, 0)
            self.mh = nn.Conv2d(in_chnls, 2 * in_chnls, 1, 1, 0)
            self.merge = nn.Conv2d(in_chnls//2 + in_chnls + 2 * in_chnls, in_chnls, 1, 1, 0)

        elif self.loc == "begin":
            self.squeeze_h = nn.AdaptiveAvgPool2d((1, 1))
            self.compress_h = nn.Conv2d(2 * in_chnls, 2 * in_chnls // ratio, 1, 1, 0)
            self.excitation_h = nn.Conv2d(2 * in_chnls // ratio, 2 * in_chnls, 1, 1, 0)

            self.mh = nn.Conv2d(in_chnls, 2 * in_chnls, 1, 1, 0)
            self.merge = nn.Conv2d(in_chnls + 2 * in_chnls, in_chnls, 1, 1, 0)

        elif self.loc == "end":
            self.squeeze_l = nn.AdaptiveAvgPool2d((1, 1))
            self.compress_l = nn.Conv2d(in_chnls // 2, in_chnls // (2 * ratio), 1, 1, 0)
            self.excitation_l = nn.Conv2d(in_chnls // (2 * ratio), in_chnls // 2, 1, 1, 0)

            self.ml = nn.Conv2d(in_chnls, in_chnls // 2, 1, 1, 0)
            self.merge = nn.Conv2d(in_chnls // 2 + in_chnls, in_chnls, 1, 1, 0)

        self.merge_BN = nn.BatchNorm2d(in_chnls)

    def forward(self, x):
        out_m = self.squeeze_m(x[0])
        out_m = self.compress_m(out_m)
        out_m = torch.relu(out_m)
        out_m = self.excitation_m(out_m)
        out_m = x[0] * torch.sigmoid(out_m)

        if self.loc == "middle":
            out_l = self.squeeze_l(x[1])
            out_l = self.compress_l(out_l)
            out_l = torch.relu(out_l)
            out_l = self.excitation_l(out_l)
            ml = self.ml(x[0])
            out_l = ml * torch.sigmoid(out_l)

            out_h = self.squeeze_h(x[2])
            out_h = self.compress_h(out_h)
            out_h = torch.relu(out_h)
            out_h = self.excitation_h(out_h)
            mh = self.mh(x[0])
            out_h = mh * torch.sigmoid(out_h)

            out = torch.cat((out_l, out_m), dim=1)
            out = torch.cat((out, out_h), dim=1)
            out = self.merge(out)
            out = self.merge_BN(out)

        elif self.loc == "begin":
            out_h = self.squeeze_h(x[1])
            out_h = self.compress_h(out_h)
            out_h = torch.relu(out_h)
            out_h = self.excitation_h(out_h)
            mh = self.mh(x[0])
            out_h = mh * torch.sigmoid(out_h)

            out = torch.cat((out_m, out_h), dim=1)
            out = self.merge(out)
            out = self.merge_BN(out)

        elif self.loc == "end":
            out_l = self.squeeze_l(x[1])
            out_l = self.compress_l(out_l)
            out_l = torch.relu(out_l)
            out_l = self.excitation_l(out_l)
            ml = self.ml(x[0])
            out_l = ml * torch.sigmoid(out_l)
            out = torch.cat((out_l, out_m), dim=1)
            out = self.merge(out)
            out = self.merge_BN(out)

        else:
            out = x[0]

        return out
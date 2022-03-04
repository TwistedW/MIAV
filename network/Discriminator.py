import math
from functools import partial
from math import log2

import torch
import torch.nn as nn
from torch.nn import Parameter


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, batch_size=64, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        SN = nn.utils.spectral_norm
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SN(nn.Conv2d(5, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SN(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SN(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SN(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 1, 4))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, input_A, input_V, input_):
        input = torch.cat((input_, input_A), dim=1)
        x = torch.cat((input, input_V), dim=1)
        out = self.l1(x)
        topo_1 = out
        out = self.l2(out)
        topo_2 = out
        out = self.l3(out)
        topo_3 = out
        out, p1 = self.attn1(out)
        out = self.l4(out)
        out, p2 = self.attn2(out)
        out = self.last(out)
        topo = [topo_1, topo_2, topo_3]

        return out.squeeze(), topo


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Discriminator_E2E_AV(nn.Module):
    def __init__(self, input_nc=5, ndf=64, n_layers=5):
        super(Discriminator_E2E_AV, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                     nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input_A, input_V, input_):
        input = torch.cat((input_, input_A), dim=1)
        input = torch.cat((input, input_V), dim=1)
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


if __name__ == "__main__":
    in_data = torch.randn((8, 3, 288, 288))
    in_data_norm = torch.randn((8, 1, 288, 288))
    net = Discriminator()
    print('# parameters:', sum(param.numel() for param in net.parameters()))
    net_2 = Discriminator_E2E_AV(3)
    print('# parameters:', sum(param.numel() for param in net_2.parameters()))
    print(net(in_data, in_data_norm, in_data_norm)[0].size())
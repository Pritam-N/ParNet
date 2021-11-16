import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict
import ipdb
from torchsummary import summary


def activation_func(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['silu', nn.SiLU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


def conv_sampler(kernel_size=1, stride=1, groups=1):
    return partial(Conv2dAuto, kernel_size=kernel_size, stride=stride, bias=False, groups=groups)


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({
        'çonv': conv(in_channels, out_channels, *args, **kwargs),
        'bn': nn.BatchNorm2d(out_channels)
    }))

def conv_van(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({
        'çonv': conv(in_channels, out_channels, *args, **kwargs),
    }))

class GlobalAveragePool2D():
    def __init__(self, keepdim=True) -> None:
        # super(GlobalAveragePool2D, self).__init__()
        self.keepdim = keepdim

    # def forward(self, inputs):
    #     return torch.mean(inputs, axis=[2, 3], keepdim=self.keepdim)

    def __call__(self, inputs, *args, **kwargs):
        return torch.mean(inputs, axis=[2, 3], keepdim=self.keepdim)


class SSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(SSEBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.globalAvgPool = GlobalAveragePool2D()

        self.norm = nn.BatchNorm2d(self.in_channels)

    def forward(self, inputs):
        bn = self.norm(inputs)
        x = self.globalAvgPool(bn)
        x = self.conv(x)
        x = self.sigmoid(x)

        z = torch.mul(bn, x)
        return z


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Downsample, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv1 = conv_bn(self.in_channels, self.out_channels, conv_sampler(kernel_size=1))
        self.conv2 = conv_bn(self.in_channels, self.out_channels, conv_sampler(kernel_size=3, stride=2))
        self.conv3 = conv_van(self.in_channels, self.out_channels, conv_sampler(kernel_size=1))
        self.globalAvgPool = GlobalAveragePool2D()
        self.activation = activation_func('silu')
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.avgpool(inputs)
        x = self.conv1(x)

        y = self.conv2(inputs)

        z = self.globalAvgPool(inputs)
        z = self.conv3(z)
        z = self.sigmoid(z)

        a = x + y
        b = torch.mul(a, z)
        c = self.activation(b)
        return c


class Fusion(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(Fusion, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.network_in_channels = 2 * self.in_channels
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))
        self.conv1 = conv_bn(self.network_in_channels, self.out_channels, conv_sampler(kernel_size=1, groups=2))
        self.conv2 = conv_bn(self.network_in_channels, self.out_channels,
                             conv_sampler(kernel_size=3, stride=2, groups=2))
        self.conv3 = conv_van(self.network_in_channels, self.out_channels, conv_sampler(kernel_size=1, groups=2))
        self.globalAvgPool = GlobalAveragePool2D()
        self.activation = activation_func('silu')
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(self.in_channels)

    def forward(self, input1, input2):

        a = torch.cat([self.bn(input1), self.bn(input2)], dim=1)

        idx = torch.randperm(a.nelement())
        a = a.view(-1)[idx].view(a.size())

        x = self.avgpool(a)
        x = self.conv1(x)

        y = self.conv2(a)

        z = self.globalAvgPool(a)

        z = self.conv3(z)
        z = self.sigmoid(z)

        a = x + y

        b = torch.mul(a, z)
        c = self.activation(b)
        return c


class Stream(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sse = nn.Sequential(SSEBlock(self.in_channels, self.out_channels))
        self.fuse = nn.Sequential(FuseBlock(self.in_channels, self.out_channels))
        self.activation = activation_func('silu')

    def forward(self, inputs):
        a = self.sse(inputs)
        b = self.fuse(inputs)
        c = a + b

        d = self.activation(c)
        return d


class FuseBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = conv_bn(self.in_channels, self.out_channels, conv_sampler(kernel_size=1))
        self.conv2 = conv_bn(self.in_channels, self.out_channels, conv_sampler(kernel_size=3, stride=1))

    def forward(self, inputs):
        a = self.conv1(inputs)
        b = self.conv2(inputs)

        c = a + b
        return c


class ParNetEncoder(nn.Module):
    def __init__(self, in_channels, block_size, depth) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.block_size = block_size
        self.depth = depth
        self.d1 = Downsample(self.in_channels, self.block_size[0])
        self.d2 = Downsample(self.block_size[0], self.block_size[1])
        self.d3 = Downsample(self.block_size[1], self.block_size[2])
        self.d4 = Downsample(self.block_size[2], self.block_size[3])
        self.d5 = Downsample(self.block_size[3], self.block_size[4])
        self.stream1 = nn.Sequential(
            *[Stream(self.block_size[1], self.block_size[1]) for _ in range(self.depth[0])]
        )

        self.stream1_downsample = Downsample(self.block_size[1], self.block_size[2])

        self.stream2 = nn.Sequential(
            *[Stream(self.block_size[2], self.block_size[2]) for _ in range(self.depth[1])]
        )

        self.stream3 = nn.Sequential(
            *[Stream(self.block_size[3], self.block_size[3]) for _ in range(self.depth[2])]
        )

        self.stream2_fusion = Fusion(self.block_size[2], self.block_size[3])
        self.stream3_fusion = Fusion(self.block_size[3], self.block_size[3])

    def forward(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)

        y = self.stream1(x)
        y = self.stream1_downsample(y)

        x = self.d3(x)

        z = self.stream2(x)
        z = self.stream2_fusion(y, z)

        x = self.d4(x)

        a = self.stream3(x)
        b = self.stream3_fusion(z, a)

        x = self.d5(b)
        return x


class ParNetDecoder(nn.Module):
    def __init__(self, in_channels, n_classes) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_channels, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return self.softmax(x)


class ParNet(nn.Module):
    def __init__(self, in_channels, n_classes, block_size=[64, 128, 256, 512, 2048], depth=[4, 5, 5]) -> None:
        super().__init__()
        self.encoder = ParNetEncoder(in_channels, block_size, depth)
        self.decoder = ParNetDecoder(block_size[-1], n_classes)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.decoder(x)

        return x


def parnet_sm(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_size=[64, 96, 192, 384, 1280])


def parnet_md(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_size=[64, 128, 256, 512, 2048])


def parnet_l(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_size=[64, 160, 320, 640, 2560])


def parnet_xl(in_channels, n_classes):
    return ParNet(in_channels, n_classes, block_size=[64, 200, 400, 800, 3200])

if __name__ == '__main__':
    model = parnet_sm(3, 4)
    summary(model.cuda(), (3, 256, 256))
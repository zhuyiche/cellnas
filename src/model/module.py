import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import time


def _isArrayLike(obj):
    """
    check if this is array like object.
    """
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class Identity_block(nn.Module):
    '''bottleneck id block'''
    def __init__(self, in_ch, out_ch, conv1x1=False):
        super(Identity_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
        )
        self.conv1x1 = conv1x1
        if conv1x1:
            self.conv_shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, padding=0),
                nn.BatchNorm2d(out_ch)
            )

        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        _shortcut = x
        if self.conv1x1:
            _shortcut = self.conv_shortcut(_shortcut)
        #else:
        #    _shortcut = self.bn(_shortcut)
        out = self.conv(x)
        #print('identity out.shape: ', out.shape)
        out = _shortcut + out
        out = self.relu(out)
        return out


class Strided_block(nn.Module):
    '''downsample featuremap between modules'''

    def __init__(self, in_ch, out_ch):
        super(Strided_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 1 ,padding=0),
            nn.BatchNorm2d(out_ch),
        )
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1, padding=0)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        _shortcut = x
        _shortcut = self.conv_shortcut(_shortcut)
        _shortcut = self.downsample(_shortcut)
        out = self.conv(x)
        #print('shortcut.shape: {}, out.shape: {}'.format(_shortcut.shape, out.shape))
        #print('stride out.shape: ', out.shape)
        out = _shortcut + out
        out = self.relu(out)
        return out


class Conv_1x1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Conv_1x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Res_Module(nn.Module):
    def __init__(self, in_ch, out_ch, block_num=9, downsample=True, conv1x1=False):
        super(Res_Module, self).__init__()
        self.block_num = block_num - 1
        if downsample:
            self.conv1 = Strided_block(in_ch, out_ch)
        else:
            self.conv1 = Identity_block(in_ch, out_ch, conv1x1=conv1x1)

        res_layers = []
        for i in range(block_num):
            res_layers.append(Identity_block(out_ch, out_ch))
        self.res_block = nn.Sequential(*res_layers)

    def forward(self, x):
        x = self.conv1(x)
        out = self.res_block(x)
        return out


class Dilated_bottleneck(nn.Module):
    """
    Dilated block without 1x1 convolution projection, structure like res-id-block
    """
    def __init__(self, channel, dilate_rate=2):
        super(Dilated_bottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, dilation=dilate_rate, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, *input):
        x = input[0]
        x_ori = x
        x = self.conv(x)
        x = x + x_ori
        x = self.relu(x)
        return x


class Dilated_with_projection(nn.Module):
    """
    Dilated block with 1x1 convolution projection for the shortcut, structure like res-conv-block
    """
    def __init__(self, channel, dilate_rate=2):
        super(Dilated_with_projection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 3, dilation=dilate_rate, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(channel, channel, 1, padding=0),
            nn.BatchNorm2d(channel),
        )
        self.relu = nn.ReLU()

    def forward(self, *input):
        x = input[0]
        x_ori = x
        x_ori = self.shortcut(x_ori)
        x = self.conv(x)
        x = x + x_ori
        x = self.relu(x)
        return x


class Dilated_Det_Module(nn.Module):
    """
    """
    def __init__(self, channel):
        super(Dilated_Det_Module, self).__init__()
        self.dilated_with_project = Dilated_with_projection(channel)
        self.dilate_bottleneck1 = Dilated_bottleneck(channel)
        self.dilate_bottleneck2 = Dilated_bottleneck(channel)

    def forward(self, *input):
        x = input[0]
        x = self.dilated_with_project(x)
        x = self.dilate_bottleneck1(x)
        x = self.dilate_bottleneck2(x)
        return x
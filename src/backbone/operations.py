import torch
import torch.nn as nn

blocks_keys = ['dil_sep_conv_3x3',
    'avg_pool_3x3',
    'max_pool_3x3',
    'identity',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7'
]
# check affine = true
blocks_dict = {
    'dil_sep_conv_3x3': lambda in_channel, out_channel, stride: DilSepConv(in_channel,out_channel,
                                                                           3, stride, 2, 2, affine=True),
    'avg_pool_3x3': lambda in_channel, out_channel, stride: nn.AvgPool2d(3, stride=stride, padding=1,
                                                                         count_include_pad=False),
    'max_pool_3x3': lambda in_channel, out_channel, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    'identity': lambda in_channel, out_channel, stride: Identity() if stride == 1 else FactorizedReduce(C_in=in_channel,
                                                                                                        C_out=out_channel,
                                                                                                        affine=True),
    'sep_conv_3x3': lambda in_channel, out_channel, stride: SepConv(in_channel,out_channel,
                                                                    3, stride, 1, affine=True),
    'sep_conv_5x5': lambda in_channel, out_channel, stride: SepConv(in_channel,out_channel,
                                                                    5, stride, 2, affine=True),
    'sep_conv_7x7': lambda in_channel, out_channel, stride: SepConv(in_channel,out_channel,
                                                                    7, stride, 3, affine=True),
}

node_types = ['add-copy', 'concat-split', 'concat-copy']


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        return self.op(x)


class DilSepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilSepConv, self).__init__()
        self.op = nn.Sequential(
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding,
                      dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    self.op = nn.Sequential(
        nn.BatchNorm2d(C_in, affine=affine),
        nn.ReLU(),
        nn.Conv2d(C_in, C_out, 1, stride=2, padding=0, bias=False)
    )

  def forward(self, x):
    out = self.op(x)
    #out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    return out
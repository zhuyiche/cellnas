import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from .module import Conv_1x1, Res_Module


class CellDet(nn.Module):
    def __init__(self, stage=4, block_num=9, channels=[32, 64, 128, 256]):
        super(CellDet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        #self.conv2 = Conv_1x1(64, 2)
        #self.conv3 = Conv_1x1(128, 2)
        self.convstage4 = Conv_1x1(256,256)


        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0),
            nn.BatchNorm2d(64)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(256, 256, 1)
        self.final_conv = nn.Conv2d(32, 2, kernel_size=3, padding=1)

        self.module1 = Res_Module(32, 32, block_num=block_num, downsample=False)
        self.module2 = Res_Module(32, 64, block_num=block_num, downsample=True)
        self.module3 = Res_Module(64, 128, block_num=block_num, downsample=True)
        self.module4 = Res_Module(128, 256, block_num=block_num, downsample=True)
        self.out = nn.LogSoftmax(dim=1)

    def forward(self, x):
        #print('input.shape: ', x.shape)
        x = self.conv1(x)
        #print()
        #print('conv1.shape: ', x.shape)
        x = self.module1(x)
        #print('module1.shape: ',x.shape)
        x = self.module2(x)
        #print('module2.shape: ', x.shape)
        x = self.module3(x)
        #print('module3.shape: ', x.shape)
        x = self.module4(x)
        #print('module4.shape: ', x.shape)
        x = self.conv4(x)
        x = self.deconv1(x)
        #print('deconv1.shape: ', x.shape)
        x = self.deconv2(x)
        #print('deconv2.shape: ', x.shape)
        x = self.deconv3(x)
        #print('deconv3.shape: ', x.shape)
        x = self.final_conv(x)
        x = F.softmax(x, dim=1)
        #x = x[:, 1, :, :]
        #x = torch.transpose(x, (x.shape[0], x.shape[1], 500, 500))
        #print('out.shape: ', x.shape)
        return x
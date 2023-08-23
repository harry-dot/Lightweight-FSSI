import torch
import torch.nn as nn
import torch.nn.functional as F
import os,sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from main.utils import euclidean_dist
from torch.autograd import Variable

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class Statistic_pooling(nn.Module):
    def __init__(self):
        super(Statistic_pooling, self).__init__()

    def forward(self, x):
        mean_x = x.mean(dim = 2)
        std_x = x.std(dim = 2)
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std

class Conv_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Conv_block, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
    def forward(self, x):
        x = self.conv_block(x)
        # print(x.size())
        return x

class Fc_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Fc_block, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.fc_layer(x)

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.model_name = 'ProtNet'
        self.x_dim = 1
        self.hid_dim = 64
        self.z_dim = 64
        # self.pooling = Statistic_pooling()
        self.encoder = nn.Sequential(
        Conv_block(80, 128),
        Conv_block(128, 128),
        Conv_block(128, 128),
        Conv_block(128, 128)
        )
        self.flatten = Flatten()
        # self.fc = Fc_block(15360, 512)

    def forward(self, x, xavg=0, xstd=0):
    # def forward(self, x):
    #     x=x.float()
        # print(x.size())
        #zx = (x - xavg) / xstd  # (130, 1, 128, 160)输入数据归一化处理
        zx = x
        # zx=x.float()
        zx = self.encoder(zx)
        # print(zx.size())
        # zx = self.pooling(zx)
        zx = self.flatten(zx)
        # print(zx.size())
        # output = self.fc(zx)
        return zx

if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim
    x = torch.zeros(5, 80, 699)
    model = net()
    out = model(x)
    print(model)
    print(out.shape)  # should be [2, 192]
"""
blstm + statistics pooling + shortcut + all_dimension=64 + dialation=1+ groups=4

"""
import torch
import torch.nn as nn
import math

"""
blstm+statistics pooling
"""
fr = 500
def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )

class GhostModule(nn.Module):
    def __init__(self, inp, oup, dialation, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, fmap_order=None):
        super(GhostModule, self).__init__()
        self.fmap_order = fmap_order
        self.oup = oup
        init_channels = int(math.ceil(oup / ratio)) #math.ceil 向上取整
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv1d(inp, init_channels, kernel_size, stride, kernel_size//2, dilation=dialation, bias=False),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv1d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm1d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)

        return out

class Statistic_pooling(nn.Module):
    def __init__(self):
        super(Statistic_pooling, self).__init__()

    def forward(self, x):
        mean_x = x.mean(dim = 2)
        std_x = x.std(dim = 2)
        mean_std = torch.cat((mean_x, std_x), 1)
        return mean_std


class GConv1_block(nn.Module):
    def __init__(self,in_channels, out_channels, groups):
        super(GConv1_block, self).__init__()
        self.groups = groups
        self.gconv1_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, groups=self.groups),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            # nn.MaxPool1d(2)
        )
    def forward(self, x):
        x = self.gconv1_block(x)
        # print(x.size())
        return x

class BiLSTM(nn.Module):
    def __init__(self, in_channels, hidden_size, bidirectional):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional)

    def forward(self, x):
        return self.lstm(x)

class ESPGnet(nn.Module):
    def __init__(self, channel_num):
        super(ESPGnet, self).__init__()
        self.model_name = 'GC'
        # self.blstm1 = BiLSTM(20, 20, bidirectional=True)
        # self.blstm2 = BiLSTM(20, 20, bidirectional=True)
        # self.blstm3 = BiLSTM(20, 20, bidirectional=True)
        # self.blstm4 = BiLSTM(20, 20, bidirectional=True)
        # self.blstm2 = BiLSTM(20, 20, bidirectional=True)
        # self.blstm3 = BiLSTM(20, 20, bidirectional=True)
        # self.blstm4 = BiLSTM(20, 20, bidirectional=True)
        self.channel_num = channel_num
        self.pooling = Statistic_pooling()
        # self.ghost1_1 = GhostModule(40, 64, 1)
        # self.ghost1_2 = GhostModule(40, 64, 1)
        # self.ghost1_3 = GhostModule(40, 64, 1)
        # self.ghost1_4 = GhostModule(40, 64, 1)
        self.GConv = nn.Conv1d(in_channels=80, out_channels=512, kernel_size=3, groups=4, padding=1)
        # self.GConv2 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, groups=4, padding=1)
        # self.GConv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, groups=4, padding=1)
        self.shortcut = nn.Sequential(
            nn.Conv1d(80, 512, 1),
            nn.BatchNorm1d(512))

    def group_divided(self, inp, groups):
        channel = inp.size()[1]
        group_channel = int(channel/groups)
        output = []
        for i in range(groups):
            output.append(inp[:, i*group_channel:(i+1)*group_channel, :])
        return output

    def forward(self, x, xavg, xstd):
    # def forward(self, x):
    #     zx = x.float()
    #
        zx = (x - xavg) / xstd
        # zx=zx.float()
        # gc1, gc2, gc3, gc4 = [self.group_divided(zx, self.channel_num)[i].view(-1,fr,20) for i in range(len(self.group_divided(zx, self.channel_num)))]
        #
        # gc1,_ = self.blstm1(gc1)
        # gc2,_ = self.blstm2(gc2)
        # gc3,_ = self.blstm3(gc3)
        # gc4,_ = self.blstm4(gc4)

        # gc1 = gc1.view(-1, 40, fr)
        # gc2 = gc2.view(-1, 40, fr)
        # gc3 = gc3.view(-1, 40, fr)
        # gc4 = gc4.view(-1, 80, fr)
        # print(gc1.size())
        # fea = torch.cat((gc1, gc2, gc3, gc4), 1)
        fea1 = self.GConv(zx)
        # fea1 = self.GConv2(fea1)
        # fea1 = self.GConv2(fea1)
        # fea1 = self.GConv3(fea1)
        # fea1 = self.GConv2(fea1)
        # print(fea1.size())
        shortcut = self.shortcut(x)
        # print(shortcut.size())
        total = fea1+shortcut

        embedding = self.pooling(total)
        return embedding
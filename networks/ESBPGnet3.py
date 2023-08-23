"""
blstm + statistics pooling + shortcut + all_dimension=64 + dialation=1+ groups=4

"""
import torch
import torch.nn as nn
import math
import numpy as np
from .transformer import *
"""
blstm+statistics pooling
"""
fr = 699 # why 699
input_dim = 80
class CCALayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CCALayer, self).__init__()

        #self.contrast = torch.std
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_du = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def forward(self, x):
        y = torch.std(x,dim=-1,keepdim=True) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
def depthwise_conv(inp, oup, kernel_size=3, stride=1, relu=False):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) if relu else nn.Sequential(),
    )
class BSConv(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size=3,stride=1,padding=0,dilation=1,bias=False):
 
        #这一行千万不要忘记
        super(BSConv, self).__init__()
 
        # 逐通道卷积
        self.depth_conv = nn.Conv1d(in_channels=out_channel,
                                    out_channels=out_channel,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    groups=out_channel,bias=bias)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
 
        #逐点卷积
        self.point_conv = nn.Conv1d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    bias=bias)
    
    def forward(self,input):
        out = self.point_conv(input)
        out = self.depth_conv(out)
        return out

class GhostModule(nn.Module):
    def __init__(self, inp, oup, dialation, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, fmap_order=None):
        super(GhostModule, self).__init__()
        self.fmap_order = fmap_order
        self.oup = oup
        init_channels = int(math.ceil(oup / ratio)) #math.ceil 向上取整
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv1d(inp, init_channels, kernel_size, stride, kernel_size//2, dilation=dialation, bias=False),
            #BSConv(inp,init_channels,kernel_size,stride,kernel_size//2,dilation=dialation),
            nn.BatchNorm1d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            #nn.Conv1d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            BSConv(init_channels, new_channels, dw_size, 1, dw_size//2, bias=False),
            nn.BatchNorm1d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        # print(x1.size())
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
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size, num_layers=1, batch_first=True,bidirectional=bidirectional)

    def forward(self, x):
        return self.lstm(x)
class ESPGnet_blstm(nn.Module):
    def __init__(self, channel_num):
        super(ESPGnet_blstm, self).__init__()
        self.model_name = 'ESPGnet'
        self.num_node = input_dim//channel_num           # 20
        #self.blstm = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.blstm1 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.blstm2 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.blstm3 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.blstm4 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.channel_num = channel_num                  # 4
        self.pooling = Statistic_pooling()
        #self.ghost = GhostModule(2*self.num_node, 64, 1)
        #self.ghost = GhostModule(self.num_node, 64, 1)
        self.ghost1_1 = GhostModule(2*self.num_node, 64, 1)
        self.ghost1_2 = GhostModule(2*self.num_node, 64, 1)
        self.ghost1_3 = GhostModule(2*self.num_node, 64, 1)
        self.ghost1_4 = GhostModule(2*self.num_node, 64, 1)

        self.shortcut = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1),
            nn.BatchNorm1d(256))

    def group_divided(self, inp, groups):
        channel = inp.size()[1]
        group_channel = int(channel/groups)
        output = []
        for i in range(groups):
            output.append(inp[:, i*group_channel:(i+1)*group_channel, :])
        return output

    #def forward(self, x, xavg, xstd):
    def forward(self, x):
        zx=x.float().transpose(1,2)        # x.shape = [50,80,699]
        #zx=x.float()
        #zx = (zx - xavg) / xstd
        # print(zx.size())
        #gc1, gc2, gc3, gc4 = [self.group_divided(zx, self.channel_num)[i].view(-1,fr,20) for i in range(len(self.group_divided(zx, self.channel_num)))]
        # gci.shape = [50,699,20]
        # print(gc1.shape)
        # print(gc1.size())
        #print(self.num_node)

        gc1,gc2,gc3,gc4 = torch.split(zx,self.num_node, dim=2)
        #gc = list(torch.split(zx,self.num_node, dim=1))     
        # print(np.array(gc).shape)
                #print(feature.size())
        # print(gc.size())
            # print(gs_i.size())
            # if i == 0:
            #     gs = gs_i
                # print(gs.size())
                # avg = gs_i
            # else:
            #     gs = torch.cat((gs, gs_i), dim=1)
            #     avg += gs_i


        #avg = avg//self.channel_num
        #for i in range(self.channel_num):
        #    print(avg.size())
        #    print(gs[:,i*64:(i+1)*64,:].size())
        #    gs = gs[:i*64:(i+1)*64,:]+avg
        gc1,_ = self.blstm1(gc1)
        #print(type(gc1))
        gc2,_ = self.blstm2(gc2)
        gc3,_ = self.blstm3(gc3)
        gc4,_ = self.blstm4(gc4)

        #gc1 = gc1.contiguous().view(-1, 2*self.num_node, fr)
        #gc2 = gc2.contiguous().view(-1, 2*self.num_node, fr)
        #gc3 = gc3.contiguous().view(-1, 2*self.num_node, fr)
        #gc4 = gc4.contiguous().view(-1, 2*self.num_node, fr)
        gc1 = gc1.reshape(-1, 2*self.num_node, fr)
        gc2 = gc2.reshape(-1, 2*self.num_node, fr)
        gc3 = gc3.reshape(-1, 2*self.num_node, fr)
        gc4 = gc4.reshape(-1, 2*self.num_node, fr)
        #print(gc1.size())

        gs1_1 = self.ghost1_1(gc1)
        gs1_2 = self.ghost1_2(gc2)
        gs1_3 = self.ghost1_3(gc3)
        gs1_4 = self.ghost1_4(gc4)
        #print(gs1_1.size())
        avg = (gs1_1+gs1_2+gs1_3+gs1_4)/4
        #print('avg',avg.size())
        gs1 = gs1_1 + avg
        gs2 = gs1_2 + avg
        gs3 = gs1_3 + avg
        gs4 = gs1_4 + avg
        # print(gs1.size())
        feature = torch.cat((gs1, gs2, gs3, gs4), 1)
        #print(fea1.size())
        #input = x.transpose(1,2)
        input = x
        shortcut = self.shortcut(input)
        total = feature+shortcut
        #
        embedding = self.pooling(total)
        #print(embedding.size())
        # print('--------')
        return embedding
class ESPGnet(nn.Module):
    def __init__(self, channel_num):
        super(ESPGnet, self).__init__()
        self.model_name = 'ESPGnet'
        self.num_node = input_dim//channel_num           # 20
        self.transformer = Encoder(n_head=2,d_model=self.num_node,d_inner=self.num_node//2,n_position=699)
        #self.transformer = FastEncoder(n_head=2,d_model=self.num_node,d_inner=self.num_node//2,n_position=699)
        #self.blstm = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm1 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm2 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm3 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm4 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.channel_num = channel_num                  # 4
        self.pooling = Statistic_pooling()
        #self.ghost = GhostModule(2*self.num_node, 64, 1)
        self.ghost = GhostModule(self.num_node, 256//channel_num, 1)
        #self.ghost1_1 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_2 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_3 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_4 = GhostModule(2*self.num_node, 64, 1)

        self.shortcut = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1),
            nn.BatchNorm1d(256))

    def group_divided(self, inp, groups):
        channel = inp.size()[1]
        group_channel = int(channel/groups)
        output = []
        for i in range(groups):
            output.append(inp[:, i*group_channel:(i+1)*group_channel, :])
        return output

    #def forward(self, x, xavg, xstd):
    def forward(self, x):
        zx=x.float().transpose(1,2)        # x.shape = [50,80,699]
        #seq_len = zx.shape[-1]

        #zx=x.float()
        #zx = (zx - xavg) / xstd
        # print(zx.size())
        #gc1, gc2, gc3, gc4 = [self.group_divided(zx, self.channel_num)[i].view(-1,fr,20) for i in range(len(self.group_divided(zx, self.channel_num)))]
        # gci.shape = [50,699,20]
        # print(gc1.shape)
        # print(gc1.size())
        #print(self.num_node)

        gc = list(torch.split(zx,self.num_node, dim=2))
        #gc = list(torch.split(zx,self.num_node, dim=1))     
        # print(np.array(gc).shape)
        gs_total = 0
        gs = []

        for i in range(self.channel_num):
            # print(len(gc[i]))
            # gc_i = np.array(gc[i])
            # print(type(gc[i]))
            #gc_i,_ = self.blstm(gc[i])
            gc_i = self.transformer(gc[i])
            gs_i = self.ghost(gc_i.transpose(1,2))#
            gs.append(gs_i.unsqueeze(1))
            gs_total += gs_i
        gs_avg = gs_total/self.channel_num

        gs_std = torch.std(torch.cat(gs,dim=1),dim=1)

        #gs_max = torch.max(torch.cat(gs,dim=1),dim=1)[0]
        
        for i in range(self.channel_num):
            if i==0:
                feature = gs[0].squeeze(1)+gs_avg+gs_std
                #feature = gs[0]+gs_avg
                #feature = (gs[0]-gs_avg)/gs_std
            else:
                gs_i = gs[i].squeeze(1)+gs_avg+gs_std
                #gs_i = gs[i]+gs_avg
                #gs_i = (gs[i]-gs_avg)/gs_std
                feature = torch.cat((feature, gs_i), dim=1)
                #print(feature.size())
        # print(gc.size())
            # print(gs_i.size())
            # if i == 0:
            #     gs = gs_i
                # print(gs.size())
                # avg = gs_i
            # else:
            #     gs = torch.cat((gs, gs_i), dim=1)
            #     avg += gs_i


        #avg = avg//self.channel_num
        #for i in range(self.channel_num):
        #    print(avg.size())
        #    print(gs[:,i*64:(i+1)*64,:].size())
        #    gs = gs[:i*64:(i+1)*64,:]+avg
        #gc1,_ = self.blstm1(gc1)
        #print(type(gc1))
        #gc2,_ = self.blstm2(gc2)
        #gc3,_ = self.blstm3(gc3)
        #gc4,_ = self.blstm4(gc4)

        #gc1 = gc1.view(-1, 2*self.num_node, fr)
        #gc2 = gc2.view(-1, 2*self.num_node, fr)
        #gc3 = gc3.view(-1, 2*self.num_node, fr)
        #gc4 = gc4.view(-1, 2*self.num_node, fr)
        #print(gc1.size())

        #gs1_1 = self.ghost1_1(gc1)
        #gs1_2 = self.ghost1_2(gc2)
        #gs1_3 = self.ghost1_3(gc3)
        #gs1_4 = self.ghost1_4(gc4)
        #print(gs1_1.size())
        #avg = (gs1_1+gs1_2+gs1_3+gs1_4)/4
        #print('avg',avg.size())
        #gs1 = gs1_1 + avg
        #gs2 = gs1_2 + avg
        #gs3 = gs1_3 + avg
        #gs4 = gs1_4 + avg
        # print(gs1.size())
        #fea1 = torch.cat((gs1, gs2, gs3, gs4), 1)
        #print(fea1.size())
        #input = x.transpose(1,2)
        input = x
        shortcut = self.shortcut(input)
        total = feature+shortcut
        #
        embedding = self.pooling(total)
        #print(embedding.size())
        # print('--------')
        return embedding
    
class ESPGnet_branch(nn.Module):
    def __init__(self, channel_num):
        super(ESPGnet_branch, self).__init__()
        self.model_name = 'ESPGnet'
        self.num_node = input_dim//channel_num           # 20
        self.transformer = Encoder(n_head=2,d_model=self.num_node,d_inner=self.num_node//2,n_position=699)
        self.project = nn.Linear(self.num_node,64)
        self.channel_num = channel_num                  # 4
        self.pooling = Statistic_pooling()
        #self.ghost = nn.Sequential(GhostModule(self.num_node, 64, 1),GhostModule(64,self.num_node,2))
        self.ghost = GhostModule(self.num_node, 64, 1)
        self.shortcut = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1),
            nn.BatchNorm1d(256))

    def group_divided(self, inp, groups):
        channel = inp.size()[1]
        group_channel = int(channel/groups)
        output = []
        for i in range(groups):
            output.append(inp[:, i*group_channel:(i+1)*group_channel, :])
        return output

    #def forward(self, x, xavg, xstd):
    def forward(self, x):
        zx=x.float().transpose(1,2)        # x.shape = [50,80,699]

        gc = list(torch.split(zx,self.num_node, dim=2))
        #gc = list(torch.split(zx,self.num_node, dim=1))     
        # print(np.array(gc).shape)
        gs_total = 0
        gs = []
        for i in range(self.channel_num):
            gc_i = self.project(self.transformer(gc[i])).transpose(1,2)
            gs_i = self.ghost(gc[i].transpose(1,2))#
            #gs_i = self.ghost2(gs_i)
            ga_i = gc_i*gs_i
            #ga_i = (gs_i + gc_i)/2
            gs.append(ga_i)
            gs_total += ga_i
        gs_avg = gs_total/self.channel_num

        gs_std = torch.std(torch.cat(gs,dim=0),dim=0,keepdim=True)

        #feature = torch.cat(gs,dim=1)

        #feature = self.merge(feature)
        
        for i in range(self.channel_num):
            if i==0:
                feature = gs[0]+gs_avg+gs_std
                #feature = gs[0]+gs_avg
                #feature = (gs[0]-gs_avg)/gs_std
            else:
                gs_i = gs[i]+gs_avg+gs_std
                #gs_i = gs[i]+gs_avg
                #gs_i = (gs[i]-gs_avg)/gs_std
                feature = torch.cat((feature, gs_i), dim=1)
                #print(feature.size())
        input = x
        shortcut = self.shortcut(input)
        total = feature+shortcut
        #
        embedding = self.pooling(total)

        return embedding
class ESPGnet_OG(nn.Module):
    def __init__(self, channel_num):
        super(ESPGnet_OG, self).__init__()
        self.model_name = 'ESPGnet'
        self.num_node = input_dim//channel_num           # 20
        self.blstm = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm1 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm2 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm3 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm4 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.channel_num = channel_num                  # 4
        self.pooling = Statistic_pooling()
        #self.ghost = GhostModule(2*self.num_node, 64, 1)
        self.ghost = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_1 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_2 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_3 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_4 = GhostModule(2*self.num_node, 64, 1)
        
        self.shortcut = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1),
            nn.BatchNorm1d(256))

    def group_divided(self, inp, groups):
        channel = inp.size()[1]
        group_channel = int(channel/groups)
        output = []
        for i in range(groups):
            output.append(inp[:, i*group_channel:(i+1)*group_channel, :])
        return output

    #def forward(self, x, xavg, xstd):
    def forward(self, x):
        zx=x.float().transpose(1,2)        # x.shape = [50,80,699]
        #seq_len = zx.shape[-1]

        #zx=x.float()
        #zx = (zx - xavg) / xstd
        # print(zx.size())
        #gc1, gc2, gc3, gc4 = [self.group_divided(zx, self.channel_num)[i].view(-1,fr,20) for i in range(len(self.group_divided(zx, self.channel_num)))]
        # gci.shape = [50,699,20]
        # print(gc1.shape)
        # print(gc1.size())
        #print(self.num_node)

        gc = list(torch.split(zx,self.num_node, dim=2))
        #gc = list(torch.split(zx,self.num_node, dim=1))     
        # print(np.array(gc).shape)
        gs_total = 0
        gs = []

        for i in range(self.channel_num):

            gc_i,_ = self.blstm(gc[i])
      
            gs_i = self.ghost(gc_i.transpose(1,2))#
            gs.append(gs_i)
            gs_total += gs_i
        gs_avg = gs_total/self.channel_num

    

        
        for i in range(self.channel_num):
            if i==0:
                feature = gs[0]+gs_avg
                #feature = gs[0]+gs_avg
                #feature = (gs[0]-gs_avg)/gs_std
            else:
                gs_i = gs[i]+gs_avg
                #gs_i = gs[i]+gs_avg
                #gs_i = (gs[i]-gs_avg)/gs_std
                feature = torch.cat((feature, gs_i), dim=1)

        input = x
        shortcut = self.shortcut(input)
        total = feature+shortcut
        #
        embedding = self.pooling(total)
        #print(embedding.size())
        # print('--------')
        return embedding

class ESPGnet_V2(nn.Module):
    def __init__(self, channel_num):
        super(ESPGnet_V2, self).__init__()
        self.model_name = 'ESPGnet'
        self.num_node = input_dim//channel_num           # 20
        self.transformer = Encoder(n_head=2,d_model=64,d_inner=64//2,n_position=699)
        #self.blstm = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm1 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm2 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm3 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm4 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.channel_num = channel_num                  # 4
        self.pooling = Statistic_pooling()
        #self.ghost = GhostModule(2*self.num_node, 64, 1)
        self.ghost = GhostModule(self.num_node, 64, 1)
        #self.ghost1_1 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_2 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_3 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_4 = GhostModule(2*self.num_node, 64, 1)

        self.shortcut = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1),
            nn.BatchNorm1d(256))

    def group_divided(self, inp, groups):
        channel = inp.size()[1]
        group_channel = int(channel/groups)
        output = []
        for i in range(groups):
            output.append(inp[:, i*group_channel:(i+1)*group_channel, :])
        return output

    #def forward(self, x, xavg, xstd):
    def forward(self, x):
        zx=x.float().transpose(1,2)        # x.shape = [50,80,699]
        
        gc = list(torch.split(zx,self.num_node, dim=2))

        gs_total = 0
        gs = []

        for i in range(self.channel_num):
            # print(len(gc[i]))
            # gc_i = np.array(gc[i])
            # print(type(gc[i]))
            #gc_i,_ = self.blstm(gc[i])
            
            gc_i = self.ghost(gc[i].transpose(1,2)).transpose(1,2)#
            gs_i = self.transformer(gc_i).transpose(1,2)

            gs.append(gs_i)
            gs_total += gs_i
        gs_avg = gs_total/self.channel_num

        gs_std = torch.std(torch.cat(gs,dim=0),dim=0,keepdim=True)

        
        for i in range(self.channel_num):
            if i==0:
                feature = gs[0]+gs_avg+gs_std
                #feature = gs[0]+gs_avg
                #feature = (gs[0]-gs_avg)/gs_std
            else:
                gs_i = gs[i]+gs_avg+gs_std
                #gs_i = gs[i]+gs_avg
                #gs_i = (gs[i]-gs_avg)/gs_std
                feature = torch.cat((feature, gs_i), dim=1)


        input = x
        shortcut = self.shortcut(input)
        total = feature+shortcut
        #
        embedding = self.pooling(total)
        #print(embedding.size())
        # print('--------')
        return embedding

class ESPGnet_CCA(nn.Module):
    def __init__(self, channel_num):
        super(ESPGnet_CCA, self).__init__()
        self.model_name = 'ESPGnet'
        self.num_node = input_dim//channel_num           # 20
        self.transformer = Encoder(n_head=2,d_model=self.num_node,d_inner=self.num_node//2,n_position=699)
        #self.blstm = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm1 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm2 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm3 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm4 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.channel_num = channel_num                  # 4
        self.pooling = Statistic_pooling()
        #self.ghost = GhostModule(2*self.num_node, 64, 1)
        self.ghost = GhostModule(self.num_node, 64, 1)
        #self.ghost1_1 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_2 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_3 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_4 = GhostModule(2*self.num_node, 64, 1)
        self.CCA = CCALayer(channel = 64*self.channel_num)
        self.shortcut = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1),
            nn.BatchNorm1d(256))

    def group_divided(self, inp, groups):
        channel = inp.size()[1]
        group_channel = int(channel/groups)
        output = []
        for i in range(groups):
            output.append(inp[:, i*group_channel:(i+1)*group_channel, :])
        return output

    #def forward(self, x, xavg, xstd):
    def forward(self, x):
        zx=x.float().transpose(1,2)        # x.shape = [50,80,699]


        gc = list(torch.split(zx,self.num_node, dim=2))
 
        
        gs = []

        for i in range(self.channel_num):

            gc_i = self.transformer(gc[i])
            gs_i = self.ghost(gc_i.transpose(1,2))#
            gs.append(gs_i)
        
        feature = torch.cat(gs,dim=1)
        feature = self.CCA(feature)
        
        input = x
        shortcut = self.shortcut(input)
        total = feature+shortcut
        #
        embedding = self.pooling(total)

        return embedding

class ESPGnet_subband(nn.Module):
    def __init__(self, channel_num):
        super(ESPGnet_subband, self).__init__()
        self.model_name = 'ESPGnet'
        self.num_node = input_dim//channel_num           # 20
        if channel_num == 16:
            head = 1
        else:
            head = 2
        self.transformer = Encoder(n_head=head,d_model=self.num_node,d_inner=self.num_node//2,n_position=699)
        #self.transformer = FastEncoder(n_head=2,d_model=self.num_node,d_inner=self.num_node//2,n_position=699)
        #self.blstm = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm1 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm2 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm3 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm4 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.channel_num = channel_num                  # 4
        self.pooling = Statistic_pooling()
        self.hidden = 256//channel_num
        #self.pooling = AttentionPooling(256)
        #self.ghost = GhostModule(2*self.num_node, 64, 1)
        self.ghost = GhostModule(self.num_node, self.hidden, 1)
        #self.ghost1_1 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_2 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_3 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_4 = GhostModule(2*self.num_node, 64, 1)
        #self.map = nn.Sequential(nn.Conv1d(128,64,1,bias=False),nn.ReLU())
        #self.map = nn.Conv1d(2*self.hidden,self.hidden,1,bias=False) if self.channel_num!=1 else 0
        self.map = nn.Conv1d(2*self.hidden,self.hidden,1,bias=False)
        #self.shortcut = nn.Sequential(
        #    nn.Conv1d(input_dim, 256, 1),
        #    nn.BatchNorm1d(256))

    def group_divided(self, inp, groups):
        channel = inp.size()[1]
        group_channel = int(channel/groups)
        output = []
        for i in range(groups):
            output.append(inp[:, i*group_channel:(i+1)*group_channel, :])
        return output

    #def forward(self, x, xavg, xstd):
    def forward(self, x,is_draw=False):
        zx=x.float().transpose(1,2)        # x.shape = [50,80,699]


        gc = list(torch.split(zx,self.num_node, dim=2))

        gs_total = 0
        gs = []

        for i in range(self.channel_num):

            gc_i = self.transformer(gc[i])
            gs_i = self.ghost(gc_i.transpose(1,2))#
            gs.append(gs_i.unsqueeze(1))
            gs_total += gs_i
        gs_avg = gs_total/self.channel_num
        if self.channel_num == 1:
            gs_std=0
        else:
            gs_std = torch.std(torch.cat(gs,dim=1),dim=1)

        #gs_max = torch.max(torch.cat(gs,dim=0),dim=0,keepdim=True)

        #if self.channel_num != 1:
        for i in range(self.channel_num):
            if i==0:
                feature = self.map(torch.cat([gs[0].squeeze(1),gs_avg+gs_std],dim=1))+gs[0].squeeze(1)
                        #feature = gs[0]+gs_avg+gs_std
                        #feature = gs[0]+gs_avg
                        #feature = (gs[0]-gs_avg)/gs_std
            else:
                gs_i = self.map(torch.cat([gs[i].squeeze(1),gs_avg+gs_std],dim=1))+gs[i].squeeze(1)
                        #gs_i = gs[i]+gs_avg
                        #gs_i = (gs[i]-gs_avg)/gs_std
                feature = torch.cat((feature, gs_i), dim=1)
                    #print(feature.size())
        #else:
            #feature = gs[0].squeeze(1)



        #input = x
        #shortcut = self.shortcut(input)
        #total = feature+shortcut
        #
        #embedding = self.pooling(total)
        embedding = self.pooling(feature)
        if not is_draw:
            return embedding
        else:
            return embedding,gs,feature

class ESPGnet_map(nn.Module):
    def __init__(self, channel_num):
        super(ESPGnet_map, self).__init__()
        self.model_name = 'ESPGnet'
        self.num_node = input_dim//channel_num           # 20
        self.transformer = Encoder(n_head=2,d_model=self.num_node,d_inner=self.num_node//2,n_position=699)
        #self.transformer = FastEncoder(n_head=2,d_model=self.num_node,d_inner=self.num_node//2,n_position=699)
        #self.blstm = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm1 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm2 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm3 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        #self.blstm4 = BiLSTM(self.num_node, self.num_node, bidirectional=True)
        self.channel_num = channel_num                  # 4
        self.pooling = Statistic_pooling()
        #self.ghost = GhostModule(2*self.num_node, 64, 1)
        self.ghost = GhostModule(self.num_node, 64, 1)

        self.map = nn.Conv1d(64,64,1,bias=False)
        #self.ghost1_1 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_2 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_3 = GhostModule(2*self.num_node, 64, 1)
        #self.ghost1_4 = GhostModule(2*self.num_node, 64, 1)

        self.shortcut = nn.Sequential(
            nn.Conv1d(input_dim, 256, 1),
            nn.BatchNorm1d(256))

    def group_divided(self, inp, groups):
        channel = inp.size()[1]
        group_channel = int(channel/groups)
        output = []
        for i in range(groups):
            output.append(inp[:, i*group_channel:(i+1)*group_channel, :])
        return output

    #def forward(self, x, xavg, xstd):
    def forward(self, x):
        zx=x.float().transpose(1,2)        # x.shape = [50,80,699]
        #seq_len = zx.shape[-1]

        #zx=x.float()
        #zx = (zx - xavg) / xstd
        # print(zx.size())
        #gc1, gc2, gc3, gc4 = [self.group_divided(zx, self.channel_num)[i].view(-1,fr,20) for i in range(len(self.group_divided(zx, self.channel_num)))]
        # gci.shape = [50,699,20]
        # print(gc1.shape)
        # print(gc1.size())
        #print(self.num_node)

        gc = list(torch.split(zx,self.num_node, dim=2))
        #gc = list(torch.split(zx,self.num_node, dim=1))     
        # print(np.array(gc).shape)
        gs_total = 0
        gs = []

        for i in range(self.channel_num):
            # print(len(gc[i]))
            # gc_i = np.array(gc[i])
            # print(type(gc[i]))
            #gc_i,_ = self.blstm(gc[i])
            gc_i = self.transformer(gc[i])
            gs_i = self.ghost(gc_i.transpose(1,2))#
            gs.append(gs_i.unsqueeze(1))
            gs_total += gs_i
        gs_avg = gs_total/self.channel_num

        gs_std = torch.std(torch.cat(gs,dim=1),dim=1)

        s = self.map(gs_std+gs_avg)
        #gs_max = torch.max(torch.cat(gs,dim=1),dim=1)[0]
        
        for i in range(self.channel_num):
            if i==0:
                feature = gs[0].squeeze(1)+s
                #feature = gs[0]+gs_avg
                #feature = (gs[0]-gs_avg)/gs_std
            else:
                gs_i = gs[i].squeeze(1)+s
                #gs_i = gs[i]+gs_avg
                #gs_i = (gs[i]-gs_avg)/gs_std
                feature = torch.cat((feature, gs_i), dim=1)
                #print(feature.size())

        input = x
        shortcut = self.shortcut(input)
        total = feature+shortcut
        #
        embedding = self.pooling(total)
        #print(embedding.size())
        # print('--------')
        return embedding
if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim
    x = torch.zeros(50, 700, 80)
    model = ESPGnet_CCA(channel_num=4)
    out = model(x)
    print(model)
    # print(out.shape)  # should be [2, 192]
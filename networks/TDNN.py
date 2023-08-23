import torch
import torch.nn as nn
import torch.nn.functional as F


class TDNN(nn.Module):

    def __init__(self, din, dout, kernel, dilation, padding=0, keep_prob=1):
        super(TDNN, self).__init__()

        self.conv1 = nn.Conv1d(din, dout, kernel, 1, padding, dilation)
        self.bn = nn.BatchNorm1d(dout, eps=1e-3, momentum=0.95, affine=False)
        self.dropout = nn.Dropout(p=keep_prob)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.bn(out)
        # out = self.dropout(out)

        return out


class StatsPooling(nn.Module):

    def __init__(self, eps=1e-12):
        super(StatsPooling, self).__init__()
        self.eps = eps

    def forward(self, x, dim=-1):
        # (batches, channels, frames)
        mul = x.mean(dim=dim, keepdim=True)
        var = (x - mul).pow(2).mean(dim=dim)
        mul = mul.squeeze(dim)

        mask = (var <= self.eps).type(var.dtype).to(var.device)
        var = (1.0 - mask) * var + mask * self.eps
        std = var.sqrt()
        # (batches, 2 * channels)
        pooling = torch.cat((mul, std), dim=1)

        return pooling


class EXvector(nn.Module):

    def __init__(self):
        super(EXvector, self).__init__()
        self.model_name = 'TDNN'
        # Frame-level
        self.tdnn1 = TDNN(80, 512, 5, 1, 2)
        self.tdnn2 = TDNN(512, 512, 1, 1, 0)
        self.tdnn3 = TDNN(512, 512, 3, 2, 2)
        self.tdnn4 = TDNN(512, 512, 1, 1, 0)
        self.tdnn5 = TDNN(512, 512, 3, 3, 3)
        self.tdnn6 = TDNN(512, 512, 1, 1, 0)
        self.tdnn7 = TDNN(512, 512, 3, 4, 4)
        self.tdnn8 = TDNN(512, 512, 1, 1, 0)
        self.tdnn9 = TDNN(512, 512, 1, 1, 0)
        self.tdnn10 = TDNN(512, 1500, 1, 1, 0)

        # Statistics pooling layer
        self.pooling = StatsPooling()

        # Segment-level
        self.fc1 = nn.Linear(2 * 1500, 512)

    def forward(self, x, xavg, xstd):
    # def forward(self, x):
        # Frame-level
        # x = self.bn0(x)
        x=x.float()
        zx = (x - xavg) / xstd  # (130, 1, 128, 160)输入数据归一化处理
        # zx=x.float()
        # print(zx.size())
        out = self.tdnn1(zx)
        out = self.tdnn2(out)
        out = self.tdnn3(out)
        out = self.tdnn4(out)
        out = self.tdnn5(out)
        out = self.tdnn6(out)
        out = self.tdnn7(out)
        out = self.tdnn8(out)
        out = self.tdnn9(out)
        out = self.tdnn10(out)
        # Statistics pooling layer
        out = self.pooling(out, 2)
        embedding_a = self.fc1(out)
            # out = self.bn1(embedding_a)
            # out = self.relu(out)
            # embedding_b = self.fc2(out)
            # out = self.bn2(embedding_b)
            # out = self.relu(out)

        return embedding_a
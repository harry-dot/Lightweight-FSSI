import torch
import torch.nn.init as init

#初始化模型,apply(model_init)
def model_init(m):
    classname = m.__class__.__name__#得到网络层的名字
    if classname.find('Conv') != -1:#检查网络的名字中是否包含'Conv'，是的话是卷积层，要初始化
        init.kaiming_normal_(m.weight, mode='fan_out')#‘fan_out’表示weight的方差在反向传播中不变
        #init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:#对BN层进行初始化
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:#对全连接层进行初始化
        init.kaiming_normal_(m.weight, mode='fan_out')
        init.constant_(m.bias, 0)

#计算网络总的参数量
def show_model_params(model):
    params = 0
    for i in model.parameters():
        params += i.view(-1).size()[0]#叠加网络每一层的参数，也可以单独输出每一层参数
    print('Model:' + model.module.model_name + '\t#params:%d'%(params))
    #print('Model:' + model.model_name + '\t#params:%d'%(params))
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from networks.ECAPA_lite import *
from networks.ESBPGnet3 import *
if __name__ == "__main__":
    net = ESPGnet(channel_num=4)
    show_model_params(net)

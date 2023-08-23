from torchviz import make_dot
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
#sys.path.append('./net')
from fun import *
# from 研二.少样本说话人辨认.fewshot_speaker.InterSpeech2022.train2.net2.ESBPGnet.ESBPGnet3 import *
# from 研二.少样本说话人辨认.fewshot_speaker.InterSpeech2022.train2.data_loader.ESBPGnet_ratio.ESBPGnet_r4 import *
from networks.ECAPA_lite import *
#from 研二.少样本说话人辨认.fewshot_speaker.InterSpeech2022.train2.net2.ECAPA_lite.ECAPA_TDNN import *
# from 研二.少样本说话人辨认.fewshot_speaker.InterSpeech2022.train2.net2.TDNN.X_Vector import *
from networks.protonet3 import *
from networks.ESBPGnet3 import *

net = ESPGnet_subband(channel_num=4)

x = torch.rand(1,80,500)

yhat = net(x)

MyConvNetVis = make_dot(yhat)

MyConvNetVis.format = "png"
 # 指定文件生成的文件夹
MyConvNetVis.directory = "/data/hqs/lightweight_spk_rec"
 # 生成文件
MyConvNetVis.view()
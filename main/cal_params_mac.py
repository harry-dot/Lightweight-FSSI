from ptflops import get_model_complexity_info
from torchstat import stat
from thop import profile
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from networks.ECAPA_lite import *
from networks.ESBPGnet3 import *
# net =Res2Net(Bottle2neck,[2, 2, 2, 2], 1) #  0.365 M, 100.000% Params, 0.03 GMac
# ops, params = get_model_complexity_info(net, (1, 80,500), as_strings=True,
# 										print_per_layer_stat=True, verbose=True)

# from 研二.少样本说话人辨认.fewshot_speaker.InterSpeech2022.train2.net2.ECAPA_lite.ECAPA_TDNN import *
#net = ECAPA_TDNN(in_channels=80, channels=144, embd_dim=192) #  0.365 M, 100.000% Params, 0.03 GMac
# ops, params = get_model_complexity_info(net, ( 80,100), as_strings=True,
# 										print_per_layer_stat=True, verbose=True)
from ptflops import get_model_complexity_info
from torchstat import stat
from thop import profile
from networks.protonet3 import *
from networks.ECAPA_lite import ECAPA_TDNN
net = ESPGnet_subband(channel_num=1)  #0.053 M, 100.000% Params, 0.008 GMac
#net = ESPGnet_branch(channel_num=4)
#net = ECAPA_TDNN(80,144,192)
#ops, params = get_model_complexity_info(net, (40,500),as_strings=True,
#                                        print_per_layer_stat=True, verbose=True)
#nets = net()
ops, params = get_model_complexity_info(net, (80,699),as_strings=True,
                                        print_per_layer_stat=True, verbose=True)
#input = torch.randn(1,40, 500)
input = torch.randn(1,80,699)
flops, params = profile(net, inputs=(input, ))
print(params/1e3,flops/1e6)

# from 研二.少样本说话人辨认.fewshot_speaker.InterSpeech2022.train2.data_loader.ESBPGnet_ratio.ESBPGnet_depth import *
# net = DepthConv() #0.036 M, 100.000% Params, 0.013 GMac
# from 研二.少样本说话人辨认.fewshot_speaker.InterSpeech2022.train2.net2.ESBPGnet.ESBPGnet3_depth import *
# net = ESPGnet(channel_num=4)
# ops, params = get_model_complexity_info(net, (80,699), as_strings=True,
# 										print_per_layer_stat=True, verbose=True)
# input = torch.randn(1,80, 100)
# flops, params = profile(net, inputs=(input, ))
# print(params/1e3,flops/1e6)

# from 研二.少样本说话人辨认.fewshot_speaker.final.net.Thin_Resnet34.Resnet34 import *
# # net = MainModel() #2.181 M, 100.000% Params, 0.025 GMac
# # ops, params = get_model_complexity_info(net, (80,240),as_strings=True,
# #                                         print_per_layer_stat=True, verbose=True)
#
# # from 研二.少样本说话人辨认.fewshot_speaker.InterSpeech2022.train2.net2.TDNN.X_Vector2 import *
# net = X_vector() #0.036 M, 100.000% Params, 0.013 GMac
# ops, params = get_model_complexity_info(net, (300,80), as_strings=True,
# 										print_per_layer_stat=True, verbose=True)
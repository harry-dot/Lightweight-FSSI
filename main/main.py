import argparse
from train import *

# pre_trained model path
pmp = '/data/hqs/lightweight_spk_rec/Exp/vox2_logger/ESBPGnet_subband_Ls_0.01Lev2/7s_10w1s_new' #10w5s
logger = '/data/hqs/lightweight_spk_rec/Exp/vox2_logger/ESBPGnet_subband_Ls_0.01Lev2/7s_10w1s_new.log'
writer = '/data/hqs/lightweight_spk_rec/Exp/vox2_logger/ESBPGnet_subband_Ls_0.01Lev2/log_10w1s'

#pmp = '/data/hqs/lightweight_spk_rec/Exp/libri_logger/ESBPGnet_subband16_Ls_0.01Le/7s_10w1s_new' #10w5s
#logger = '/data/hqs/lightweight_spk_rec/Exp/libri_logger/ESBPGnet_subband16_Ls_0.01Le/7s_10w1s_new.log'
#writer = '/data/hqs/lightweight_spk_rec/Exp/libri_logger/ESBPGnet_subband16_Ls_0.01Le/log_10w1s'


tr_path= '/data/datasets/FSSV_feature/Vox2/Vox2_7s/train'
ts_path = '/data/datasets/FSSV_feature/Vox2/Vox2_7s/test'
val_path = '/data/datasets/FSSV_feature/Vox2/Vox2_7s/test'
mean_root = '/data/datasets/FSSV_feature/Vox2/Vox2_7s/tr_mean_std/mean.npy'
std_root = '/data/datasets/FSSV_feature/Vox2/Vox2_7s/tr_mean_std/std.npy'


parser = argparse.ArgumentParser(description= 'PyTorch Implementation for few-shot sound recognition')

parser.add_argument('--dn',  default='clean', type=str, help='dataset'
                                                             ' name')#这个参数选择用原始样本还是加噪之后的样本
parser.add_argument('--sr',  default=16000, type=int, help='[fea_ext] sample rate')
parser.add_argument('--ws',  default=400,  type=int, help='[fea_ext] windows size')
parser.add_argument('--hs',  default=200,   type=int, help='[fea_ext] hop sizze')
parser.add_argument('--mel', default=39,   type=int, help='[fea_ext] mel bands')
parser.add_argument('--msc', default=5,     type=int, help='[fea_ext] top duration of audio clip')
parser.add_argument('--et',  default=10000, type=int, help='[fea_ext] spect manti')
parser.add_argument('--tr_path',  default=tr_path, type=str, help='training data root')
parser.add_argument('--val_path',  default=val_path, type=str, help='training data root')
parser.add_argument('--ts_path',  default=ts_path, type=str, help='training data root')
parser.add_argument('--mean_root',  default=mean_root, type=str, help='training data root')
parser.add_argument('--std_root',  default=std_root, type=str, help='training data root')
parser.add_argument('--writer',  default=writer, type=str, help='[fea_ext] sample rate')
parser.add_argument('--bs',   default=1,    type=int,   help='[net] batch size')
parser.add_argument('--way',  default=10,    type=int,   help='[net] n-way')
parser.add_argument('--shot', default=1,    type=int,   help='[net] m-shot')
parser.add_argument('--n_query', default=5,    type=int,   help='[net] m-shot')
parser.add_argument('--x_dim',  default=1, type=int, help='[net] input channel')
parser.add_argument('--h_dim',  default=64, type=int, help='[net] hidden channel')
parser.add_argument('--z_dim',  default=64, type=int, help='[net] output channel')
parser.add_argument('--lrde', default=20,    type=int,   help='[net] divided the learning rate 10 by every lrde epochs')
parser.add_argument('--mom',  default=0.9,   type=float, help='[net] momentum')
parser.add_argument('--wd',   default=1e-4,  type=float, help='[net] weight decay')
parser.add_argument('--lr',   default=0.003,   type=float, help='[net] learning rate')
parser.add_argument('--fr', default=699,     type=int, help='[fea_ext] length of audio clip')
parser.add_argument('--ep',   default=100, type=int,   help='[net] epoch')
parser.add_argument('--beta', default=0.3,   type=float, help='[net] hyperparameter for pre-class loss weight')
parser.add_argument('--pmp',  default=pmp,   type=str,   help='[net] save model path')
parser.add_argument('--logger',  default=logger,   type=str,   help='[net] logger path')
parser.add_argument('--theta',  default=0.01,   type=float,   help='[net] weight of sisnr loss')
parser.add_argument('--m',  default=0.998,   type=float,   help='[net] weight of sisnr loss')
parser.add_argument('--channel_num',  default=4,   type=float,   help='[net] weight of sisnr loss')
args = parser.parse_args()


if __name__=='__main__':
    Trer = Trainer(args)
    Trer.fit()

import argparse
from tester import *
import logging
from compute_EER import validate
# pre_trained model path


pmp = '/data/hqs/lightweight_spk_rec/Exp/vox2_logger/ESBPGnet_subband_Ls_0.01Lev1/7s_10w1s_new/BEST_MODEL'
logger = '/data/hqs/lightweight_spk_rec/Exp/vox2_logger/ESBPGnet_subband_Ls_0.01Lev2/Vox2_test_7s/7s_10w1s_test.log'

tr_path= '/data/datasets/FSSV_feature/Vox2/Vox2_7s/train'
ts_path = '/data/datasets/FSSV_feature/Vox2/Vox2_7s/test'
val_path = '/data/datasets/FSSV_feature/Vox2/Vox2_7s/test'
mean_root = '/data/datasets/FSSV_feature/Vox2/Vox2_7s/tr_mean_std/mean.npy'
std_root = '/data/datasets/FSSV_feature/Vox2/Vox2_7s/tr_mean_std/std.npy'


# # params for audio feature extraction (mel-spectrogram)
parser = argparse.ArgumentParser(description= 'PyTorch Implementation for few-shot sound recognition')
# parser.add_argument('--dn',  default='aishell_noise_20', type=str, help='dataset name')
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
# parser.add_argument('--tr_root',  default=ts_root, type=str, help='test_all data root')
# parser.add_argument('--tr_root',  default=tr_root, type=str, help='test_all data root')
# params for training
parser.add_argument('--bs',   default=1,    type=int,   help='[net] batch size')
parser.add_argument('--way',  default=10,    type=int,   help='[net] n-way')
parser.add_argument('--shot', default=5,    type=int,   help='[net] m-shot')
parser.add_argument('--n_query', default=10,    type=int,   help='[net] m-shot')
parser.add_argument('--x_dim',  default=1, type=int, help='[net] input channel')
parser.add_argument('--h_dim',  default=64, type=int, help='[net] hidden channel')
parser.add_argument('--z_dim',  default=64, type=int, help='[net] output channel')
parser.add_argument('--lrde', default=20,    type=int,   help='[net] divided the learning rate 10 by every lrde epochs')
parser.add_argument('--mom',  default=0.9,   type=float, help='[net] momentum')
parser.add_argument('--wd',   default=1e-4,  type=float, help='[net] weight decay')
# parser.add_argument('--lr',   default=0.0000005,   type=float, help='[net] learning rate')
parser.add_argument('--lr',   default=0.00005,   type=float, help='[net] learning rate')
parser.add_argument('--fr', default=699,     type=int, help='[fea_ext] length of audio clip') # 699 500 300 100
parser.add_argument('--ep',   default=100, type=int,   help='[net] epoch')
parser.add_argument('--beta', default=0.3,   type=float, help='[net] hyperparameter for pre-class loss weight')
parser.add_argument('--pmp',  default=pmp,   type=str,   help='[net] save model path')
parser.add_argument('--logger',  default=logger,   type=str,   help='[net] logger path')
parser.add_argument('--train_method',  default="meta",   type=str,   help='[net] logger path')
parser.add_argument('--val_dir',  default="/data/datasets/FSSV_feature/data_list/vox2.txt",   type=str,   help='[net] logger path')
parser.add_argument('--test_mode',  default="acc",   type=str,   help='[net] logger path')
parser.add_argument('--channel_num',  default=4,   type=float,   help='[net] weight of sisnr loss')
args = parser.parse_args()

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

acc_5_1 = 0
acc_5_5 = 0
acc_10_1 = 0
acc_10_5 = 0
eer = 0
cost = 0
# for i in range(10):
#     Trer = Tester(args)
#     acc = Trer.run()
#     print(acc)
if not os.path.exists('/'.join(logger.split("/")[:-1])):
    os.makedirs('/'.join(logger.split("/")[:-1]))

t = logger.split('/')[-2].split('_')[-1][0]

if t not in ['1','3','5','7']:
    t = logger.split('/')[-1][0]
    
logger = get_logger(logger)
logger.info('start testing!')


frqs = [100,300,500,699]
args.fr = frqs[int(t)//2]
for i in range(10):
    Trer = Tester(args)
    pred = Trer.run()
    # acc_5_5 += pred[0]
    acc_5_1+= pred[0]
    acc_5_5+= pred[1]
    acc_10_1+= pred[2]
    acc_10_5+=pred[3]
    acc_5_1avg = float(acc_5_1)/float(i+1)
    acc_5_5avg = float(acc_5_5)/float(i+1)
    acc_10_1avg = float(acc_10_1)/float(i+1)
    acc_10_5avg = float(acc_10_5)/float(i+1)
    
    logger.info('Epoch:[{}/{}]\t  acc_5_1avg={:.4f}\t  acc_5_5avg={:.4f}\t acc_10_1avg={:.4f}\t acc_10_5avg={:.4f}\t '.format(i+1, 10, acc_5_1avg,
    acc_5_5avg, acc_10_1avg, acc_10_5avg))
if args.test_mode=="acc_eer":
    eer,cost  = Trer.validate(ts_path,args) 
    logger.info('eer = {}, cost = {}\t'.format(eer,cost))
    # logger.info('Epoch:[{}/{}]\t  acc_5_5avg={:.4f}\t '.format(i+1, 10, acc_5_5avg))
# logger.info('finish testing!')
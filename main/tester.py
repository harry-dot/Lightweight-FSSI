import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import time
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
sys.path.append('./net')
from fun import *
from networks.Res2Net import *
from networks.ESBPGnet3 import *

from networks.ECAPA_lite import *

from networks.protonet3 import *

from loader_vox import *

from utils import euclidean_dist,cos_similarity
from compute_EER import *

class Tester:
    def run(self):
        op, acc = self.Tester()
        return acc

    def __init__(self, args):
        self.args = args
        # TR_ESC_X, TR_ESC_Y, trvate = load_data(args.dn)
        # TS_ESC_X, TS_ESC_Y, trvate = load_data2(args.dn)
        # tridx, teidx = trvate
        tr_path = self.args.tr_path
        val_path = self.args.val_path
        ts_path = self.args.ts_path
        mean_root = self.args.mean_root
        std_root = self.args.std_root
        # build model
        self.model = nn.DataParallel(ESPGnet_subband(channel_num=args.channel_num).cuda())

        # data builder
        # default: n-ways m-shots
        self.tr_DS = B_DS(tr_path,val_path,ts_path, self.args, n=self.args.way, m=self.args.shot, mode='Training')
        self.ts_DS = B_DS(tr_path,val_path,ts_path, self.args, n=self.args.way, m=self.args.shot, mode='Test')
        te_DS_n5m5 = B_DS(tr_path,val_path,ts_path, self.args, n=5, m=5, mode='Test')
        te_DS_n5m1 = B_DS(tr_path,val_path,ts_path, self.args, n=5, m=1, mode='Test')
        te_DS_n10m5 = B_DS(tr_path,val_path,ts_path, self.args, n=10, m=5, mode='Test')
        te_DS_n10m1 = B_DS(tr_path,val_path,ts_path, self.args, n=10, m=1, mode='Test')
        self.te_DS = [te_DS_n5m1, te_DS_n5m5, te_DS_n10m1, te_DS_n10m5]
        self.evl_nm = [[5, 1], [5, 5], [10, 1], [10, 5]]

        mean = np.load(mean_root)
        std = np.load(std_root)
        Xavg = torch.from_numpy(mean)
        Xstd = torch.from_numpy(std)
        self.Xavg, self.Xstd = Variable(Xavg.view(1,1,1).cuda()), Variable(Xstd.view(1,1,1).cuda())

        self.show_dataset_model_params()
        self.load_pretrained_model(self.model)

    def Tester(self):
        st = time.time()
        self.model.eval()
        with torch.no_grad():

            te_print = []
            total = 0
            correct = 0
            f_score = 0
            for i in range(4):
                total = 0
                correct = 0
                tar_DS = self.te_DS[i]
                n, m = self.evl_nm[i]
                #m = self.args.n_query
                for batch_idx, (xs, xq) in enumerate(tar_DS):
                    xs = xs.view(n, m, 80, self.args.fr)
                    xq = xq.view(n, self.args.n_query, 80, self.args.fr)
                    #xq = xq.view(n, m, 80, self.args.fr)
                    #xs = xs.view(n, m, 80, -1)
                    #xq = xq.view(n, m, 80, -1)
                    n_class = xs.size(0)
                    # assert xq.size(0) == n_class
                    n_support = xs.size(1)
                    n_query = xq.size(1)
                    # total += n_query
                    # print(n_class)
                    # print(n_support)
                    # print(n_query)
                    target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
                    target_inds = Variable(target_inds, requires_grad=False)

                    if xq.is_cuda:
                        target_inds = target_inds.cuda()
                    # print(*xs.size()[:])
                    xs = xs.view(n_class * n_support, 80, self.args.fr)
                    xq = xq.view(n_class * n_query, 80, self.args.fr)
                    x = torch.cat([xs, xq], 0)
                    # self.optimizer.zero_grad()

                    #pred = self.model(x, self.Xavg, self.Xstd)
                    pred = self.model(x)
                    z_dim = pred.size(-1)
                    z_proto = pred[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
                    zq = pred[n_class * n_support:]
                    dists = euclidean_dist(zq, z_proto)
                    #dists = cos_similarity(zq,z_proto)
                    # print(dists.size())#torch.Size([1, 5])
                    # print(dists)
                    log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
                    #log_p_y = F.log_softmax(dists, dim=1).view(n_class, n_query, -1)
                    # loss_val = -log_p_y.gather(2, target_inds.cuda()).squeeze().view(-1).mean()
                    _, y_hat = log_p_y.max(2)
                    total += y_hat.size()[0]*y_hat.size()[1]

                    correct += torch.eq(y_hat, target_inds.view(n_class,n_query).cuda()).sum().float()
                acc_val = correct.item() / total
                te_print.append(acc_val)
                oprint = '%d-way %d-shot acc:%f  Time:%2f' % (n, m, acc_val, time.time() - st)
                print(oprint)
            print(te_print)
            return oprint, te_print

    def load_pretrained_model(self, model):
        # pre-training
        if os.path.exists(self.args.pmp):
            pretrained_model = torch.load(self.args.pmp)
            model_param = model.state_dict()
            for k in pretrained_model['state_dict'].keys():
                try:
                    model_param[k].copy_(pretrained_model['state_dict'][k])
                    # print(k)
                except:
                    print('[ERROR] Load pre-trained model %s' % (k))
                    # self.model.apply(model_init)
                    # break
            print('Load Pre_trained Model : ' + self.args.pmp)

        else:
            print('Learning from scrath')
            # self.model.apply(model_init)

    def show_dataset_model_params(self):
        # show model structure
        # print(self.model)
        # show params
        print(show_model_params(self.model))
    def validate(self,ts_root,args):
    #kwargs = {'batch_size': 1, 'num_workers': 8, 'pin_memory': True, #8 
    #              'drop_last': True}
        data = eer_data(ts_root)
        val_dataloader = torch.utils.data.DataLoader(data,batch_size=256,num_workers=8,pin_memory=True,shuffle=False)
        self.model.eval()
        embd_dict={}
        eer, cost = 1,1
        with torch.no_grad():
            for j, (feat, utt) in enumerate(val_dataloader):
                outputs = self.model(feat.cuda())  
                for i in range(len(utt)):
                    print(j, utt[i],feat.shape[2])
                    embd_dict[utt[i]] = outputs[i,:].cpu().numpy()
        #np.save('exp/%s/%s_%s.npy' % (opt.save_dir,opt.save_name, epoch),embd_dict)
        eer,_, cost,_ = get_eer(embd_dict,trial_file=args.val_dir)
            #get_score(embd_dict,trial_file='data/%s/trials' % opt.val_dir)

        return eer, cost
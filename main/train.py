import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] = '2'
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import logging
import time
import copy
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
#sys.path.append('./net')
from fun import *

from networks.ECAPA_lite import *

from networks.protonet3 import *
import networks.ESBPGnet3 as ESBPGnets

from loader_vox import *

from utils import *



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

class Trainer:
    def __init__(self, args):
        self.args = args
        tr_path = self.args.tr_path
        val_path = self.args.val_path
        ts_path = self.args.ts_path
        self.theta = self.args.theta

        # model builder
        # default segmentation I=4
        self.model = nn.DataParallel(getattr(ESBPGnets,self.args.model)(channel_num=args.channel_num).cuda())
            

        # data builder
        # default: n-ways m-shots
        
        self.tr_DS = B_DS(tr_path,val_path,ts_path, self.args, n=self.args.way, m=self.args.shot, mode='Training')
        self.va_DS = B_DS(tr_path,val_path,ts_path,  self.args, n=self.args.way, m=self.args.shot, mode='Valid')
        self.te_DS_n5m5 = B_DS(tr_path,val_path,ts_path, self.args, n=self.args.way, m=self.args.shot, mode='Test')

        # load avg and std for Z-score

        self.show_dataset_model_params()

    def fit(self):
        st = time.time()
        save_dict = {}
        best_ts_acc = 0
        best_te_acc = []
        loss_list = []
        acc_list = []
        self.model.train()
        torch.backends.cudnn.enabled = False
        if not os.path.exists("/".join(self.args.logger.split("/")[:-1])):
            os.makedirs("/".join(self.args.logger.split("/")[:-1]))
        logger = get_logger(self.args.logger)
        logger.info('start training!')
        lr = self.args.lr
        #self.optimizer = optim.SGD(self.model.parameters(),
        #                               lr=lr, momentum=self.args.mom, weight_decay=self.args.wd)
        self.optimizer = optim.Adam(self.model.parameters(),
                                       lr=lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=2e-5,
                                 amsgrad=False)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [40,70], gamma=0.5, last_epoch=-1)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [40,70], gamma=0.1, last_epoch=-1)
        step=0
        for e in range(1, self.args.ep + 1):
            # set optimizer (SGD)
            total_loss = 0
            
            print('\n==> Training Epoch #%d lr=%4f Best ts_acc:%f' % (e, lr, best_ts_acc))

            
            # self.optimizer = optim.Adam(self.model.parameters(), lr=lr, )
            acc_val = 0
            total = 0
            total_loss = 0
            # Training
            for batch_idx, (xs, xq) in enumerate(self.tr_DS):
                n_class = xs.size(1)
                assert xq.size(1) == n_class
                n_support = xs.size(2)
                n_query = xq.size(2)
                target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
                target_inds = Variable(target_inds, requires_grad=False)

                if xq.is_cuda:
                    target_inds = target_inds.cuda()
                xs = xs.view(n_class * n_support,80,self.args.fr)
                xq = xq.view(n_class * n_query,80,self.args.fr)
                x = torch.cat([xs,xq],0)
                # print(x.size())
                # print('xs',xs.size())
                # print('xq',xq.size())
                # print('x',x.size())
                self.optimizer.zero_grad()

                #pred = self.model(x, self.Xavg, self.Xstd)
                pred = self.model(x)
                #with torch.no_grad():
                #    e_s = self.Siamese_model(xs)
                
                z_dim = pred.size(-1)
                #proto =  e_s[:n_class*n_support].view(n_class,n_support,z_dim).mean(1)
                # print(pred.size())
                z_proto = pred[:n_class*n_support].view(n_class,n_support,z_dim).mean(1)
                zq = pred[n_class*n_support:]
                dists = euclidean_dist(zq, z_proto)
                
                
                #dists = cos_similarity(zq,z_proto)
                log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
                #log_p_y = F.softmax(-dists,dim=1).view(n_class, n_query, -1)
                #log_p_y = F.log_softmax(dists, dim=1).view(n_class, n_query, -1)
                #if e<10:
                #    loss_val = -log_p_y.gather(2, target_inds.cuda()).squeeze().view(-1).mean()
                #else:
                if self.theta>0:
                    sisnr_loss,intra_class,inter_class = sisnr(zq,z_proto,n_query)
                #sisnr_loss = sisnr(zq,proto,n_query)

                #loss_val = -(1-self.theta)*log_p_y.gather(2, target_inds.cuda()).squeeze().view(-1).mean() - self.theta*sisnr_loss
                lc = -log_p_y.gather(2, target_inds.cuda()).squeeze().view(-1).mean()
    
                if self.theta>0:
                    loss_val = lc + self.theta*sisnr_loss
                else:
                    loss_val = lc


                #loss_val = -log_p_y.gather(2, target_inds.cuda()).squeeze().view(-1).mean() + self.theta*sisnr_loss
                _, y_hat = log_p_y.max(2)

                total += y_hat.size()[0] * y_hat.size()[1]
                acc_val += torch.eq(y_hat, target_inds.view(n_class, n_query).cuda()).sum().float()

                loss_val.backward()
                self.optimizer.step()
                step+=1
                #with torch.no_grad():
                #    for param_q, param_k in zip(self.model.parameters(), self.Siamese_model.parameters()):
                #        param_k.data = param_k.data * self.args.m + param_q.data * (1. - self.args.m)
                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d]\tLoss %4f\tTime %d'
                                 % (e, self.args.ep, batch_idx + 1, len(self.tr_DS),
                                    loss_val.item(), time.time() - st))
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%4d/%4d]\tTime %d'
                                 % (e, self.args.ep, batch_idx + 1, len(self.tr_DS),
                                    time.time() - st))
                sys.stdout.flush()
            sys.stdout.write('\r')
            print(acc_val.item()/total)

            # Test
            self.result, va_acc = self.Tester(self.va_DS, 'Valid')
            self.result, ts_acc = self.Tester(self.te_DS_n5m5, 'Test')
            logger.info('Epoch:[{}/{}]\t lr:{:.10f}\t loss={:.5f}\t theta={:.5f}\t nway:{:.3f}\t mshot:{:.3f}\t val_acc={:.3f}\t ts_acc={:.3f}'.format(e, self.args.ep, lr, loss_val.item(),self.args.theta,self.args.way, self.args.shot, va_acc, ts_acc))
            scheduler.step()
            if best_ts_acc <= ts_acc:
                best_ts_acc = ts_acc
                best_te_acc = self.result
                self.Saver()

        logger.info('finish training!')
    
    
    def Tester(self, DS, vate):
        st = time.time()
        self.model.eval()
        te_print = []
        total = 0
        correct = 0
        if vate != 'Test':
            tar_DS = self.va_DS
            n, m = [self.args.way, self.args.shot]
        if vate == 'Test':
            tar_DS = self.te_DS_n5m5
            n, m = [self.args.way, self.args.shot]

        for batch_idx, (xs, xq) in enumerate(tar_DS):

            xs = xs.view(n, m, 80, self.args.fr)
            xq = xq.view(n, self.args.n_query, 80, self.args.fr)
            #xq = xq.view(n, m, 80, self.args.fr)
            # print('xs',xs.size())
            # print('xq',xq.size())
            n_class = xs.size(0)
            n_support = xs.size(1)
            n_query = xq.size(1)
            target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
            target_inds = Variable(target_inds, requires_grad=False)

            if xq.is_cuda:
                target_inds = target_inds.cuda()
            xs = xs.view(n_class * n_support, 80, self.args.fr)
            xq = xq.view(n_class * n_query, 80, self.args.fr)
            x = torch.cat([xs, xq], 0)

            #
            with torch.no_grad():
                #pred = self.model(x, self.Xavg, self.Xstd)
                pred = self.model(x)
            z_dim = pred.size(-1)
            z_proto = pred[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
            zq = pred[n_class * n_support:]
            dists = euclidean_dist(zq, z_proto)
            log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
            _, y_hat = log_p_y.max(2)
            total += y_hat.size()[0]*y_hat.size()[1]
            correct += torch.eq(y_hat, target_inds.view(n_class,n_query).cuda()).sum().float()

        if vate != 'Test':
            acc_val = correct.item() / total
            oprint = '%s %d-way %d-shot acc:%f Time:%1f' % (vate, n, m, acc_val, time.time() - st)
            print(oprint)
            return oprint, acc_val
        else:
            acc_val = correct.item() / total
            oprint = '%s %d-way %d-shot acc:%f Time:%1f' % (vate, n, m, acc_val, time.time() - st)
            print(oprint)
            return oprint, acc_val

    def load_pretrained_model(self, model):
        # pre-training
        if os.path.exists(self.args.pmp):
            pretrained_model = torch.load(self.args.pmp)
            model_param = model.state_dict()
            for k in pretrained_model['state_dict'].keys():
                try:
                    model_param[k].copy_(pretrained_model['state_dict'][k])
                    print(k)
                except:
                    print('[ERROR] Load pre-trained model %s' % (k))

            print('Load Pre_trained Model : ' + self.args.pmp)

        else:
            print('Learning from scrath')

    def show_dataset_model_params(self):
        # show model structure
        print(self.model)
        # show params
        print(show_model_params(self.model))


    def Saver(self):
        save_dict = {}

        if not os.path.exists(self.args.pmp):
            os.makedirs(self.args.pmp)

        save_dict['state_dict'] = self.model.state_dict()
        save_dict['result'] = self.result
        torch.save(save_dict, self.args.pmp + '/BEST_MODEL')
        print('Already save the model')
from scipy import spatial
import torch
import numpy as np
import os
import argparse
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from networks.ESBPGnet3 import *
# from 研二.少样本说话人辨认.fewshot_speaker.InterSpeech2022.loader2 import *
from loader_vox import *


def get_eer(embd_dict, trial_file):
    true_score = []
    false_score = []

    with open(trial_file) as fh:
        for line in fh:
            line = line.strip()
            #utt1, utt2, key = line.split()
            #utt1 = utt1 + '.wav'
            key, utt1, utt2 = line.split()
            #i = random.randint(1,4)
            #utt2 = utt2.format(i)
            result = 1 - spatial.distance.cosine(embd_dict[utt1.replace('ch','datasets')], embd_dict[utt2.replace('ch','datasets')])
            #result = 1 - spatial.distance.cosine(embd_dict[utt1], embd_dict[utt2.format('recorded2')])
            #result = spatial.distance.euclidean(F.normalize(torch.from_numpy(embd_dict[utt1]),dim=0).numpy(), F.normalize(torch.from_numpy(embd_dict[utt2]),dim=0).numpy())
            #if key == 'target':
            #    true_score.append(result)
            #elif key == 'nontarget':
            #    false_score.append(result)
            if key == '1':
                true_score.append(result)
            elif key == '0':
                false_score.append(result)  
    eer, threshold, mindct, threashold_dct = compute_eer(np.array(true_score), np.array(false_score))
    return eer, threshold, mindct, threashold_dct

def validate(ts_root,args,model):
    #kwargs = {'batch_size': 1, 'num_workers': 8, 'pin_memory': True, #8 
    #              'drop_last': True}
    data = eer_data(ts_root)
    val_dataloader = torch.utils.data.DataLoader(data,batch_size=256,num_workers=8,pin_memory=True,shuffle=False)
    model.eval()
    embd_dict={}
    eer, cost = 1,1
    with torch.no_grad():
        for j, (feat, utt) in enumerate(val_dataloader):
            outputs = model(feat.cuda())  
            for i in range(len(utt)):
                print(j, utt[i],feat.shape[2])
                embd_dict[utt[i]] = outputs[i,:].cpu().numpy()
    #np.save('exp/%s/%s_%s.npy' % (opt.save_dir,opt.save_name, epoch),embd_dict)
    eer,_, cost,_ = get_eer(embd_dict,trial_file=args.val_dir)
        #get_score(embd_dict,trial_file='data/%s/trials' % opt.val_dir)

    return eer, cost

def get_score(embd_dict, trial_file):
    true_score = []
    false_score = []

    with open(trial_file) as fh:
        with open('score.txt','w') as f:
            for line in fh:
                line = line.strip()
                #utt1, utt2, key = line.split()
                utt1, utt2 = line.split()
                #i = random.randint(1,4)
                #utt2 = utt2.format(i)
                result = 1 - spatial.distance.cosine(embd_dict[utt1], embd_dict[utt2])
                f.write(str(result))
                f.write('\n')

def compute_eer(target_scores, nontarget_scores, p_target=0.01):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    mindcf, threshold = ComputeMinDcf(frr, far, thresholds,p_target=p_target)
        
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index], mindcf, threshold   

def ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.01, c_miss=1, c_fa=1):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores
    return frr, far, thresholds


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description= 'PyTorch Implementation for few-shot sound recognition')
    
    parser.add_argument('--val_dir',  default="/data/datasets/FSSV_feature/data_list/vox2.txt",   type=str,   help='[net] logger path')
    args = parser.parse_args()
    eer,cost = validate('/data/datasets/FSSV_feature/Vox2/Vox2_7s/test',args)
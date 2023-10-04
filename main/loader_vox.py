import torch
from torch.utils.data import Dataset
import numpy as np
from random import sample, shuffle
import random
import os
from itertools import cycle
# data loader

class tr_data(Dataset):
    def __init__(self, tr_root, args, n, m, mode, tr_speaker):
        self.tr_root = tr_root
        self.args = args
        self.mode = mode
        self.way = n
        self.shot = m
        self.tr_speaker = tr_speaker
        self.num = 0
        self.init = 0
        self.speakers = os.listdir(tr_root)

        self.tr_speaker = len(self.speakers)
        print("tr_speaker:"+str(self.tr_speaker)+"\n")


    def list_of_groups(self, list_info, per_list_len):
        '''
        :param list_info:   列表
        :param per_list_len:  每个小列表的长度
        :return:
        '''
        list_of_group = zip(*(iter(list_info),) * per_list_len)
        end_list = [list(i) for i in list_of_group]  # i is a tuple
        count = len(list_info) % per_list_len
        end_list.append(list_info[-count:]) if count != 0 else end_list
        return end_list
    
        
    def __getitem__(self, index):
        # 5-way (5 classes)
        # way = self.args.way
        if self.init == 0:
            shuffle(self.speakers)

        if self.mode == 'Training':
            total_speakers = self.list_of_groups(self.speakers, self.way)
            # print(total_speakers)
            speakers = total_speakers[index]
            # print('index',index)
            # print('speakers',speakers)
            self.init += 1
            Support_set = []
            Query_set = []
            for speaker in speakers:
                path = os.path.join(self.tr_root, speaker)
                train_data = os.listdir(path)
                #train_data = sample(train_data, self.shot*2)
                train_data = sample(train_data, self.shot + self.args.n_query)
                train_npy = [os.path.join(path, ori) for ori in train_data]
                support = train_npy[:self.shot]
                query = train_npy[self.shot:]
                # print(support)
                # print(query)

                Support_set.append([np.load(x) for x in support])
                Query_set.append([np.load(x) for x in query])

            Support_set = np.array(Support_set) # (way,shot,T,F)
            Query_set = np.array(Query_set) # (way,shot,T,F)
            # print('total', total_speakers)
            return torch.Tensor(Support_set), torch.Tensor(Query_set)

    def __len__(self):
        # return 1
        return int(self.tr_speaker/self.way)
    
class tr_data_LDP(Dataset):
    def __init__(self, tr_root, args, n, m, mode, tr_speaker):
        self.tr_root = tr_root
        self.args = args
        self.mode = mode
        self.way = n
        self.shot = m
        self.tr_speaker = tr_speaker
        self.num = 0
        self.init = 0
        self.speakers = os.listdir(tr_root)
        self.truncate_num = 6
        self.n_query = args.n_query
    def list_of_groups(self, list_info, per_list_len):
        '''
        :param list_info:   列表
        :param per_list_len:  每个小列表的长度
        :return:
        '''
        list_of_group = zip(*(iter(list_info),) * per_list_len)
        end_list = [list(i) for i in list_of_group]  # i is a tuple
        count = len(list_info) % per_list_len
        end_list.append(list_info[-count:]) if count != 0 else end_list
        return end_list
    
        
    def __getitem__(self, index):
        # 5-way (5 classes)
        # way = self.args.way
        if self.init == 0:
            shuffle(self.speakers)

        if self.mode == 'Training':
            total_speakers = self.list_of_groups(self.speakers, self.way)
            # print(total_speakers)
            speakers = total_speakers[index]
            # print('index',index)
            # print('speakers',speakers)
            self.init += 1
            Support_set = []
            Query_set = []
            Query_set_1s = []
            for speaker in speakers:
                path = os.path.join(self.tr_root, speaker)
                train_data = os.listdir(path)
                #train_data = sample(train_data, self.shot*2)
                train_data = sample(train_data, self.shot + self.n_query)
                train_npy = [os.path.join(path, ori) for ori in train_data]
                support = train_npy[:self.shot]
                query = train_npy[self.shot:]
                # print(support)
                # print(query)
                query_1s = []
                query_7s = [np.load(x) for x in query]
                for _ in range(self.truncate_num):
                    query_1s.append(list(map(truncate,query_7s)))

                Query_set.append(query_7s)
                Query_set_1s.append(query_1s)
                Support_set.append([np.load(x) for x in support])
                #Query_set.append([np.load(x) for x in query])

            Support_set = np.array(Support_set) # (way,shot,T,F)
            Query_set = np.array(Query_set) # (way,shot,T,F)
            Query_set_1s = np.array(Query_set_1s)
            # print('total', total_speakers)
            return torch.Tensor(Support_set), torch.Tensor(Query_set),torch.Tensor(Query_set_1s).transpose(0,1)

    def __len__(self):
        # return 1
        return int(self.tr_speaker/self.way)
class val_data(Dataset):
    def __init__(self, val_root, args, n, m, mode, val_speaker):
        self.val_root = val_root
        self.args = args
        self.mode = mode
        self.way = n
        self.shot = m
        self.val_speaker = val_speaker
        self.num = 0
        self.init = 0
        self.speakers = sorted(os.listdir(val_root))[:val_speaker]

    def list_of_groups(self, list_info, per_list_len):
        '''
        :param list_info:   列表
        :param per_list_len:  每个小列表的长度
        :return:
        '''
        list_of_group = zip(*(iter(list_info),) * per_list_len)
        end_list = [list(i) for i in list_of_group]  # i is a tuple
        count = len(list_info) % per_list_len
        end_list.append(list_info[-count:]) if count != 0 else end_list
        return end_list

    def __getitem__(self, index):
        # 5-way (5 classes)
        # way = self.args.way
        # if self.init == 0:
        shuffle(self.speakers)

        if self.mode == 'Valid':
            # print('val')
            # self.speakers = self.speakers[:100]
            # print(self.speakers)
            speakers = self.list_of_groups(self.speakers, self.way)
            # print('val sp', speakers)
            speakers = speakers[index]
            # print('index',index)
            # print('speakers',speakers)
            self.init += 1
            Support_set = []
            Query_set = []
            for speaker in speakers:
                path = os.path.join(self.val_root, speaker)
                train_data = os.listdir(path)
                train_data = sample(train_data, self.shot * 2)
                train_npy = [os.path.join(path, ori) for ori in train_data]
                support = train_npy[:self.shot]
                query = train_npy[self.shot:]
                # print(support)
                # print(query)
                Support_set.append([np.load(x) for x in support])
                Query_set.append([np.load(x) for x in query])
            Support_set = np.array(Support_set)
            Query_set = np.array(Query_set)
            return torch.Tensor(Support_set), torch.Tensor(Query_set)

    def __len__(self):
        return int(self.val_speaker / self.way)
        # return 10

class ts_data(Dataset):
    def __init__(self, ts_root, args, n, m, mode, ts_speaker):
        self.ts_root = ts_root
        self.args = args
        self.mode = mode
        self.way = n
        self.shot = m
        self.n_query = args.n_query
        self.ts_speaker = ts_speaker
        self.num = 0
        self.init = 0
        # self.speakers = sorted(os.listdir(ts_root))[100:100+ts_speaker]
        self.speakers = sorted(os.listdir(ts_root))
        
    def list_of_groups(self, list_info, per_list_len):
        '''
        :param list_info:   列表
        :param per_list_len:  每个小列表的长度
        :return:
        '''
        list_of_group = zip(*(iter(list_info),) * per_list_len)
        end_list = [list(i) for i in list_of_group]  # i is a tuple
        count = len(list_info) % per_list_len
        end_list.append(list_info[-count:]) if count != 0 else end_list
        return end_list

    def __getitem__(self, index):
        # 5-way (5 classes)
        # way = self.args.way
        # if self.init == 0:
        shuffle(self.speakers)

        if self.mode == 'Test':
            # self.speakers = self.speakers[100:400]
            # print(self.speakers)
            # print(index)
            speakers = self.list_of_groups(self.speakers, self.way)
            # print('test_all sp', speakers)
            speakers = speakers[0]
            #speakers = self.n_speakers[index]

            self.init += 1
            Support_set = []
            Query_set = []
            for speaker in speakers:
                path = os.path.join(self.ts_root, speaker)
                train_data = os.listdir(path)
                train_data = sample(train_data, self.shot + self.n_query)
                #train_data = sample(train_data, self.shot*2)
                train_npy = [os.path.join(path, ori) for ori in train_data]
                support = train_npy[:self.shot]
                query = train_npy[self.shot:]
                Support_set.append([np.load(x) for x in support])
                Query_set.append([np.load(x) for x in query])
            Support_set = np.array(Support_set)
            Query_set = np.array(Query_set)
            return torch.Tensor(Support_set), torch.Tensor(Query_set)

    def __len__(self):
        #return len(self.speakers)//self.way
        return 100


def B_DS(X,Y,Z, args, n, m, mode='Training', tr_speaker=828, val_speaker=100, ts_speaker =300):

    if mode == 'Training':
        kwargs = {'batch_size': 1, 'num_workers': 8, 'pin_memory': True, #8 
                  'drop_last': True}
        # num = random.randint(0,1000)
        
        tr_DS = tr_data(X, args, n, m, mode,tr_speaker)

        # print('tr1',tr_DS.num)num
        # tr_DS.num += 1
        # print('tr',tr_DS.num)
        tr_loader = torch.utils.data.DataLoader(tr_DS, shuffle=True, **kwargs)
        return tr_loader


    if mode == 'Valid':
        kwargs = {'batch_size': 1, 'num_workers': 8, 'pin_memory': True}
        te_DS = val_data(Y, args, n, m, mode, val_speaker)
        te_loader = torch.utils.data.DataLoader(te_DS, **kwargs)
        return te_loader

    if mode == 'Test':
        kwargs = {'batch_size': 1, 'num_workers': 8, 'pin_memory': True}
        ts_DS = ts_data(Z, args, n, m, mode, ts_speaker)
        ts_loader = torch.utils.data.DataLoader(ts_DS, **kwargs)
        return ts_loader

def truncate(audio):
    start = random.randint(0,audio.shape[-1]-100)
    t_audio = audio[:,start:start+100]
    return t_audio

class val_data(Dataset):
    def __init__(self, val_root, args, n, m, mode, val_speaker):
        self.val_root = val_root
        self.args = args
        self.mode = mode
        self.way = n
        self.shot = m
        self.val_speaker = val_speaker
        self.num = 0
        self.init = 0
        self.speakers = sorted(os.listdir(val_root))[:val_speaker]

    def list_of_groups(self, list_info, per_list_len):
        '''
        :param list_info:   列表
        :param per_list_len:  每个小列表的长度
        :return:
        '''
        list_of_group = zip(*(iter(list_info),) * per_list_len)
        end_list = [list(i) for i in list_of_group]  # i is a tuple
        count = len(list_info) % per_list_len
        end_list.append(list_info[-count:]) if count != 0 else end_list
        return end_list

    def __getitem__(self, index):
        # 5-way (5 classes)
        # way = self.args.way
        # if self.init == 0:
        shuffle(self.speakers)

        if self.mode == 'Valid':
            # print('val')
            # self.speakers = self.speakers[:100]
            # print(self.speakers)
            speakers = self.list_of_groups(self.speakers, self.way)
            # print('val sp', speakers)
            speakers = speakers[index]
            # print('index',index)
            # print('speakers',speakers)
            self.init += 1
            Support_set = []
            Query_set = []
            for speaker in speakers:
                path = os.path.join(self.val_root, speaker)
                train_data = os.listdir(path)
                train_data = sample(train_data, self.shot+self.args.n_query)
                train_npy = [os.path.join(path, ori) for ori in train_data]
                support = train_npy[:self.shot]
                query = train_npy[self.shot:]
                # print(support)
                # print(query)
                Support_set.append([np.load(x) for x in support])
                Query_set.append([np.load(x) for x in query])
            Support_set = np.array(Support_set)
            Query_set = np.array(Query_set)
            return torch.Tensor(Support_set), torch.Tensor(Query_set)

    def __len__(self):
        return int(self.val_speaker / self.way)
        # return 10


from glob import glob
class eer_data(Dataset):
    def __init__(self, ts_root):
        self.ts_root = ts_root
        self.all_data = glob(os.path.join(ts_root,"*","*.npy"))

    def __getitem__(self, index):
        # 5-way (5 classes)
        # way = self.args.way
        # if self.init == 0:
        utt = self.all_data[index]
        feat = np.load(utt)

        return torch.Tensor(feat),utt


    def __len__(self):
        return len(self.all_data)
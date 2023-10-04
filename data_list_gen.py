import os

vox1 = "/data/datasets/FSSV_feature/Vox/Vox7s"
vox2 = "/data/datasets/FSSV_feature/Vox2/Vox2_7s"
lib = "/data/datasets/FSSV_feature/Lib2/Lib7s"
prefix = ['/data/datasets/voxceleb1/dev','/data/datasets/voxceleb1/test','/data/datasets/voxceleb2/train','/data/datasets/voxceleb2/test']
sets = ['train','test']
v1 = {}
v2 = {}
datasets = [vox1,vox2]

'''for dataset in datasets:
    train = []
    test = []
    for set in sets:
        with open('{}_{}.txt'.format(dataset.split('/')[-2],set),'w') as f:
            spk_list = os.path.join(dataset,set)
            for spk in os.listdir(spk_list):
                #if(spk not in train):
                #    if set == 'train':
                #        train.append(spk)
                #    else:
                #        test.append(spk)
                for audio in os.listdir(os.path.join(spk_list,spk)):
                    audio_folder = ''.join(audio[:-10])
                    audio_name = audio.split('_')[-1].replace('npy','wav')
                    path = os.path.join(spk,audio_folder,audio_name)
                    for pre in prefix:
                        if(os.path.exists(os.path.join(pre,path))):
                            t = '/'.join(pre.split('/')[3:])
                            path = os.path.join(t,path)

                f.write(path)
                f.write('\n')
            f.close()
    print(dataset+" train size:{}, test size:{}\n".format(len(train),len(test)))'''
           
vox2 = ["/data/datasets/FSSV_feature/Lib2/Lib7s/train",'/data/datasets/FSSV_feature/Vox_test/Vox7_test']
train = []
test = []
for i,path in enumerate(vox2):

    for spk in os.listdir(path):
        if(spk not in train):
            if i==0:
                train.append(spk)
            else:
                test.append(spk)
print(" train size:{}, test size:{}\n".format(len(train),len(test)))
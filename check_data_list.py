import os 

vox_train = ['Vox_train.txt','Vox_test.txt','Vox2_train.txt','Vox2_test.txt']

prefix = ['/data/datasets/voxceleb1/dev','/data/datasets/voxceleb1/test','/data/datasets/voxceleb2/train','/data/datasets/voxceleb2/test']

for pre,vox in zip(prefix,vox_train):
    with open(vox,'r') as f:
        for line in f.readlines():
            line = line.strip()
            assert os.path.exists(os.path.join(pre,line)),"file {} doesn't exist".format(os.path.join(pre,line))

print("all clear")
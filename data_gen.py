import librosa
import numpy as np
from config.config import Config
import os
from tqdm import tqdm


args = Config()

yourPATH = 'data/datasets'  # path to Voxceleb

with open('Vox_train.txt','r') as f:
    paths = f.readlines()
    for path in tqdm(paths):
        path = path.strip().replace('yourPATH',yourPATH)
        id,folder,name = path.split('/')[-3:]
        name = name.replace('wav','npy')
        name = '_'.join([folder,name])
        wav,sr = librosa.load(path,sr=None,duration=7)
        mean = np.mean(wav)
        std = np.std(wav)
        std = max(std,1e-6)
        wav = (wav-mean)/std
        mel = librosa.feature.melspectrogram(wav,sr,n_fft=512,hop_length=args.hs,win_length=args.ws,n_mels=80)
        log_mel = np.log(mel)[:,:-2]
        save_path = os.path.join('datasets/Vox/Vox7s/train',id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path,name),log_mel)

with open('Vox_test.txt','r') as f:
    paths = f.readlines()
    for path in tqdm(paths):
        path = path.strip().replace('yourPATH','data/datasets')
        id,folder,name = path.split('/')[-3:]
        name = name.replace('wav','npy')
        name = '_'.join([folder,name])
        wav,sr = librosa.load(path,sr=None,duration=7)
        mean = np.mean(wav)
        std = np.std(wav)
        std = max(std,1e-6)
        wav = (wav-mean)/std
        mel = librosa.feature.melspectrogram(wav,sr,n_fft=512,hop_length=args.hs,win_length=args.ws,n_mels=80)
        log_mel = np.log(mel)[:,:-2]
        save_path = os.path.join('datasets/Vox/Vox7s/test',id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path,name),log_mel)

with open('Vox2_train.txt','r') as f:
    paths = f.readlines()
    for path in tqdm(paths):
        path = path.strip().replace('yourPATH','data/datasets')
        id,folder,name = path.split('/')[-3:]
        name = name.replace('wav','npy')
        name = '_'.join([folder,name])
        wav,sr = librosa.load(path,sr=None,duration=7)
        mean = np.mean(wav)
        std = np.std(wav)
        std = max(std,1e-6)
        wav = (wav-mean)/std
        mel = librosa.feature.melspectrogram(wav,sr,n_fft=512,hop_length=args.hs,win_length=args.ws,n_mels=80)
        log_mel = np.log(mel)[:,:-2]
        save_path = os.path.join('datasets/Vox2/Vox2_7s/train',id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path,name),log_mel)

with open('Vox2_test.txt','r') as f:
    paths = f.readlines()
    for path in tqdm(paths):
        path = path.strip().replace('yourPATH','data/datasets')
        id,folder,name = path.split('/')[-3:]
        name = name.replace('wav','npy')
        name = '_'.join([folder,name])
        wav,sr = librosa.load(path,sr=None,duration=7)
        mean = np.mean(wav)
        std = np.std(wav)
        std = max(std,1e-6)
        wav = (wav-mean)/std
        mel = librosa.feature.melspectrogram(wav,sr,n_fft=512,hop_length=args.hs,win_length=args.ws,n_mels=80)
        log_mel = np.log(mel)[:,:-2]
        save_path = os.path.join('datasets/Vox2/Vox2_7s/test',id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path,name),log_mel)


        
        
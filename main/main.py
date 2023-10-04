from train import *
from config.config import Config
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

args = Config()


if __name__=='__main__':
    Trer = Trainer(args)
    Trer.fit()

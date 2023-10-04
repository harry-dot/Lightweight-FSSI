import argparse
from tester import *
import logging
from compute_EER import validate
# pre_trained model path
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from config.config_test import Config


args = Config()

logger = args.logger

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

    # logger.info('Epoch:[{}/{}]\t  acc_5_5avg={:.4f}\t '.format(i+1, 10, acc_5_5avg))
# logger.info('finish testing!')
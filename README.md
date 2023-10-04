# "Low-Complexity Speaker Embedding Module with Feature Segmentation, Transformation and Reconstruction for Few-Shot Speaker Identification" source code

# 0.Introduction

### This repository is the source code of the proposed FSSI system, which includes:

+ ### /config: the configuration files for the experiments.
+ ### /datasets: the training\test sets are out here.
+ ### /Exp: folder where the experiment logs\model\results are stored.
+ ### /main: core codes for the experiments.
+ ### /networks: implementation of the models.
+ ### requirements.txt: the required depencies for running the code
+ ### Vox_train.txt,Vox_test.txt,Vox2_train.txt,Vox2_test.txt: the samples selected from voxceleb1,voxceleb2, which are used to generate the datasets in the paper. 

# 1.Usage
 
+ ### Step 1 Data Preparation:
```
    python data_gen.py
```
+ ### Step 2 Training Model:
```
    bash train.sh
```
+ ### Step 3 Testing Model:
```
    bash test.sh
```
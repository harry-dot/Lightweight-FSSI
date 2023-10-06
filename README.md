# "Low-Complexity Speaker Embedding Module with Feature Segmentation, Transformation and Reconstruction for Few-Shot Speaker Identification" source code

# 0.Introduction

### This repository is the source code of the proposed FSSI system, which includes:

+ ### */config*: the configuration files for the experiments.
+ ### */datasets*: the training&test sets are put here.
+ ### */Exp*: folder where the experiment logs&model&results are stored.
+ ### */main*: core codes for the experiments.
+ ### */networks*: implementation of the models.
+ ### *requirements.txt*: the required depencies for running the code
+ ### *Vox*.txt*: lists of samples selected from voxceleb1,voxceleb2, which are used to generate the datasets in the paper. 

# 1.Usage
 
+ ### Step 1 Data Preparation:
The V1-set,V2-set and Lib-set used in the paper derive from Voxceleb1,Voxceleb2 and Librispeech respectively, the statistics of which are as follow. 
![datasets](./1696556770865.png)

We provide the script for generating the three sets. To run it, please download Voxceleb1 and Voxceleb2 from [here](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) and change *yourPATH* in the script to your local path, then run
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

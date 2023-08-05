# FLAME_public
This repository contains the source code used in the paper "FLAME: Federated Learning across Multi-device Environments" published in IMWUT 2022.

## Run
```
python hparam_tuning.py --exp_name <experiment name> --working_directory <working_directory_name> --dataset {realworld,pamap2,opportunity} --gpu_device '0,1,2,3,4,5,6,7'
```

## Cite
```
@article{Cho2022_FLAME,
author = {Cho, Hyunsung and Mathur, Akhil and Kawsar, Fahim},
title = {FLAME: Federated Learning across Multi-Device Environments},
year = {2022},
issue_date = {September 2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {6},
number = {3},
url = {https://doi.org/10.1145/3550289},
doi = {10.1145/3550289},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = {sep},
articleno = {107},
numpages = {29},
keywords = {Federated Learning, Human Activity Recognition}
}
```

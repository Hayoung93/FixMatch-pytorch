- TODO  
ctaugment  
randomize magnitude of randaug

## Unofficial Pytorch implementation of FixMatch (NIPS 2020)

### Links to official implementation
- [Official GitHub](https://github.com/google-research/fixmatch/tree/d4985a158065947dba803e626ee9a6721709c570) (Tensorflow)
- [Paper link](https://arxiv.org/abs/2001.07685)

### Environments
- Use docker image: `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel`
- Install packages: `pip install tensorboard termcolor yacs`

### Usage
- `python train.py --input_size 96 --log_name 20230131 --randaug --num_epochs 300 --t_max 300`

### Trained weight
- RandomAugment  

| RA magnitude | RA number | epochs | Initial LR | val acc |
| :---: | :---: | :---: | :---: | :---: |
| 1 | 2 | 200 | 0.005 | 91.275 |
| 1 | 2 | 300 | 0.005 | xx.xxx |
- CTAugment
    - Note: Augmentation's behavior of the original paper and official implement seems different
        - Additional 'blur' operation
        - Blending after smoothing
        - ...
    - In this version, I followed the paper.

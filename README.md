- TODO
ctaugment  
randomize magnitude of randaug

### Environments
- Use docker image: `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel`
- Install packages: `pip install tensorboard termcolor yacs`

### Usage
- `python train.py --input_size 96 --log_name 20230131 --randaug --num_epochs 300 --t_max 300`

### Trained weight
- RandomAugment  
| RA magnitude | RA number | epochs | val acc |
| --- | --- | --- | --- |
| 1 | 2 | 200 | 91.275 |
| 1 | 2 | 300 | xx.xxx |
- CTAugment

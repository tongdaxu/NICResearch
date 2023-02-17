#!/bin/bash
#SBATCH --job-name gg18                    # 任务名叫 example
#SBATCH --gres gpu:a100-80G                 # 每个子任务都用一张 A100 GPU
#SBATCH --time 192:00:00                      # 子任务 1 天 1 小时就能跑完
#SBATCH --output %A_%a.out                  # 100个程序输出重定向到 [任务id]_[子任务序号].out
#SBATCH --mail-user x.tongda@nyu.edu        # 这些程序开始、结束、异常突出的时候都发邮件告诉我
#SBATCH --mail-type ALL                     

# 任务 ID 通过 SLURM_ARRAY_TASK_ID 环境变量访问
# 上述行指定参数将传递给 sbatch 作为命令行参数
# 中间不可以有非 #SBATCH 开头的行
# 执行 sbatch 命令前先通过 conda activate [env_name] 进入环境

# python -u train.py -d /home/JJ_Group/xutd/CPNIC/dataset/ --epochs 128 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.01 --cuda --save
# python -u train.py -d /home/JJ_Group/xutd/git/Dataset/COCO --epochs 128 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0025 --cuda --save
# python -u train_perceptual.py -d /home/JJ_Group/xutd/git/Dataset/COCO --epochs 128 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0005 --cuda --save
python -u train_perceptual.py -d /home/JJ_Group/xutd/git/Dataset/COCO --epochs 128 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0003 --cuda --save
# python -u train_con_perceptual.py --epochs 128 -lr 1e-4 --batch-size 16 --test-batch-size 1 --lambda 0.0003 --cuda --save
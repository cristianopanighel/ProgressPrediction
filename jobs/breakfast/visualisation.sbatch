#!/bin/sh
### job name
#SBATCH --job-name=pn_bf_1

###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cristiano.panighel@studenti.unipd.it

### Standard output and error
#SBATCH --error=pn_bf_1.err
#SBATCH --output=pn_bf_1.out

### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=16G

### Time limit for the job
#SBATCH --begin=now

### GPU request
#SBATCH --gres=gpu:1
#SBATCH --constraint=cudadrv510

#SBATCH --qos=short
#SBATCH --cpus-per-task=16

wandb offline
python3 /home/cpanighe/ProgressPrediction/code/main.py \
    --seed 42 \
    --dataset breakfast \
    --data_dir rgb-images \
    --train_split train_s1.txt \
    --test_split test_small.txt \
    --batch_size 1 \
    --network progressnet \
    --backbone vgg11 \
    --load_backbone vgg11.pth \
    --num_workers 2 \
    --max_length 550 \
    --load_experiment pn_bf_random \
    --load_iteration 50000 \
    --subsample_fps 15 \
    --random \
    --save_dir pn_bf_random
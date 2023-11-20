#!/bin/sh
### job name
#SBATCH --job-name=pn_ucf

###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cristiano.panighel@studenti.unipd.it

### Standard output and error
#SBATCH --error=pn_ucf_1.err
#SBATCH --output=pn_ucf_1.out

### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=32G

### Time limit for the job
#SBATCH --begin=now

### GPU request
#SBATCH --gres=gpu:1
#SBATCH --constraint=cudadrv510

#SBATCH --qos=medium
#SBATCH --cpus-per-task=16

wandb offline
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
python3 /home/cpanighe/ProgressPrediction/code/main.py \
    --seed 42 \
    --experiment_name pn_ucf_1 \
    --wandb_name pn_ucf_1 \
    --wandb_project pn_ucf_1 \
    --dataset ucf24 \
    --data_dir rgb-images \
    --bboxes \
    --train_split train_small.txt \
    --test_split test_small.txt \
    --batch_size 1 \
    --iterations 50000 \
    --network progressnet \
    --backbone vgg11 \
    --load_backbone vgg11.pth \
    --num_workers 2 \
    --max_length 550 \
    --load_experiment pn_ucf_segments \
    --load_iteration 50000 \
    --save_dir pn_ucf_segments
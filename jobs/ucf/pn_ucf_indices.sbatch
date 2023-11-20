#!/bin/sh
### job name
#SBATCH --job-name=pn_ucf_indices

###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cristiano.panighel@studenti.unipd.it

### Standard output and error
#SBATCH --error=pn_ucf_indices.err
#SBATCH --output=pn_ucf_indices.out

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
    --experiment_name pn_ucf_indices \
    --wandb_name pn_ucf_indices \
    --wandb_project pn_ucf_indices \
    --dataset ucf24 \
    --data_dir rgb-images \
    --bboxes \
    --train_split train.txt \
    --test_split test.txt \
    --batch_size 1 \
    --iterations 50000 \
    --network progressnet \
    --backbone vgg11 \
    --load_backbone vgg11.pth \
    --dropout_chance 0.5 \
    --lr_decay 1.0 \
    --lr_decay_every 1000000 \
    --log_every 100 \
    --test_every 2500 \
    --num_workers 2 \
    --max_length 600 \
    --subsample \
    --indices \
    --indices_normalizer 128
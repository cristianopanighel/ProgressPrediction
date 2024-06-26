#!/bin/sh
### job name
#SBATCH --job-name=pn_ucf_vgg11_gru_segments_progress_change

###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cristiano.panighel@studenti.unipd.it

### Standard output and error
#SBATCH --error=pn_ucf_vgg11_gru_segments_progress_change.err
#SBATCH --output=pn_ucf_vgg11_gru_segments_progress_change.out

### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=32G

### Time limit for the job
#SBATCH --begin=now

### GPU request
#SBATCH --gres=gpu
#SBATCH --constraint=cudadrv495

#SBATCH --qos=medium
#SBATCH --cpus-per-task=16

wandb offline
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
python3 /home/cpanighe/ProgressPrediction/code/main.py \
    --seed 42 \
    --experiment_name pn_ucf_vgg11_gru_segments_progress_change \
    --wandb_name pn_ucf_segments \
    --wandb_project pn_ucf_segments \
    --dataset ucf24 \
    --data_dir rgb-images \
    --bboxes \
    --train_split train_telic.txt \
    --test_split test_telic.txt \
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
    --max_length 300 \
    --subsample
#!/bin/sh
### job name
#SBATCH --job-name=pn_bf

###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cristiano.panighel@studenti.unipd.it

### Standard output and error
#SBATCH --error=pn_bf.err
#SBATCH --output=pn_bf.out

### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=32G

### Time limit for the job
#SBATCH --begin=now

### GPU request
#SBATCH --gres=gpu:1
#SBATCH --constraint=cudadrv495

#SBATCH --qos=medium
#SBATCH --cpus-per-task=16
#SBATCH --array=1,2,3,4

wandb offline
python3 /home/cpanighe/ProgressPrediction/code/main.py \
    --seed 42 \
    --experiment_name pn_bf_${SLURM_ARRAY_TASK_ID} \
    --wandb_name pn_bf_${SLURM_ARRAY_TASK_ID} \
    --wandb_project pn_bf_${SLURM_ARRAY_TASK_ID} \
    --dataset breakfast \
    --data_dir rgb-images \
    --train_split train_s${SLURM_ARRAY_TASK_ID}.txt \
    --test_split test_s${SLURM_ARRAY_TASK_ID}.txt \
    --batch_size 1 \
    --iterations 50000 \
    --network progressnet \
    --backbone vgg11 \
    --load_backbone vgg11.pth \
    --dropout_chance 0.5 \
    --lr_decay 1.0 \
    --lr_decay_every 1000000 \
    --log_every 100 \
    --test_every 5000 \
    --num_workers 2 \
    --subsample_fps 15
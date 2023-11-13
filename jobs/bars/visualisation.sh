#!/bin/sh
### job name
#SBATCH --job-name=pn_bars_1

###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cristiano.panighel@studenti.unipd.it

### Standard output and error
#SBATCH --error=pn_bars_1.err
#SBATCH --output=pn_bars_1.out

### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=16G

### Time limit for the job
#SBATCH --begin=now

### GPU request
#SBATCH --gres=gpu
#SBATCH --constraint=cudadrv510

#SBATCH --qos=short
#SBATCH --cpus-per-task=2

wandb offline
python3 /home/cpanighe/ProgressPrediction/code/main.py \
    --seed 42 \
    --experiment_name pn_bars_1 \
    --wandb_name pn_bars_1 \
    --wandb_project pn_bars_1 \
    --dataset bars \
    --data_dir rgb-images \
    --train_split 1.txt \
    --test_split 1.txt \
    --batch_size 1 \
    --network progressnet \
    --backbone vgg16 \
    --load_backbone vgg16.pth \
    --num_workers 2 \
    --subsample \
    --max_length 100 \
    --load_experiment pn_bars_segments \
    --load_iteration 40000 \
    --save_dir pn_bars \
    --no_resize

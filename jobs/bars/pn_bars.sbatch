#!/bin/sh
### job name
#SBATCH --job-name=pn_bars_progresstest

###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cristiano.panighel@studenti.unipd.it

### Standard output and error
#SBATCH --error=pn_bars.err
#SBATCH --output=pn_bars.out

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
srun python3 /home/cpanighe/ProgressPrediction/code/main.py \
    --seed 42 \
    --experiment_name pn_bars \
    --wandb_name pn_bars \
    --wandb_project pn_bars \
    --dataset bars \
    --data_dir rgb-images \
    --train_split train.txt \
    --test_split test.txt \
    --batch_size 1 \
    --iterations 50000 \
    --network progresstest \
    --backbone swintransformer \
    --load_backbone None \
    --dropout_chance 0.5 \
    --lr_decay 1.0 \
    --lr_decay_every 1000000 \
    --log_every 100 \
    --test_every 1000 \
    --num_workers 2

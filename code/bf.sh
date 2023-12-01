#!/bin/sh

### job name
#SBATCH --job-name=pn_bf_script

###
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cristiano.panighel@studenti.unipd.it

### Standard output and error
#SBATCH --error=pn_bf_script.err
#SBATCH --output=pn_bf_script.out

### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=16G

### Time limit for the job
#SBATCH --begin=now

### GPU request
#SBATCH --gres=gpu:1
#SBATCH --constraint=K20

#SBATCH --qos=short
#SBATCH --cpus-per-task=16

### embed
python3 main.py \
    --seed 42 \
    --experiment_name progressnet_bf_scrambledegg_embed \
    --dataset breakfast \
    --data_dir rgb-images \
    --train_split all_scrambledegg.txt \
    --test_split test_scrambledegg_s1.txt \
    --batch_size 1 \
    --network progressnet \
    --backbone vgg11 \
    --load_backbone vgg11.pth \
    --embed \
    --embed_batch_size 32 \
    --embed_dir features/progressnet_scrambledegg_1 \
    --num_workers 1

### sequence
python3 main.py \
    --seed 42 \
    --experiment_name progressnet_bf_scrambledegg_sequence \
    --dataset breakfast \
    --data_dir features/progressnet_scrambledegg_1 \
    --train_split train_scrambledegg_s1.txt \
    --test_split test_scrambledegg_s1.txt \
    --feature_dim 2048 \
    --batch_size 1 \
    --iterations 5000 \
    --network progressnet \
    --dropout_chance 0.3 \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --log_every 50 \
    --test_every 500

### segment
python3 main.py \
    --seed 42 \
    --experiment_name progressnet_bf_scrambledegg_segment \
    --dataset breakfast \
    --data_dir features/progressnet_scrambledegg_1 \
    --train_split train_scrambledegg_s1.txt \
    --test_split test_scrambledegg_s1.txt \
    --feature_dim 2048 \
    --subsample \
    --batch_size 1 \
    --iterations 5000 \
    --network progressnet \
    --dropout_chance 0.3 \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --log_every 50 \
    --test_every 500

### indices
python3 main.py \
    --seed 42 \
    --experiment_name progressnet_bf_scrambledegg_indices \
    --dataset breakfast \
    --data_dir features/progressnet_scrambledegg_1 \
    --train_split train_scrambledegg_s1.txt \
    --test_split test_scrambledegg_s1.txt \
    --feature_dim 2048 \
    --indices \
    --indices_normalizer 3117 \
    --subsample \
    --batch_size 1 \
    --iterations 5000 \
    --network progressnet \
    --dropout_chance 0.3 \
    --optimizer sgd \
    --loss smooth_l1 \
    --momentum 0.9 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --lr_decay 0.1 \
    --lr_decay_every 10000 \
    --log_every 50 \
    --test_every 500
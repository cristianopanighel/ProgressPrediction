python main.py \
    --seed 42 \
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
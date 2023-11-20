#! /bin/sh

python3 main.py --train True --dataset MNIST --seed 1 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 100 --z_dim 15 --max_iter 1e6 \
    --gpu 0 --viz_name VAE --save_step 10000 --load_ckp True
    
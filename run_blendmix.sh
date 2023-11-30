#!/usr/bin/env bash

# python -m debugpy --listen 5678 --wait-for-client train.py \
# python train.py \

# python -m debugpy --listen 5678 --wait-for-client train.py \
python train_blendmix.py \
--net_type resnet \
--dataset FashionMNIST \
--depth 34 \
--batch_size 64 \
--lr 0.25 \
--expname blendmix_FashionMNIST_34 \
--epochs 300 \
--beta 1.0 \
--cutmix_prob 0.5 \
-j 12 \
--no-verbose

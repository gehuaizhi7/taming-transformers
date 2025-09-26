#!/bin/bash

# Distributed training script for ImageNet-256 dataset
# This script uses PyTorch's DistributedDataParallel for optimal multi-GPU performance

echo "Starting ImageNet-256 distributed training..."

# Kill any existing training processes
pkill -f "train_custom.py"

# Set environment variables for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=3
export RANK=0
export CUDA_VISIBLE_DEVICES=1,2,3

# Launch distributed training (PyTorch 1.7.0 compatible) - using only GPUs 1-3
python -m torch.distributed.launch --nproc_per_node=3 --master_port=12355 train_custom.py \
    --config configs/imagenet256_transformer_100k.yaml \
    --epochs 50 \
    --max_checkpoints 3 \
    --save_every 1 \
    --distributed \
    --dist_backend nccl \
    --sync_batchnorm \
    --kl_anneal_mode none \
    --save_dir checkpoints_imagenet256_distributed

echo "ImageNet-256 distributed training completed!"

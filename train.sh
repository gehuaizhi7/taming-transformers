#!/bin/bash

# Optimized training script for Taming Transformers
# This script demonstrates various checkpoint optimization options

echo "Starting optimized training with checkpoint management..."

# Basic training with default settings (keeps 3 checkpoints, saves every 5 epochs)
python train_custom.py \
    --config configs/unconditional_transformer.yaml \
    --epochs 50 \
    --max_checkpoints 3 \
    --save_every 1 \
    --gpu_ids 7

echo "Training completed!"

# Alternative: Very aggressive space saving (only best models, every 10 epochs)
# python train_custom.py \
#     --config configs/unconditional_transformer.yaml \
#     --epochs 50 \
#     --max_checkpoints 2 \
#     --save_every 10 \
#     --save_best_only \
#     --compress_checkpoints

# Alternative: Minimal space usage (only best model, no regular checkpoints)
# python train_custom.py \
#     --config configs/unconditional_transformer.yaml \
#     --epochs 50 \
#     --max_checkpoints 1 \
#     --save_best_only \
#     --compress_checkpoints

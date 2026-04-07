#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1   

DATA_DIR="Your dataset path"
VISION_CKPT="PL-Stitch vision foundation model"
SAVE_DIR="./checkpoints_surgvlp"
PORT=29501 # 随机可用端口

# 确保保存目录存在
mkdir -p $SAVE_DIR

echo "=========================================="
echo "Starting DDP Training for Surgical VLP..."
echo "Data Directory: $DATA_DIR"
echo "Save Directory: $SAVE_DIR"
echo "=========================================="

# 启动 DDP 训练
python -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:$PORT \
  train.py \
  --root_dir "$DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --vision_ckpt_path "$VISION_CKPT" \
  --batch_size 240 \
  --epochs 10 \
  --warmup_epochs 1.0 \
  --lr 2e-4 \
  --proj_lr_multiplier 2.0 \
  --lora_r 16 \
  --save_interval 10
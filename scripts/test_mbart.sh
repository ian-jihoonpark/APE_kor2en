#!/bin/bash
CKPT_DIR="/mnt/data/checkpoints"

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 Tester.py \
--seed 42 \
--mode test \
--experiment_name "mBART-adapter" \
--ckpt_path ${CKPT_DIR} \
--data_path "/mnt/data/data/APE" \
--adapter_latent_size 64 \
# --train_mode "prompting" \
#!/bin/bash
CKPT_DIR="/mnt/data/checkpoints"

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 Trainer.py \
--seed 42 \
--mode train \
--boost_mode amp \
--local_rank 0 \
--project_name "AutomaticPostEditing" \
--cached_dir /media/storage/checkpoints/NLX_GPT/cached \
--experiment_name "mBART-adapter" \
--num_train_epochs 10 \
--ckpt_path ${CKPT_DIR} \
--weight_decay 0.0 \
--data_path "/mnt/data/data/APE" \
--train_batch_size 8 \
--eval_batch_size 8 \
--learning_rate 1e-5 \
--gradient_accumulation_steps 2 \
--opt_level "O2" \
--adapter_latent_size 64 \
# --train_mode "prompting" \
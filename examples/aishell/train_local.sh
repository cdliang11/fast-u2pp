#!/usr/bin/bash

data=data

train_config=conf/train_u2++_conformer.yaml
gpu_ids="1"
dir=exp/conformer_u2pp_distill_e2etrain_noadd

checkpoint=ckpt/avg_30.pt
# checkpoint=

. path.sh
. tools/parse_options.sh

echo "Start Training"
mkdir -p $dir

num_gpus=$(echo $gpu_ids | awk -F ',' '{print NF}')
torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
  wenet/bin/train.py --device-ids $gpu_ids \
    --local_run \
    --dist_url "file://${dir}/ddp_init" \
    --ddp.dist_backend gloo \
    --data_type "shard" \
    --config $train_config \
    --symbol_table  $data/dict/lang_char.txt \
    --train_data $data/train/data.list \
    --cv_data $data/dev/data.list \
    --model_dir $dir \
    --num_workers 2 \
    --cmvn $data/train/global_cmvn \
    --pin_memory \
    --train_utt $data/train/wav.scp \
    ${checkpoint:+--checkpoint $checkpoint} \
    --finetune


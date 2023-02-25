#!/usr/bin/bash

data=data
dist_url=
train_config=conf/train_conformer.yaml
# For cluster training, we always use 8 gpus
gpu_ids="0,1,2,3,4,5,6,7"
dir=exp/conformer
hdfs_dir=aishell_conformer
# Aidi cluster gets the same name as local machine by whoami
hdfs_username=`whoami`
checkpoint=ckpt/avg_30.pt
# checkpoint=

. path.sh
. tools/parse_options.sh

# NOTE(xcsong): print config for debugging
cat $train_config

echo "Start Training"
mkdir -p $dir

python3 wenet/bin/train.py --device-ids $gpu_ids \
    --dist_url $dist_url \
    --ddp.dist_backend gloo \
    --data_type "shard" \
    --config $train_config \
    --symbol_table  $data/dict/lang_char.txt \
    --train_data $data/train/data.list \
    --cv_data $data/dev/data.list \
    --model_dir $dir \
    --hdfs_dir $hdfs_dir \
    --hdfs_username $hdfs_username \
    --prefetch 100 \
    --num_workers 2 \
    --tensorboard_dir /job_tboard \
    --cmvn $data/train/global_cmvn \
    --pin_memory \
    --train_utt $data/train/wav.scp \
    ${checkpoint:+--checkpoint $checkpoint} \
    --finetune


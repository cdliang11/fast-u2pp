#!/usr/bin/env bash

# job.yaml related config
job_dir=k8s_job
num_node=1
job_name=aishell_fast_u2++_conformer_sweight0.5_distill0.015_schedule_ns_slayer7_handns_layer_11_12_finetune
# For cluster training, we always use 8 gpus
gpu_per_worker=8
gpu_ids="0,1,2,3,4,5,6,7"

# train_cluster.sh related config
data=data
train_config=conf/train_u2++_conformer.yaml
dir=exp/aishell_fast_u2++_conformer_sweight0.5_distill0.015_schedule_ns_slayer7_handns_layer_11_12_finetune
hdfs_dir=aishell_fast_u2++_conformer_sweight0.5_distill0.015_schedule_ns_slayer7_handns_layer_11_12_finetune

# decoding related config
decoding_chunk_size=-1
num_decoding_left_chunks=-1
ctc_weight=0.3
reverse_weight=0.3

. tools/parse_options.sh

mkdir -p $job_dir

echo "Step: Copy related files"
for f in $data conf path.sh train_cluster.sh decode.sh wenet tools ckpt; do
  cp -L -r $f $job_dir
done

echo "Step: Generate job.yaml"
cat <<EOF > job.yaml
REQUIRED:
  JOB_NAME: "torch-${job_name}"
  JOB_PASSWD: "newk8s666"
  UPLOAD_DIR: "${job_dir}"
  PROJECT_ID: "PDT2021005"
  WORKER_MIN_NUM: ${num_node}
  WORKER_MAX_NUM: ${num_node}
  GPU_PER_WORKER: ${gpu_per_worker}
  RUN_SCRIPTS: "\${WORKING_PATH}/job.sh"
OPTIONAL:
  PRIORITY: 5
  DOCKER_IMAGE: "docker.hobot.cc/imagesys/pytorch:cuda11.1-torch1.9.1-torchvision0.10.1-torchaudio0.9.1"
  WALL_TIME: 20000
  DATA_SPACE:
      DATA_TYPE: "dmp"
      INPUT: "speech"
      OUTPUT: "e-learning"
EOF

echo "Step: Generate job.sh"
total_gpu=$(expr $num_node \* $gpu_per_worker)
cat tools/job.sh | \
  sed -e "s:@TOTAL_GPU:${total_gpu}:g" | \
  sed -e "s:@GPU_PER_WORKER:${gpu_per_worker}:g" | \
  sed -e "s:@DATA:${data}:g" | \
  sed -e "s:@TRAIN_CONFIG:${train_config}:g" | \
  sed -e "s:@DIR:${dir}:g" | \
  sed -e "s:@DECODING_CHUNK_SIZE:${decoding_chunk_size}:g" | \
  sed -e "s:@NUM_DECODING_LEFT_CHUNKS:${num_decoding_left_chunks}:g" | \
  sed -e "s:@CTC_WEIGHT:${ctc_weight}:g" | \
  sed -e "s:@REVERSE_WEIGHT:${reverse_weight}:g" | \
  sed -e "s:@GPU_IDS:${gpu_ids}:g" | \
  sed -e "s:@HDFS_DIR:${hdfs_dir}:g" > $job_dir/job.sh
chmod +x $job_dir/job.sh

echo "Step: Submit job"
traincli submit -f job.yaml

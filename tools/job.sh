set -e
export PYTHONPATH=${WORKING_PATH}:$PYTHONPATH
date
env
export PYTHONUNBUFFERED=0
cd ${WORKING_PATH}
python3 tools/url2IP.py
cat /job_data/mpi_hosts
dis_url=$(head -n +1 /job_data/mpi_hosts)
mpirun -n @TOTAL_GPU -ppn @GPU_PER_WORKER --hostfile /job_data/mpi_hosts bash train_cluster.sh \
    --dist-url tcp://$dis_url:8000 \
    --data @DATA \
    --train_config @TRAIN_CONFIG \
    --dir @DIR \
    --gpu_ids @GPU_IDS \
    --hdfs_dir @HDFS_DIR

bash decode.sh \
  --dir @DIR \
  --average_checkpoint true \
  --gpuid 0 \
  --decoding_chunk_size @DECODING_CHUNK_SIZE \
  --num_decoding_left_chunks @NUM_DECODING_LEFT_CHUNKS \
  --ctc_weight @CTC_WEIGHT \
  --reverse_weight @REVERSE_WEIGHT \
  --average_num 20

# TODO(xcsong): upload test_xxxx files to hdfs_dir
echo "test_ctc_greedy_search/wer"
tail @DIR/test*ctc_greedy_search/wer
echo "test_ctc_prefix_beam_search/wer"
tail @DIR/test*ctc_prefix_beam_search/wer
echo "test_attention/wer"
tail @DIR/test*attention/wer
echo "test_attention_rescoring/wer"
tail @DIR/test*attention_rescoring/wer

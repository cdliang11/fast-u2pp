# hdfs_dir=hdfs://hobot-bigdata/user/chengdong.liang/aishell_fast_u2++_conformer
hdfs_dir=
gpuid=1
dir=/jfs-hdfs/user/chengdong.liang/fastu2pp_model/exp/aishell_fast_u2++_conformer_sweight0.5_distill0.015_schedule_ns_slayer7_handns_layer_11_12_finetune
# decode_modes="ctc_greedy_search ctc_prefix_beam_search attention_rescoring"
decode_modes="ctc_greedy_search"

average_checkpoint=true
average_num=30

epoch=359_latency

# decoding related parameter
s_decoding_chunk_size=4
b_decoding_chunk_size=24
s_num_decoding_left_chunks=-1
b_num_decoding_left_chunks=-1
ctc_weight=0.3
reverse_weight=0.5
beam_size=10

. path.sh
. tools/parse_options.sh

decode_checkpoint=$dir/final.pt

# Optional, download model dir from HDFS
if [ ! -z $hdfs_dir ]; then
  echo "download ckpt from k8s docker..."
  hdfs dfs -get $hdfs_dir exp
  echo "download ckpt from k8s docker... done."
fi


# Optional, do model average
if ${average_checkpoint}; then
  decode_checkpoint=$dir/avg_${average_num}.pt
  echo "do model average and final checkpoint is $decode_checkpoint"
  python3 wenet/bin/average_model.py \
    --dst_model $decode_checkpoint \
    --src_path $dir  \
    --num ${average_num} \
    --val_best
fi


echo "start decoding"
for mode in ${decode_modes}; do
{
  test_dir=$dir/test_${mode}_chunk${s_decoding_chunk_size}_${b_decoding_chunk_size}_left${s_num_decoding_left_chunks}_${b_num_decoding_left_chunks}_ctc${ctc_weight}_r${reverse_weight}_beam${beam_size}_ep${epoch}
  mkdir -p $test_dir
  python3 wenet/bin/recognize.py --gpu "$gpuid" \
    --mode $mode \
    --config $dir/train.yaml \
    --data_type "shard" \
    --test_data data/test/data.list \
    --checkpoint $decode_checkpoint \
    --beam_size $beam_size \
    --batch_size 1 \
    --penalty 0.0 \
    --dict data/dict/lang_char.txt \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_file $test_dir/text \
    ${s_num_decoding_left_chunks:+--s_num_decoding_left_chunks $s_num_decoding_left_chunks} \
    ${s_decoding_chunk_size:+--s_decoding_chunk_size $s_decoding_chunk_size} \
    ${b_num_decoding_left_chunks:+--b_num_decoding_left_chunks $b_num_decoding_left_chunks} \
    ${b_decoding_chunk_size:+--b_decoding_chunk_size $b_decoding_chunk_size} \
    --is_output_b_chunk \
    --is_lh_output_chunk
  python3 tools/compute-wer.py --char=1 --v=1 \
      data/test/text $test_dir/text > $test_dir/wer
  python3 tools/compute-wer.py --char=1 --v=1 \
      data/test/text $test_dir/text_b_chunk > $test_dir/wer_b_chunk
  python3 tools/compute-wer.py --char=1 --v=1 \
      data/test/text $test_dir/text_lh_chunk > $test_dir/wer_lh_chunk
} &
done
wait

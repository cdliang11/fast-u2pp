

gpuid=3
decode_modes='CTC_greedy_search'

dir=/jfs-hdfs/user/chengdong.liang/fastu2pp_model/exp/aishell_fast_u2++_conformer_sweight0.5_distill0.01_schedule_ns_slayer7_handns_layer_11_12_finetune
dir1=exp/aishell_fast_u2++_conformer_sweight0.5_distill0.05_schedule_ns_slayer7_handns_add_two_finetune
dir2=/jfs-hdfs/user/chengdong.liang/fastu2pp_model/exp/aishell_fast_u2++_conformer_sweight0.5_distill0.05_schedule_ns_slayer7_handns_layer_11_12_finetune
tags="甚至出现交易几乎停滞的情况"
wav="BAC009S0764W0121.wav"
chunk_size="8"
font_path=/home/users/chengdong.liang/anaconda3/envs/wenet/lib/python3.6/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf

. path.sh
. tools/parse_options.sh


python3 wenet/bin/plot_posterior_multi.py \
  --config $dir/train.yaml,$dir1/train.yaml,$dir2/train.yaml \
  --ckpts $dir/avg_30.pt,$dir1/175.pt,$dir2/147.pt \
  --tags $tags,$tags,$tags \
  --wav $wav \
  --chunk_size $chunk_size \
  --font_path $font_path \
  --dict_path data/dict/lang_char.txt \
  --result_file res_plot/plot_pttest_9_0.05_handns_two_finetune.png



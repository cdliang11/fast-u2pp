

# 输出 ali_word文件
def latency(ali_word_file):
    # 计算首字和尾字延迟
    # 返回 首字 和 尾字的位置
    utt2timesamp = {}
    with open(ali_word_file, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            line = line.strip().split()
            utt = line[0]
            ctc_out = line[1:]
            _start = 0
            _end = 0
            for idx, c in enumerate(ctc_out):
                if c != '<blank>':
                    _start = idx
                    break
            for i in range(len(ctc_out)-1, -1, -1):
                if ctc_out[i] != '<blank>':
                    _end = i
                    break
            utt2timesamp[utt] = [_start, _end]
    return utt2timesamp

def main(ali_word, asr_word):
    timestamp_1 = latency(ali_word)
    timestamp_2 = latency(asr_word)
    print(timestamp_1)
    print(timestamp_2)

    total_start_latency = 0
    total_end_latency = 0
    num = 0
    for key in timestamp_1.keys():
        total_start_latency += timestamp_2[key][0] * 4 - timestamp_1[key][0]
        total_end_latency += timestamp_2[key][1] * 4 - timestamp_1[key][1]
        num += 1
    print(total_end_latency)
    total_start_latency /= num
    total_end_latency /= num
    print(total_start_latency * 10)
    print(total_end_latency * 10)


if __name__ == '__main__':
    ali_word_file = 'exp/latency/ali_word.txt'
    # asr_word_file = 'exp/aishell_fast_u2++_conformer_sweight0.5_distill0.03_schedule_ns_slayer7_handns_add_two_finetune/test_ctc_greedy_search_chunk4_24_left-1_-1_ctc0.3_r0.5_beam10_ep359_latency/text_ctc_out'
    # asr_word_file = 'exp/handns_0.1_two_funetune/test_ctc_greedy_search_chunk4_24_left-1_-1_ctc0.3_r0.5_beam10_ep359_latency/text_ctc_out'
    # asr_word_file = 'exp/handns_0.0/test_ctc_greedy_search_chunk4_24_left-1_-1_ctc0.3_r0.5_beam10_ep359_latency/text_ctc_out'
    # asr_word_file = 'exp/baseline/test_ctc_greedy_search_chunk8_8_left-1_-1_ctc0.3_r0.5_beam10_ep359_latency/text_b_ctc_out'
    # asr_word_file = 'exp/aishell_fast_u2++_conformer_sweight0.5_distill0.02_schedule_ns_slayer7_handns_add_two_finetune/test_ctc_greedy_search_chunk4_24_left-1_-1_ctc0.3_r0.5_beam10_ep359_latency/text_ctc_out'
    # asr_word_file = '/jfs-hdfs/user/chengdong.liang/fastu2pp_model/exp/aishell_fast_u2++_conformer_sweight0.5_distill0.01_schedule_ns_slayer7_handns_layer_11_12_finetune'
    asr_word_file = '/jfs-hdfs/user/chengdong.liang/fastu2pp_model/exp/aishell_fast_u2++_conformer_sweight0.5_distill0.02_schedule_ns_slayer7_handns_layer_11_12_finetune'
    main(ali_word_file, asr_word_file)

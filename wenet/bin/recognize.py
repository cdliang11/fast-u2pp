# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
from http.client import NotConnected
from json import encoder
import logging
from multiprocessing import context
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table, read_non_lang_symbols
from wenet.utils.config import override_config

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument("--non_lang_syms",
                        help="non-linguistic symbol file. One symbol per line.")
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--pad_num',
                        type=int,
                        default=-1,
                        help='num of padded zeros')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring'
                        ],
                        default='attention',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--s_decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--s_num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--b_decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--b_num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--is_output_b_chunk',
                        action='store_true',
                        help='')
    parser.add_argument('--is_lh_output_chunk',
                        action='store_true',
                        help='')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
    elif 'mfcc_conf' in test_conf:
        test_conf['mfcc_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = "static"
    test_conf['batch_conf']['batch_size'] = args.batch_size
    non_lang_syms = read_non_lang_symbols(args.non_lang_syms)

    test_dataset = Dataset(args.data_type,
                           args.test_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           non_lang_syms,
                           partition=False)

    test_data_loader = DataLoader(test_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_asr_model(configs)

    # Load dict
    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1

    load_checkpoint(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)

    model.eval()
    if args.is_lh_output_chunk:
        lh_out = open(args.result_file + '_lh_chunk', 'w', encoding='utf-8')
    if args.is_output_b_chunk:
        b_fout = open(args.result_file + '_b_chunk', 'w', encoding='utf-8')
    # ctc out for latency
    ctc_out = open(args.result_file + '_ctc_out', 'w', encoding='utf-8')
    b_ctc_out = open(args.result_file + '_b_ctc_out', 'w', encoding='utf-8')
    with torch.no_grad(), open(args.result_file, 'w', encoding="utf-8") as fout:
        for batch_idx, batch in enumerate(test_data_loader):
            keys, feats, target, feats_lengths, target_lengths = batch
            num_bins = test_conf['fbank_conf']['num_mel_bins']
            if args.pad_num > 0:
                feats_pad = torch.zeros((1, args.pad_num, num_bins))
                feats = torch.cat((feats, feats_pad), dim=1)
                feats_lengths = feats_lengths + args.pad_num
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            if args.mode == 'attention':
                hyps, _ = model.recognize(
                    feats,
                    feats_lengths,
                    beam_size=args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp.tolist() for hyp in hyps]
            elif args.mode == 'ctc_greedy_search':
                s_hyps, b_hyps, lh_hyps, ori_hyps, b_ori_hyps = model.ctc_greedy_search(
                    feats,
                    feats_lengths,
                    s_decoding_chunk_size=args.s_decoding_chunk_size,
                    s_num_decoding_left_chunks=args.s_num_decoding_left_chunks,
                    b_decoding_chunk_size=args.b_decoding_chunk_size,
                    b_num_decoding_left_chunks=args.b_num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                    is_output_b_chunk=args.is_output_b_chunk,
                    is_lh_output_chunk=args.is_lh_output_chunk)
            # ctc_prefix_beam_search and attention_rescoring only return one
            # result in List[int], change it to List[List[int]] for compatible
            # with other batch decoding mode
            elif args.mode == 'ctc_prefix_beam_search':
                assert (feats.size(0) == 1)
                s_hyp, b_hyp, lh_hyp = model.ctc_prefix_beam_search(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    s_decoding_chunk_size=args.s_decoding_chunk_size,
                    s_num_decoding_left_chunks=args.s_num_decoding_left_chunks,
                    b_decoding_chunk_size=args.b_decoding_chunk_size,
                    b_num_decoding_left_chunks=args.b_num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming,
                    is_output_b_chunk=args.is_output_b_chunk,
                    is_lh_output_chunk=args.is_lh_output_chunk)
                s_hyps = [s_hyp]
                b_hyps = [b_hyp]
                lh_hyps = [lh_hyp]
                ori_hyps = None
                b_ori_hyps = None
            elif args.mode == 'attention_rescoring':
                assert (feats.size(0) == 1)
                s_hyp, b_hyp, lh_hyp = model.attention_rescoring(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    s_decoding_chunk_size=args.s_decoding_chunk_size,
                    s_num_decoding_left_chunks=args.s_num_decoding_left_chunks,
                    b_decoding_chunk_size=args.b_decoding_chunk_size,
                    b_num_decoding_left_chunks=args.b_num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight,
                    is_output_b_chunk=args.is_output_b_chunk,
                    is_lh_output_chunk=args.is_lh_output_chunk)
                s_hyps = [s_hyp]
                b_hyps = [b_hyp]
                lh_hyps = [lh_hyp]
                ori_hyps = None
                b_ori_hyps = None
            for i, key in enumerate(keys):
                content = ''
                for w in s_hyps[i]:
                    if w == eos:
                        break
                    content += char_dict[w]
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))
                if args.is_lh_output_chunk:
                    content = ''
                    for w in lh_hyps[i]:
                        if w == eos:
                            break
                        content += char_dict[w]
                    logging.info('{} {}'.format(key, content))
                    lh_out.write('{} {}\n'.format(key, content))
                if args.is_output_b_chunk:
                    content = ''
                    for w in b_hyps[i]:
                        if w == eos:
                            break
                        content += char_dict[w]
                logging.info('{} {}'.format(key, content))
                b_fout.write('{} {}\n'.format(key, content))
                if ori_hyps is not None:
                    ctc_out.write('{}'.format(key))
                    for w in ori_hyps[i]:
                        ctc_out.write(' {}'.format(char_dict[w]))
                    ctc_out.write('\n')
                if b_ori_hyps is not None:
                    b_ctc_out.write('{}'.format(key))
                    for w in b_ori_hyps[i]:
                        b_ctc_out.write(' {}'.format(char_dict[w]))
                    b_ctc_out.write('\n')
    if args.is_lh_output_chunk:
        lh_out.close()
    if args.is_output_b_chunk:
        b_fout.close()
    ctc_out.close()
    b_ctc_out.close()

if __name__ == '__main__':
    main()

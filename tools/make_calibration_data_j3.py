#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2022-04-25] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
"""Make Calibration Data."""

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from wenet.dataset.dataset import Dataset
from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table


def prepare_data(tensor, dirs, prefix):
    if tensor.requires_grad:
        data = tensor.detach().numpy().astype(np.float32)
    else:
        data = tensor.numpy().astype(np.float32)
    os.makedirs(dirs, exist_ok=True)
    data.tofile(dirs + "/" + prefix + ".bin")


def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--calibration_data',
                        required=True,
                        help='calibration data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard', 'only_wav'],
                        help='calibration data type')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='saving data')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--chunk_size', required=True,
                        type=int, help='decoding chunk size')
    parser.add_argument('--left_chunks', required=True,
                        type=int, help='cache chunks')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--num_sentences',
                        default=2000,
                        type=int, help='number of sentences')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str("-1")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['fbank_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_type'] = 'static'
    test_conf['batch_conf']['batch_size'] = 1

    cali_dataset = Dataset(args.data_type,
                           args.calibration_data,
                           symbol_table,
                           test_conf,
                           args.bpe_model,
                           partition=False)
    cali_data_loader = DataLoader(cali_dataset, batch_size=None, num_workers=0)

    # Init asr model from configs
    model = init_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    print(model)

    for batch_idx, batch in enumerate(cali_data_loader):
        keys, feats, target, feats_lengths, target_lengths = batch
        subsampling = model.encoder.embed.subsampling_rate
        context = model.encoder.embed.right_context + 1  # Add current frame
        stride = subsampling * args.chunk_size
        decoding_window = (args.chunk_size - 1) * subsampling + context
        num_frames = feats.size(1)
        prefix = keys[0]
        offset = -1
        required_cache_size = args.chunk_size * args.left_chunks
        att_cache = torch.zeros(
            [1, model.encoder.encoders[0].attn.h * len(model.encoder.encoders),
             required_cache_size, model.encoder.encoders[0].attn.d_k * 2],
            dtype=feats.dtype, device=feats.device)
        cnn_cache = torch.zeros(
            [1, model.encoder.output_size() * len(model.encoder.encoders),
             1, model.encoder.encoders[0].conv_module.lorder],
            dtype=feats.dtype, device=feats.device)
        att_mask = torch.ones(
            [1, model.encoder.encoders[0].attn.h, args.chunk_size,
             required_cache_size + args.chunk_size],
            dtype=feats.dtype, device=feats.device)
        att_mask[:, :, :, :required_cache_size] = 0

        # Feed forward overlap input step by step
        for i, cur in enumerate(range(0, num_frames - context + 1, stride)):
            att_mask[:, :, :, -(args.chunk_size * (i + 1)):] = 1
            end = min(cur + decoding_window, num_frames)
            chunk = feats[:, cur:end, :].unsqueeze(0)  # (1, 1, window, mel)
            if end == num_frames and end - cur < decoding_window:  # last chunk
                pad_len = decoding_window - (end - cur)
                pad_chunk = torch.zeros((1, 1, pad_len, chunk.size(-1)))
                chunk = torch.cat((chunk, pad_chunk), dim=2)  # (1, 1, win, mel)
                att_mask[:, :, :, -(pad_len // subsampling):] = 0
            prepare_data(chunk, args.output_dir + "/chunk",
                         prefix + "." + str(i))
            prepare_data(att_cache, args.output_dir + "/att_cache",
                         prefix + "." + str(i))
            prepare_data(cnn_cache, args.output_dir + "/cnn_cache",
                         prefix + "." + str(i))
            prepare_data(att_mask, args.output_dir + "/att_mask",
                         prefix + "." + str(i))
            (y, att_cache, cnn_cache) = model.encoder.forward_chunk_j3(
                chunk, offset, required_cache_size,
                att_cache, cnn_cache, att_mask)
            prepare_data(y, args.output_dir + "/output",
                         prefix + "." + str(i))
        if batch_idx % 100 == 0:
            print("process {} utts.".format(batch_idx))
        if batch_idx > args.num_sentences:
            break


if __name__ == '__main__':
    main()

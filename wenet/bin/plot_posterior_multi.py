# Copyright (c) 2022 Horizon Inc. (authors: Binbin Zhang, Xingchen Song)
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
import os

import torch
import yaml
import torchaudio
import librosa

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import torchaudio.compliance.kaldi as kaldi

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.mask import make_pad_mask


def get_args():
    parser = argparse.ArgumentParser(description='plot posterior')
    parser.add_argument('--configs',
                        required=True,
                        type=str,
                        help='list of config')
    parser.add_argument('--ckpts',
                        required=True,
                        type=str,
                        help='list of checkpoint')
    parser.add_argument('--tags', required=True, type=str, help='list of tag')
    parser.add_argument('--wav', required=True, type=str, help='wav file')
    parser.add_argument('--chunk_size',
                        required=True,
                        type=str,
                        help='chunk size')
    parser.add_argument('--font_path',
                        required=True,
                        type=str,
                        help='font file')
    parser.add_argument('--dict_path',
                        required=True,
                        type=str,
                        help='dict file')
    parser.add_argument('--result_file',
                        required=True,
                        type=str,
                        help='saving pdf')
    args = parser.parse_args()
    return args


def ctc_greedy(encoder_out, encoder_mask, model, batch_size, eos, is_l_ctc=False):
    maxlen = encoder_out.size(1)
    encoder_out_lens = encoder_mask.squeeze(1).sum(1)
    if is_l_ctc:
        ctc_probs = model.l_ctc.log_softmax(encoder_out)
    else:
        ctc_probs = model.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
    topk_prob = topk_prob.view(batch_size, maxlen)  # (B, maxlen)
    mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, eos)  # (B, maxlen)
    topk_prob = topk_prob.masked_fill_(mask, 0.0)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    print(hyps[0])
    scores = [prob.tolist() for prob in topk_prob]
    return hyps, scores




def main():
    args = get_args()
    torch.manual_seed(777)
    torch.set_num_threads(1)
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    symbol_table = read_symbol_table(args.dict_path)
    char_dict = {v: k for k, v in symbol_table.items()}
    eos = len(char_dict) - 1

    # split by comma
    # >>> configs = "exp/model1,exp/model2,exp/model3"
    # >>> ckpts = "exp/model1/final.pt,exp/model2/final.pt,exp/model3/final.pt"
    # >>> tags = "model1,model2,model3"
    config_list = args.configs.split(',')
    ckpt_list = args.ckpts.split(',')
    tag_list = args.tags.split(',')
    chunk_size_list = args.chunk_size.split(',')
    assert len(ckpt_list) == len(config_list)
    assert len(ckpt_list) == len(tag_list)
    if len(chunk_size_list) == 1:
        chunk_size = int(chunk_size_list[0])
        chunk_size_list = [chunk_size] * len(ckpt_list)
    else:
        chunk_size_list = [int(i) for i in chunk_size_list]

    font = fm.FontProperties(fname=args.font_path)
    plt.rcParams['axes.unicode_minus'] = False
    # we will have len(tag_list) sub-plots plus one wav-plot
    len_pig = 3
    fig, axes = plt.subplots(figsize=(60, 60),
                             nrows=len_pig + 1,
                             ncols=1)

    subsampling = -1

    mats = [None] * len(config_list)
    for i, (config, ckpt, tag, chunk) in enumerate(
            zip(config_list, ckpt_list, tag_list, chunk_size_list)):
        print("processing {}.".format(ckpt))
        with open(config, 'r') as fin:
            conf = yaml.load(fin, Loader=yaml.FullLoader)
        # conf['model_conf']['dual'] = False
        model = init_asr_model(conf)
        load_checkpoint(model, ckpt)
        model.eval()
        subsampling = model.encoder.embed.subsampling_rate
        print(subsampling)
        wav = args.wav
        waveform, sr = torchaudio.load(wav)
        print(waveform)
        resample_rate = conf['dataset_conf']['resample_conf']['resample_rate']
        waveform = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=resample_rate)(waveform)
        waveform = waveform * (1 << 15)
        print(waveform)
        # Only keep key, feat, label
        mat = kaldi.fbank(
            waveform,
            num_mel_bins=conf['dataset_conf']['fbank_conf']['num_mel_bins'],
            frame_length=conf['dataset_conf']['fbank_conf']['frame_length'],
            frame_shift=conf['dataset_conf']['fbank_conf']['frame_shift'],
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=resample_rate,
        )
        mats[i] = mat
        mat = mats[0]
        print(mat)

        # CTC greedy search
        with torch.no_grad():
            assert chunk != 0
            batch_size = 1
            speech = mat.unsqueeze(0)
            speech_lengths = torch.tensor([mat.size(0)])
            # Let's assume B = batch_size
            l_encoder_out, h_encoder_out, lh_encoder_out, encoder_mask = model._forward_encoder(
                speech, speech_lengths, chunk, -1, chunk, -1, simulate_streaming=False)
            print(h_encoder_out)
            # (B, maxlen, encoder_dim)
            llist = ['l_encoder_out', 'h_encoder_out', 'lh_encoder_out']
            title = ['add_two chunk8', 'add_one chunk8', 'u2++ baseline']
            for j, encoder_out in enumerate([l_encoder_out]):
                if j == 0:
                    if i == 2:
                        encoder_out = h_encoder_out
                    hyps, scores = ctc_greedy(encoder_out, encoder_mask, model, batch_size, eos, is_l_ctc=conf['model_conf']['is_l_ctc'])
                else:
                    hyps, scores = ctc_greedy(encoder_out, encoder_mask, model, batch_size, eos, is_l_ctc=False)
            # maxlen = h_encoder_out.size(1)
            # encoder_out_lens = encoder_mask.squeeze(1).sum(1)
            # ctc_probs = model.ctc.log_softmax(
            #     h_encoder_out)  # (B, maxlen, vocab_size)
            # topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
            # topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
            # topk_prob = topk_prob.view(batch_size, maxlen)  # (B, maxlen)
            # mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
            # topk_index = topk_index.masked_fill_(mask, eos)  # (B, maxlen)
            # topk_prob = topk_prob.masked_fill_(mask, 0.0)  # (B, maxlen)
            # hyps = [hyp.tolist() for hyp in topk_index]
            # print(hyps[0])
            # scores = [prob.tolist() for prob in topk_prob]

                x = np.arange(len(hyps[0])) * subsampling
                # axes[j].set_title(tag, fontsize=60, font=font)
                axes[j+i].set_title(title[i], fontsize=30)
                for frame, token, prob in zip(x, hyps[0], scores[0]):
                    if char_dict[token] != '<blank>':
                        axes[j+i].bar(
                            frame,
                            np.exp(prob),
                            label='{} {:.3f}'.format(char_dict[token],
                                                    np.exp(prob)),
                        )
                        axes[j+i].text(
                            frame,
                            np.exp(prob),
                            '{} {:.3f} {}'.format(char_dict[token], np.exp(prob),
                                                frame),
                            fontdict=dict(fontsize=24),
                            fontproperties=font,
                        )
                    else:
                        axes[j+i].bar(
                            frame,
                            0.01,
                            label='{} {:.3f}'.format(char_dict[token],
                                                    np.exp(prob)),
                        )

                # x_major_locator = MultipleLocator(subsampling)
                # axes[i].xaxis.set_major_locator(x_major_locator)
                axes[j+i].tick_params(labelsize=25)
                # axes[i].set_xticklabels(x, rotation=270)

    # wav, hardcode sample_rate to 16000
    samples, sr = librosa.load(args.wav, sr=16000)
    time = np.arange(0, len(samples)) * (1.0 / sr)
    axes[-1].plot(time, samples)

    plt.savefig(args.result_file)


if __name__ == '__main__':
    main()

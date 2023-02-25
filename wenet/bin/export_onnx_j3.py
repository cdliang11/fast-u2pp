#!/usr/bin/env python3
# Copyright (c) 2022, Xingchen Song (sxc19@mails.tsinghua.edu.cn)
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
import copy
import sys

import torch
import yaml
import numpy as np

from wenet.transformer.asr_model import init_asr_model
from wenet.utils.checkpoint import load_checkpoint

try:
    import onnx
    import onnxruntime
except ImportError:
    print('Please install onnxruntime!')
    sys.exit(1)


class J3CTC(torch.nn.Module):
    """CTC module"""
    def __init__(self, odim: int, idim: int):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            idim: number of encoder projection units
        """
        super().__init__()
        self.proj = torch.nn.ModuleList()
        self.split_size = []
        if odim > 4096:
            for idx in range(odim // 2048):
                self.proj.append(torch.nn.Conv2d(
                    idim, 2048, 1, 1,
                ))
                self.split_size.append(2048)
            if odim % 2048 > 0:
                self.proj.append(
                    torch.nn.Conv2d(idim, odim % 2048, 1, 1)
                )
                self.split_size.append(odim % 2048)
        else:
            self.proj.append(torch.nn.Conv2d(idim, odim, 1, 1))
            self.split_size.append(odim)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 4d tensor (B, eprojs, 1, Tmax)
        Returns:
            torch.Tensor: softmax applied 3d tensor (B, odim, 1, Tmax)
        """
        out = []
        for i, layer in enumerate(self.proj):
            out.append(layer(hs_pad))
        out = torch.cat(out, dim=1)
        return self.softmax(out)


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--chunk_size', required=True,
                        type=int, help='decoding chunk size')
    parser.add_argument('--num_decoding_left_chunks', required=True,
                        type=int, help='cache chunks')
    parser.add_argument('--beam', required=True,
                        type=int, help='beam wigth')
    parser.add_argument('--reverse_weight', default=0.0,
                        type=float, help='reverse_weight in attention_rescoing')
    args = parser.parse_args()
    return args


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def export_encoder(asr_model, args):
    print("Stage-1: export encoder")
    encoder = asr_model.encoder
    encoder.forward = encoder.forward_chunk_j3
    encoder_outpath = os.path.join(args['output_dir'], 'encoder.onnx')

    print("\tStage-1.1: prepare inputs for encoder")
    chunk = torch.randn(
        (args['batch'], 1, args['decoding_window'], args['feature_size']))
    assert args['left_chunks'] > 0
    assert args['chunk_size'] > 0
    offset = -1
    required_cache_size = args['chunk_size'] * args['left_chunks']
    # Real cache
    att_cache = torch.zeros(
        (1, args['num_blocks'] * args['head'], required_cache_size,
         args['output_size'] // args['head'] * 2))
    cnn_cache = torch.zeros(
        (1, args['num_blocks'] * args['output_size'],
         1, args['cnn_module_kernel'] - 1))
    # Real mask
    att_mask = torch.ones(
        (args['batch'], args['head'], args['chunk_size'],
         required_cache_size + args['chunk_size']))
    att_mask[:, :, :, :required_cache_size] = 0

    inputs = (chunk, offset, required_cache_size,
              att_cache, cnn_cache, att_mask)
    print("\t\tchunk.size(): {}\n".format(chunk.size()),
          "\t\toffset: {}\n".format(offset),
          "\t\trequired_cache: {}\n".format(required_cache_size),
          "\t\tatt_cache.size(): {}\n".format(att_cache.size()),
          "\t\tcnn_cache.size(): {}\n".format(cnn_cache.size()),
          "\t\tatt_mask.size(): {}\n".format(att_mask.size()))

    print("\tStage-1.2: torch.onnx.export")
    torch.onnx.export(
        encoder, inputs, encoder_outpath, opset_version=11,
        export_params=True, do_constant_folding=True,
        input_names=[
            'chunk', 'offset', 'required_cache_size',
            'att_cache', 'cnn_cache', 'att_mask'
        ],
        output_names=['output', 'r_att_cache', 'r_cnn_cache'],
        dynamic_axes=None, verbose=False)
    onnx_encoder = onnx.load(encoder_outpath)
    for (k, v) in args.items():
        meta = onnx_encoder.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    onnx.checker.check_model(onnx_encoder)
    onnx.helper.printable_graph(onnx_encoder.graph)
    # NOTE(xcsong): to add those metadatas we need to reopen
    #   the file and resave it.
    onnx.save(onnx_encoder, encoder_outpath)
    print("\t\tonnx_encoder inputs : {}".format(
        [node.name for node in onnx_encoder.graph.input]))
    print("\t\tonnx_encoder outputs: {}".format(
        [node.name for node in onnx_encoder.graph.output]))
    print('\t\tExport onnx_encoder, done! see {}'.format(
        encoder_outpath))

    print("\tStage-1.3: check onnx_encoder and torch_encoder")
    torch_output = []
    torch_chunk = copy.deepcopy(chunk)
    torch_offset = copy.deepcopy(offset)
    torch_required_cache_size = copy.deepcopy(required_cache_size)
    torch_att_cache = copy.deepcopy(att_cache)
    torch_cnn_cache = copy.deepcopy(cnn_cache)
    torch_att_mask = copy.deepcopy(att_mask)
    for i in range(10):
        print("\t\ttorch chunk-{}: {}, offset: {}, att_cache: {},"
              " cnn_cache: {}, att_mask: {}".format(
                  i, list(torch_chunk.size()), torch_offset,
                  list(torch_att_cache.size()),
                  list(torch_cnn_cache.size()), list(torch_att_mask.size())))
        torch_att_mask[:, :, :, -(args['chunk_size'] * (i + 1)):] = 1
        out, torch_att_cache, torch_cnn_cache = encoder(
            torch_chunk, torch_offset, torch_required_cache_size,
            torch_att_cache, torch_cnn_cache, torch_att_mask)
        torch_output.append(out)
    torch_output = torch.cat(torch_output, dim=1)

    onnx_output = []
    onnx_chunk = to_numpy(chunk)
    onnx_offset = np.array((offset)).astype(np.int64)
    onnx_required_cache_size = np.array((required_cache_size)).astype(np.int64)
    onnx_att_cache = to_numpy(att_cache)
    onnx_cnn_cache = to_numpy(cnn_cache)
    onnx_att_mask = to_numpy(att_mask)
    ort_session = onnxruntime.InferenceSession(encoder_outpath)
    for i in range(10):
        print("\t\tonnx  chunk-{}: {}, offset: {}, att_cache: {},"
              " cnn_cache: {}, att_mask: {}".format(
                  i, onnx_chunk.shape, onnx_offset, onnx_att_cache.shape,
                  onnx_cnn_cache.shape, onnx_att_mask.shape))
        onnx_att_mask[:, :, :, -(args['chunk_size'] * (i + 1)):] = 1
        ort_inputs = {
            'chunk': onnx_chunk, 'offset': onnx_offset,
            'required_cache_size': onnx_required_cache_size,
            'att_cache': onnx_att_cache,
            'cnn_cache': onnx_cnn_cache, 'att_mask': onnx_att_mask
        }
        if 'conformer' not in args['encoder']:
            ort_inputs.pop('cnn_cache')  # Transformer
        if 'cnn' in args['encoder']:
            ort_inputs.pop('required_cache_size')
            ort_inputs.pop('offset')
        ort_outs = ort_session.run(None, ort_inputs)
        onnx_att_cache, onnx_cnn_cache = ort_outs[1], ort_outs[2]
        onnx_output.append(ort_outs[0])
    onnx_output = np.concatenate(onnx_output, axis=1)

    np.testing.assert_allclose(to_numpy(torch_output), onnx_output,
                               rtol=1e-03, atol=1e-05)
    meta = ort_session.get_modelmeta()
    print("\t\tcustom_metadata_map={}".format(meta.custom_metadata_map))
    print("\t\tCheck onnx_encoder, pass!")


def export_ctc(asr_model, args):
    print("Stage-2: export ctc")
    ctc = J3CTC(odim=args['vocab_size'], idim=args['output_size'])
    weight = torch.split(asr_model.ctc.ctc_lo.weight, ctc.split_size, dim=0)
    bias = torch.split(asr_model.ctc.ctc_lo.bias, ctc.split_size, dim=0)
    for i, (w, b) in enumerate(zip(weight, bias)):
        ctc.proj[i].weight = torch.nn.Parameter(w.unsqueeze(-1).unsqueeze(-1))
        ctc.proj[i].bias = torch.nn.Parameter(b)
    ctc_outpath = os.path.join(args['output_dir'], 'ctc.onnx')

    print("\tStage-2.1: prepare inputs for ctc")
    hidden = torch.randn(
        (args['batch'], args['output_size'], 1,
         args['chunk_size'] if args['chunk_size'] > 0 else 16))
    torch.allclose(
        ctc(hidden).squeeze(2).transpose(1, 2),
        asr_model.ctc.log_softmax(hidden.squeeze(2).transpose(1, 2))
    )

    print("\tStage-2.2: torch.onnx.export")
    torch.onnx.export(
        ctc, hidden, ctc_outpath, opset_version=11,
        export_params=True, do_constant_folding=True,
        input_names=['hidden'], output_names=['probs'],
        dynamic_axes=None, verbose=False)
    onnx_ctc = onnx.load(ctc_outpath)
    for (k, v) in args.items():
        meta = onnx_ctc.metadata_props.add()
        meta.key, meta.value = str(k), str(v)
    onnx.checker.check_model(onnx_ctc)
    onnx.helper.printable_graph(onnx_ctc.graph)
    # NOTE(xcsong): to add those metadatas we need to reopen
    #   the file and resave it.
    onnx.save(onnx_ctc, ctc_outpath)
    print("\t\tonnx_ctc inputs : {}".format(
        [node.name for node in onnx_ctc.graph.input]))
    print("\t\tonnx_ctc outputs: {}".format(
        [node.name for node in onnx_ctc.graph.output]))
    print('\t\tExport onnx_ctc, done! see {}'.format(
        ctc_outpath))

    print("\tStage-2.3: check onnx_ctc and torch_ctc")
    torch_output = ctc(hidden)
    ort_session = onnxruntime.InferenceSession(ctc_outpath)
    onnx_output = ort_session.run(None, {'hidden' : to_numpy(hidden)})

    np.testing.assert_allclose(to_numpy(torch_output), onnx_output[0],
                               rtol=1e-03, atol=1e-05)
    print("\t\tCheck onnx_ctc, pass!")


def main():
    torch.manual_seed(777)
    args = get_args()
    output_dir = args.output_dir
    os.system("mkdir -p " + output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    model = init_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    model.eval()
    print(model)

    arguments = {}
    arguments['output_dir'] = output_dir
    arguments['batch'] = 1
    arguments['chunk_size'] = args.chunk_size
    arguments['left_chunks'] = args.num_decoding_left_chunks
    arguments['beam'] = args.beam
    arguments['reverse_weight'] = args.reverse_weight
    arguments['output_size'] = configs['encoder_conf']['output_size']
    arguments['num_blocks'] = configs['encoder_conf']['num_blocks']
    arguments['cnn_module_kernel'] = configs['encoder_conf']['cnn_module_kernel']
    arguments['head'] = configs['encoder_conf']['attention_heads']
    arguments['feature_size'] = configs['input_dim']
    arguments['vocab_size'] = configs['output_dim']
    # NOTE(xcsong): if chunk_size == -1, hardcode to 67
    arguments['decoding_window'] = (args.chunk_size - 1) * \
        model.encoder.embed.subsampling_rate + \
        model.encoder.embed.right_context + 1 if args.chunk_size > 0 else 67
    arguments['encoder'] = configs['encoder']
    arguments['decoder'] = configs['decoder']
    arguments['subsampling_rate'] = model.subsampling_rate()
    arguments['right_context'] = model.right_context()
    arguments['sos_symbol'] = model.sos_symbol()
    arguments['eos_symbol'] = model.eos_symbol()
    arguments['is_bidirectional_decoder'] = 1 \
        if model.is_bidirectional_decoder() else 0

    export_encoder(model, arguments)
    export_ctc(model, arguments)
    # TODO(pengshen): add decoder

if __name__ == '__main__':
    main()

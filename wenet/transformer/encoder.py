#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
# Author: di.wu@mobvoi.com (DI WU)
"""Encoder definition."""

from typing import List, Tuple

import torch
from typeguard import check_argument_types

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.attention import RelPositionMultiHeadedAttention
from wenet.transformer.convolution import ConvolutionModule, DualConvolutionModule
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.embedding import RelPositionalEncoding
from wenet.transformer.embedding import NoPositionalEncoding
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.transformer.encoder_layer import DFSMNEncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.subsampling import Conv2dSubsampling4
from wenet.transformer.subsampling import Conv2dSubsampling6
from wenet.transformer.subsampling import Conv2dSubsampling8
from wenet.transformer.subsampling import LinearNoSubsampling
from wenet.utils.common import get_activation
from wenet.utils.mask import gen_dynamic_chunk_mask, make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask


class FSMNEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 512,
        num_blocks: int = 10,
    ):

        super().__init__()
        self._output_size = output_size
        self.wins = 7
        self.num_blocks = num_blocks
        channels = [2048, 2048, 2048, 512]
        fch = [2048 for _ in range(num_blocks)]

        ks = (1, 3)
        self.conv0 = torch.nn.Conv2d(input_size, channels[0], ks,
                                     stride=(1, 1), padding=(0, 0),
                                     bias=True)
        self.batch_norm0 = torch.nn.BatchNorm2d(channels[0])

        ks = (self.wins, 1)
        self.dfsmn = torch.nn.ModuleList([
            DFSMNEncoderLayer(
                in_channel=fch[i],
                wins=ks,
                layer_index=i,
            ) for i in range(self.num_blocks)
        ])

        self.conv1 = torch.nn.Conv2d(channels[0], channels[1], (1, 1),
                                     stride=(1, 1), padding=(0, 0),
                                     bias=True)
        self.batch_norm1 = torch.nn.BatchNorm2d(channels[1])
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(channels[1], channels[2], (1, 1),
                                     stride=(1, 1), padding=(0, 0),
                                     bias=True)
        self.batch_norm2 = torch.nn.BatchNorm2d(channels[2])
        self.relu2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv2d(channels[2], channels[3], (1, 1),
                                     stride=(1, 1), padding=(0, 0),
                                     bias=True)

    def forward(
        self,
        inputs: torch.Tensor,
        xs_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        masks = ~make_pad_mask(xs_lens).unsqueeze(1)  # (B, 1, L)
        # layer_mask = masks.unsqueeze(-1)

        wins = self.wins - 2
        state = None
        outputs = self.batch_norm0(self.conv0(inputs))
        for i, dfsmn in enumerate(self.dfsmn):
            outputs, state = dfsmn(outputs, state)
            outputs = torch.nn.functional.dropout(outputs, 0.1)
        outputs = self.relu1(self.batch_norm1(self.conv1(outputs)))
        outputs = self.relu2(self.batch_norm2(self.conv2(outputs)))
        outputs = self.conv3(outputs)
        outputs = outputs.squeeze(-1)  # (B, 512, L, 1) -> (B, 512, L)
        outputs = torch.transpose(outputs, 1, 2)

        return outputs, masks


    def output_size(self) -> int:
        return self._output_size

class BaseEncoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        add_l_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        s_static_chunk_size: int = 0,
        b_static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        s_ctc_layer: int = 7,
        two_chunk: bool = False,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            concat_after (bool): whether to concat attention layer's input
                and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
        """
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.add_l_num_blocks = add_l_num_blocks
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            input_size,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.s_static_chunk_size = s_static_chunk_size
        self.b_static_chunk_size = b_static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.s_ctc_layer = s_ctc_layer
        self.two_chunk = two_chunk

        # can only be called and changed by self.set_dual_mode
        self.dual = False
        self.streaming = True

    def output_size(self) -> int:
        return self._output_size

    def set_dual_mode(self, streaming: bool = False):
        self.dual = True
        self.streaming = streaming

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        s_decoding_chunk_size: int = 0,
        s_num_decoding_left_chunks: int = -1,
        b_decoding_chunk_size: int = 0,
        b_num_decoding_left_chunks: int = -1,
        output_cache: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsample_rate)

        if output_cache is not None:
            output_layer_list = output_cache.split(',')
        else:
            output_layer_list = []

        assert self.two_chunk == False
        if self.two_chunk:
            s_xs = xs
            b_xs = xs
            s_chunk_masks = add_optional_chunk_mask(s_xs, masks,
                                                    self.use_dynamic_chunk,
                                                    self.use_dynamic_left_chunk,
                                                    s_decoding_chunk_size,
                                                    self.s_static_chunk_size,
                                                    s_num_decoding_left_chunks)
            b_chunk_masks = add_optional_chunk_mask(b_xs, masks,
                                                    self.use_dynamic_chunk,
                                                    self.use_dynamic_left_chunk,
                                                    b_decoding_chunk_size,
                                                    self.b_static_chunk_size,
                                                    b_num_decoding_left_chunks)
        else:
            s_xs = xs
            b_xs = None
            if not self.dual:
                s_chunk_masks = add_optional_chunk_mask(s_xs, masks,
                                                        self.use_dynamic_chunk,
                                                        self.use_dynamic_left_chunk,
                                                        s_decoding_chunk_size,
                                                        self.s_static_chunk_size,
                                                        s_num_decoding_left_chunks)
            else:
                # dual training of streaming and non-streaming
                if self.streaming:
                    s_chunk_masks = gen_dynamic_chunk_mask(xs, masks)
                else:
                    s_chunk_masks = masks
            b_chunk_masks = s_chunk_masks
        # NOTE: small_ctc_output_layer:
        # 0 - s_ctc_layer
        b_layer_out = []
        add_s_layer_out = []
        if not self.two_chunk:
            for layer in self.encoders[:self.s_ctc_layer - self.add_l_num_blocks]:
                s_xs, s_chunk_masks, _, _ = layer(s_xs, s_chunk_masks, pos_emb, mask_pad)
            b_xs = s_xs
            i = 6
            for layer in self.encoders[self.s_ctc_layer - self.add_l_num_blocks:]:
                b_xs, b_chunk_masks, _, _ = layer(b_xs, b_chunk_masks, pos_emb, mask_pad)
                if str(i) in output_layer_list:
                    b_layer_out.append(b_xs)
                i += 1
            # add l_encoder
            if self.add_l_num_blocks != 0:
                for layer in self.l_encoders:
                    s_xs, s_chunk_masks, _, _ = layer(s_xs, s_chunk_masks, pos_emb, mask_pad)
                    add_s_layer_out.append(s_xs)

        # s_ctc_layer - last_layer
        # for layer in self.encoders:
        #     b_xs, b_chunk_masks, _, _ = layer(b_xs, b_chunk_masks,
        #                                          pos_emb, mask_pad)

        if self.normalize_before:
            s_xs = self.after_norm(s_xs)
            b_xs = self.after_norm(b_xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return s_xs, mask_pad, b_xs, mask_pad, b_layer_out, add_s_layer_out

    def forward_two_chunk(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        s_decoding_chunk_size: int = 0,
        s_num_decoding_left_chunks: int = -1,
        b_decoding_chunk_size: int = 0,
        b_num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor. two_chunk

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
        T = xs.size(1)
        B = xs.size(0)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks  # (B, 1, T/subsamples_rate)
        s_chunks_masks = add_optional_chunk_mask(xs, masks,
                                                 self.use_dynamic_chunk,
                                                 self.use_dynamic_left_chunk,
                                                 s_decoding_chunk_size,
                                                 self.s_static_chunk_size,
                                                 s_num_decoding_left_chunks)
        b_chunks_masks = add_optional_chunk_mask(xs, masks,
                                                 self.use_dynamic_chunk,
                                                 self.use_dynamic_left_chunk,
                                                 b_decoding_chunk_size,
                                                 self.b_static_chunk_size,
                                                 b_num_decoding_left_chunks)
        s_xs = xs
        b_xs = xs
        # add add l_encoder_layer
        for layer in self.encoders[:self.s_ctc_layer - self.add_l_num_blocks]:
            s_xs, s_chunks_masks, _, _ = layer(s_xs, s_chunks_masks,
                                               pos_emb, mask_pad)
        # add s_b layer
        lh_xs = s_xs
        for layer in self.encoders[self.s_ctc_layer - self.add_l_num_blocks:]:
            lh_xs, b_chunks_masks, _, _ = layer(lh_xs, b_chunks_masks,
                                                pos_emb, mask_pad)
        for layer in self.encoders:
            b_xs, b_chunks_masks, _, _, = layer(b_xs, b_chunks_masks,
                                                pos_emb, mask_pad)
        if self.add_l_num_blocks != 0:
            for layer in self.l_encoders:
                s_xs, s_chunks_masks, _, _ = layer(s_xs, s_chunks_masks, pos_emb, mask_pad)

        if self.normalize_before:
            s_xs = self.after_norm(s_xs)
            b_xs = self.after_norm(b_xs)
            lh_xs = self.after_norm(lh_xs)

        return s_xs, b_xs, lh_xs, mask_pad


    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
        assert xs.size(0) == 1
        # tmp_masks is just for interface compatibility
        tmp_masks = torch.ones(1,
                               xs.size(1),
                               device=xs.device,
                               dtype=torch.bool)
        tmp_masks = tmp_masks.unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(
            offset=offset - cache_t1, size=attention_key_size)
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)
        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            # NOTE(xcsong): Before layer.forward
            #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
            #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs, att_mask, pos_emb,
                att_cache=att_cache[i:i + 1] if elayers > 0 else att_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache
            )
            # NOTE(xcsong): After layer.forward
            #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
            #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
        if self.normalize_before:
            xs = self.after_norm(xs)

        # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
        #   ? may be larger than cache_t1, it depends on required_cache_size
        r_att_cache = torch.cat(r_att_cache, dim=0)
        # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

        return (xs, r_att_cache, r_cnn_cache)

    def forward_chunk_j3(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_mask: torch.Tensor = torch.ones((0, 0, 0, 0), dtype=torch.float),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fake API, for passing JIT check.
        """
        return xs, xs, xs

    def forward_chunk_by_chunk(
        self,
        xs: torch.Tensor,
        decoding_chunk_size: int,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
        assert decoding_chunk_size > 0
        # The model is trained by static or dynamic chunk
        assert self.static_chunk_size > 0 or self.use_dynamic_chunk
        subsampling = self.embed.subsampling_rate
        context = self.embed.right_context + 1  # Add current frame
        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        num_frames = xs.size(1)
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
        outputs = []
        offset = 0
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        # Feed forward overlap input step by step
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = xs[:, cur:end, :]
            (y, att_cache, cnn_cache) = self.forward_chunk(
                chunk_xs, offset, required_cache_size, att_cache, cnn_cache)
            outputs.append(y)
            offset += y.size(1)
        ys = torch.cat(outputs, 1)
        masks = torch.ones((1, 1, ys.size(1)), device=ys.device, dtype=torch.bool)
        return ys, masks


    # def forward_chunk_by_chunk_fastu2pp(
    #     self,
    #     xs: torch.Tensor,
    #     s_decoding_chunk_size: int,
    #     b_decoding_chunk_size: int,
    #     s_num_decoding_left_chunks: int = -1,
    #     b_num_decoding_left_chunks: int = -1,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """fastu2pp: latency banchmark"""
    #     assert s_decoding_chunk_size > 0 and b_decoding_chunk_size > 0
    #     assert self.static_chunk_size > 0 or self.use_dynamic_chunk
    #     subsampling = self.embed.subsampling_rate
    #     context = self.embed.right_context + 1
    #     s_stride = subsampling * s_decoding_chunk_size
    #     b_stride = subsampling * b_decoding_chunk_size
    #     s_decoding_window = (s_decoding_chunk_size - 1) * subsampling + context
    #     num_frames = xs.size(1)  # 输入数据的所有帧数
    #     att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
    #     cnn_cache: torch.Tensor = torch.zeors((0, 0, 0, 0), device=xs.device)
    #     outputs = []
    #     offset = 0
    #     required_cache_size = s_decoding_chunk_size * s_num_decoding_left_chunks
        
        




class TransformerEncoder(BaseEncoder):
    """Transformer encoder module."""
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
    ):
        """ Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        assert check_argument_types()
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         concat_after, static_chunk_size, use_dynamic_chunk,
                         global_cmvn, use_dynamic_left_chunk)
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                MultiHeadedAttention(attention_heads, output_size,
                                     attention_dropout_rate),
                PositionwiseFeedForward(output_size, linear_units,
                                        dropout_rate), dropout_rate,
                normalize_before, concat_after) for _ in range(num_blocks)
        ])


class ConformerEncoder(BaseEncoder):
    """Conformer encoder module."""
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        add_l_num_blocks : int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        concat_after: bool = False,
        s_static_chunk_size: int = 0,
        b_static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        s_ctc_layer: int = 7,
        two_chunk: bool = False,
        dual: bool = False,
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
        """
        assert check_argument_types()
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, add_l_num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         concat_after, s_static_chunk_size, b_static_chunk_size,
                         use_dynamic_chunk,
                         global_cmvn, use_dynamic_left_chunk, s_ctc_layer, two_chunk)
        activation = get_activation(activation_type)

        # self-attention module definition
        if pos_enc_layer_type == "no_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
        )
        # feed-forward module definition
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        if dual:
            print('use dual convolution', dual)
            convolution_layer = DualConvolutionModule
            convolution_layer_args = (output_size, cnn_module_kernel,
                                      activation, cnn_module_norm)
        else:
            convolution_layer = ConvolutionModule
            convolution_layer_args = (output_size, cnn_module_kernel,
                                      activation, cnn_module_norm, causal)

        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(
                    *positionwise_layer_args) if macaron_style else None,
                convolution_layer(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(num_blocks)
        ])
        # NOTE: add two layers for l_ctc (finetune)
        self.add_l_num_blocks = add_l_num_blocks
        if add_l_num_blocks != 0:
            self.l_encoders = torch.nn.ModuleList([
                ConformerEncoderLayer(
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(
                        *positionwise_layer_args) if macaron_style else None,
                    convolution_layer(
                        *convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ) for _ in range(add_l_num_blocks)
            ])
        else:
            self.l_encoders = torch.nn.Identity()

    def set_dual_mode(self, streaming: bool = False):
        self.dual = True
        self.streaming = streaming
        for layer in self.encoders:
            layer.conv_module.set_dual_mode(streaming)
        if self.add_l_num_blocks != 0.0:
            for layer in self.l_encoders:
                layer.conv_module.set_dual_mode(streaming)

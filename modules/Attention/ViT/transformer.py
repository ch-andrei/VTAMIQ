# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage

from torch.nn import Dropout, Softmax, Linear, LayerNorm
from collections import OrderedDict


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

VIT_VARIANT_B16 = "ViT-B16"
VIT_VARIANT_L16 = "ViT-L16"

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


def get_b16_config():
    vit_b16_config = OrderedDict(
        vit_weights_path="./modules/Attention/ViT/weights/imagenet21k+imagenet2012_ViT-B_16.npz",
        patch_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        batch_norm=True,
        num_scales=5,
    )
    return vit_b16_config


def get_l16_config():
    vit_l16_config = OrderedDict(
        vit_weights_path="./modules/Attention/ViT/weights/imagenet21k+imagenet2012_ViT-L_16.npz",
        patch_size=16,
        hidden_size=1024,
        mlp_dim=4096,
        num_heads=16,
        num_layers=24,
        attention_dropout_rate=0.0,
        dropout_rate=0.1,
        batch_norm=True,
        num_scales=5,
    )
    return vit_l16_config


def get_vit_config(variant):
    if variant == VIT_VARIANT_B16:
        vit_config = get_b16_config()
    elif variant == VIT_VARIANT_L16:
        vit_config = get_l16_config()
    else:
        raise ValueError("ViT: Unsupported variant [{}], pick from [{}, {}].".format(
            variant, VIT_VARIANT_B16, VIT_VARIANT_L16
        ))
    return vit_config


def pjoin(*args):
    return "/".join(args)


def np2th(weights):
    """Possibly convert HWIO to OIHW."""
    if weights.ndim == 4:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()

        num_heads = config["num_heads"]
        hidden_size = config["hidden_size"]
        attention_dropout_rate = config["attention_dropout_rate"]

        self.vis = vis
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = num_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()

        hidden_size = config["hidden_size"]
        mlp_dim = config["mlp_dim"]
        dropout_rate = config["dropout_rate"]

        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Layer(nn.Module):
    def __init__(self, config, vis):
        super(Layer, self).__init__()

        hidden_size = config["hidden_size"]

        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self,
                 config,
                 num_keep_layers,
                 vis,  # boolean toggle for outputting visual attention
                 ):
        super(Encoder, self).__init__()

        hidden_size = config["hidden_size"]

        if num_keep_layers <= 0:
            num_layers = config["num_layers"]
        else:
            num_layers = max(1, min(num_keep_layers, config["num_layers"]))

        self.vis = vis
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        self.layer = nn.ModuleList()
        for _ in range(num_layers):
            layer = Layer(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class ScaleEmbedding(nn.Module):
    def __init__(self, config):
        super(ScaleEmbedding, self).__init__()

        hidden_size = config["hidden_size"]
        num_scales = config["num_scales"]

        self.num_scales = num_scales  # 1 for cls token
        self.scale_embeddings = nn.Parameter(torch.zeros(1, self.num_scales + 1, hidden_size))

    def forward(self, scale):
        scale = torch.clamp(scale[:, 0], 0, self.num_scales - 1) + 1  # offset for cls token
        scale = scale.to(dtype=torch.long)
        scale = self.scale_embeddings[:, scale]
        return scale

    def forward_cls_token(self):
        return self.scale_embeddings[:, 0]  # first embedding is cls token


class UvPosEmbedding(nn.Module):
    def __init__(self, config, img_dim=384):  # 384
        super(UvPosEmbedding, self).__init__()

        hidden_size = config["hidden_size"]
        patch_size = config["patch_size"]

        self.width_pos_embeddings = img_dim // patch_size
        num_pos_embeddings = self.width_pos_embeddings ** 2 + 1  # 1 for cls token embedding
        self.positional_embeddings = nn.Parameter(torch.zeros(1, num_pos_embeddings, hidden_size))

    def forward(self, pos):
        # print('pos', float(pos.min()), float(pos.max()))
        pos = torch.floor(pos * self.width_pos_embeddings)
        pos = (pos[:, 0] * self.width_pos_embeddings + pos[:, 1]) + 1  # +1 offset to step over cls token embedding
        pos = pos.to(dtype=torch.long)
        pos = self.positional_embeddings[:, pos]
        return pos

    def forward_cls_token(self):
        return self.positional_embeddings[:, 0]

    def load_from(self, weights):
        with torch.no_grad():
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_base = self.positional_embeddings

            # check if resizing is necessary
            if posemb.size() != posemb_base.size():
                print("Need to resize positional embeddings")

                # apply rescaling

                ntok_new = posemb_base.size(1)
                ntok_new -= 1

                posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                posemb = np2th(posemb)

            self.positional_embeddings.copy_(posemb)


class Embeddings(nn.Module):
    def __init__(self,
                 config,
                 use_cls_token=True,
                 dropout=0.2,
                 img_dim=384,
                 use_patch_embedding=True,
                 use_pos_embedding=True,
                 use_scale_embedding=False,
                 ):
        super(Embeddings, self).__init__()

        hidden_size = config["hidden_size"]
        patch_size = config["patch_size"]

        # "RGB" tensor to hidden_size dimension
        self.use_patch_embedding = use_patch_embedding
        if use_patch_embedding:
            self.patch_embeddings = nn.Conv2d(in_channels=3,
                                              out_channels=hidden_size,
                                              kernel_size=patch_size,
                                              stride=patch_size)

        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)

        self.use_pos_embedding = use_pos_embedding
        if use_pos_embedding:
            self.positional_embeddings = UvPosEmbedding(config, img_dim)

        self.use_scale_embedding = use_scale_embedding
        if use_scale_embedding:
            self.scale_embeddings = ScaleEmbedding(config)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos, scales=None):
        if len(x.shape) == 5:
            B, N, C, P, P = x.shape

            # compute patch embeddings
            x = x.view(B * N, C, P, P)
            x = self.patch_embeddings(x)
        else:  # B, N, H
            B, N, _ = x.shape

        x = x.view(B, N, -1)  # BxNxH

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)

        # add positional embeddings
        if self.use_pos_embedding:
            pos = pos.view(B * N, 2)
            pos = self.positional_embeddings(pos)
            pos = pos.view(B, N, -1)
            x = x + pos  # BxNxH
            if self.use_cls_token:
                cls_tokens = cls_tokens + self.positional_embeddings.forward_cls_token()

        # add scale embeddings
        if self.use_scale_embedding and scales is not None:
            scales = scales.reshape(B * N, 2)
            scales = self.scale_embeddings(scales)
            scales = scales.view(B, N, -1)
            x = x + scales
            if self.use_cls_token:
                cls_tokens = cls_tokens + self.scale_embeddings.forward_cls_token()

        if self.use_cls_token:
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.dropout(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 config,
                 num_keep_layers=-1,
                 use_patch_embedding=True,
                 use_pos_embedding=True,
                 use_scale_embedding=False,
                 vis=False,
                 use_cls_token=True,
                 pretrained=False,
                 ):
        super(VisionTransformer, self).__init__()
        self.embeddings = Embeddings(
            config,
            use_patch_embedding=use_patch_embedding,
            use_pos_embedding=use_pos_embedding,
            use_scale_embedding=use_scale_embedding,
            use_cls_token=use_cls_token,
        )
        self.encoder = Encoder(config, num_keep_layers, vis)
        self.use_cls_token = use_cls_token
        if pretrained:
            vit_weights_path = config["vit_weights_path"]
            print("ViT: Loading pretrained transformer from path:", vit_weights_path)
            self.load_from(np.load(vit_weights_path), use_patch_embedding, use_pos_embedding, use_cls_token)

    def forward(self, x, pos, scales):
        x = self.embeddings(x, pos, scales)  # this will transform BxNxCxPxP into B x N+1 x H
        x, attn_weights = self.encoder(x)
        x = x[:, 0] if self.use_cls_token else x  # cls_token only (B x 1 x H) or full B x N+1 x H
        x = (x, attn_weights) if self.encoder.vis else x
        return x

    def load_from(self, weights, use_patch_embedding, use_pos_embedding, use_cls_token):
        with torch.no_grad():
            if use_patch_embedding:
                self.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"]))
                self.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            if use_cls_token:
                self.embeddings.cls_token.copy_(np2th(weights["cls"]))

            if use_pos_embedding:
                self.embeddings.positional_embeddings.load_from(weights)

            self.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            for bname, block in self.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

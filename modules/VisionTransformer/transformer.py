# Copyright 2024 Andrei Chubarau
# This file contains a mix of custom modifications and various publicly available implementations for ViT
# Originally based on https://github.com/jeonsworld/ViT-pytorch

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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import ndimage
from collections import OrderedDict

from torch.nn import Dropout, Softmax, Linear, LayerNorm
from timm.models.layers import DropPath, trunc_normal_

from modules.utils import init_weights_linear

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

VIT_VARIANT_B8 = "ViT-B8"
VIT_VARIANT_B16 = "ViT-B16"
VIT_VARIANT_L16 = "ViT-L16"

ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "relu": torch.nn.functional.relu
}

# using params from DeiT: https://github.com/facebookresearch/deit/blob/main/models_v2.py#L171
DROPOUT_MLP = 0.0
DROPOUT_ATTN = 0.0
DROPOUT_PROJ = 0.0
DROPOUT_EMBEDDINGS = 0.0

INIT_NORM_STD = 0.02


def get_B16_config():
    vit_b16_config = OrderedDict(
        vit_weights_path="./modules/VisionTransformer/weights/imagenet21k+imagenet2012_ViT-B_16.npz",
        img_dim=384,
        patch_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        num_layers=12,
    )
    return vit_b16_config


def get_B8_config():
    vit_b8_config = get_B16_config()
    vit_b8_config["patch_size"] = 8
    vit_b8_config["vit_weights_path"] = "./modules/VisionTransformer/weights/imagenet21k+imagenet2012_ViT-B_8.npz"
    return vit_b8_config


def get_L16_config():
    vit_l16_config = OrderedDict(
        vit_weights_path="./modules/VisionTransformer/weights/imagenet21k+imagenet2012_ViT-L_16.npz",
        img_dim=384,
        patch_size=16,
        hidden_size=1024,
        mlp_dim=4096,
        num_heads=16,
        num_layers=24,
    )
    return vit_l16_config


def get_vit_config(variant):
    if variant == VIT_VARIANT_B8:
        return get_B8_config()
    elif variant == VIT_VARIANT_B16:
        return get_B16_config()
    elif variant == VIT_VARIANT_L16:
        return get_L16_config()
    else:
        raise ValueError("ViT: Unsupported variant [{}], pick from {}.".format(
            variant, [VIT_VARIANT_B8, VIT_VARIANT_B16, VIT_VARIANT_L16]
        ))


def pjoin(*args):
    return "/".join(args)


def np2th(weights):
    """Possibly convert HWIO to OIHW."""
    if weights.ndim == 4:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config, num_tokens):
        super(MultiHeadSelfAttention, self).__init__()

        num_heads = config["num_heads"]
        hidden_size = config["hidden_size"]

        self.num_tokens = num_tokens

        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = num_heads * self.attention_head_size

        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, hidden_size)
        self.attn_dropout = Dropout(DROPOUT_ATTN)
        self.proj_dropout = Dropout(DROPOUT_PROJ)

        self.softmax = Softmax(dim=-1)

    def transpose_for_attn(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, return_attention=True):
        q = self.transpose_for_attn(self.query(x))
        k = self.transpose_for_attn(self.key(x))
        v = self.transpose_for_attn(self.value(x))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)

        attn_prob = self.softmax(attn_scores)  # attention probability
        weights = attn_prob if return_attention else None  # save to return later
        attn_prob = self.attn_dropout(attn_prob)

        x = torch.matmul(attn_prob, v)

        x = x.permute(0, 2, 1, 3).contiguous()
        out_shape = x.size()[:-2] + (self.all_head_size,)
        x = x.view(*out_shape)
        x = self.out(x)
        x = self.proj_dropout(x)

        return x, weights


# Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", 2019
# from the original implementation in tensorflow https://github.com/google-research/adapter-bert
class Adapter(nn.Module):
    def __init__(self, channels, reduction=4):
        super(Adapter, self).__init__()

        hidden_size = channels // reduction

        self.adapter = nn.Sequential(
            Linear(channels, hidden_size),
            nn.GELU(),
            Linear(hidden_size, channels),
        )

        for layer in self.adapter:
            if isinstance(layer, Linear):
                init_weights_linear(layer)

    def forward(self, x):
        return x + self.adapter(x)


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        hidden_size = config["hidden_size"]
        mlp_dim = config["mlp_dim"]

        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(DROPOUT_MLP)

        init_weights_linear(self.fc1)
        init_weights_linear(self.fc2)

    def forward(self, x):
        x = self.dropout(self.act_fn(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x


# from https://github.com/meta-llama/llama3/blob/main/llama/model.py
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# LayerScale as described in Touvron et al., "Going Deeper with Image Transformers", 2021
# implementation from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1.0):
        # Note: paper recommends init_values=1e-4, but they train from scratch
        # we use pre-trained transformer, which is equivalent of init_values=1.0, and then fine-tune
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class EncoderLayer(nn.Module):
    def __init__(self, config, use_layer_scale, num_adapters, num_tokens, path_drop_prob):
        super(EncoderLayer, self).__init__()

        hidden_size = config["hidden_size"]

        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = MLP(config)
        self.attn = MultiHeadSelfAttention(config, num_tokens)

        self.use_adapters = 0 < num_adapters
        if self.use_adapters:
            self.adapters = []
            for i in range(num_adapters):
                adapter1 = Adapter(hidden_size)
                adapter2 = Adapter(hidden_size)
                ADAPTER_ATTR_NAME = "adapter"
                self.add_module(f"{ADAPTER_ATTR_NAME}{2 * i + 1}", adapter1)
                self.add_module(f"{ADAPTER_ATTR_NAME}{2 * i + 2}", adapter2)
                self.adapters.append((adapter1, adapter2))

        # DeiT layer scale and path drop
        self.ls1 = LayerScale(hidden_size) if use_layer_scale else nn.Identity()
        self.ls2 = LayerScale(hidden_size) if use_layer_scale else nn.Identity()
        self.drop1 = DropPath() if 0 < path_drop_prob else nn.Identity()
        self.drop2 = DropPath() if 0 < path_drop_prob else nn.Identity()

    def forward(self, x, adapter_num=-1, return_attention=True):
        h, weights = self.attn.forward(self.attention_norm(x), return_attention=return_attention)
        if self.use_adapters and 0 <= adapter_num:
            h = self.adapters[adapter_num][0](h)
        x = x + self.drop1(self.ls1(h))

        h = self.ffn(self.ffn_norm(x))
        if self.use_adapters and 0 <= adapter_num:
            h = self.adapters[adapter_num][1](h)
        x = x + self.drop2(self.ls2(h))
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
                 num_adapters,
                 num_tokens,
                 use_layer_scale,
                 path_drop_prob,
                 return_layers,
                 return_attention,
                 ):
        super(Encoder, self).__init__()

        hidden_size = config["hidden_size"]

        num_layers = config["num_layers"]  # default value, all available layers
        if 0 < num_keep_layers:
            num_layers = max(1, min(num_keep_layers, config["num_layers"]))

        self.return_layers = return_layers
        self.return_attention = return_attention

        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for _ in range(num_layers):
            layer = EncoderLayer(
                config=config,
                use_layer_scale=use_layer_scale,
                num_adapters=num_adapters,
                num_tokens=num_tokens,
                path_drop_prob=path_drop_prob,
            )
            self.layers.append(layer)

    def forward(self, x, adapter_num=-1, return_attention=True, return_layers=True):
        attn_weights = []
        layer_states = []

        for layer in self.layers:  # type: EncoderLayer
            x, weights = layer.forward(x, adapter_num=adapter_num)
            if self.return_attention and return_attention:
                attn_weights.append(weights)
            if self.return_layers and return_layers:
                layer_states.append(x)

        # NOTE: if not using all layers in ViT, disable use of self.encoder_norm?
        # If ViT is pretrained, this is intended to be applied after the final layer (full ViT)...
        encoded = self.encoder_norm(x)

        return encoded, attn_weights, layer_states


def init_embedding_param(layer: torch.Tensor):
    layer.data.normal_(mean=0.0, std=INIT_NORM_STD)


class ScaleEmbedding(nn.Module):
    def __init__(self, num_scales, config):
        super(ScaleEmbedding, self).__init__()

        hidden_size = config["hidden_size"]

        self.num_scales = num_scales
        self.scale_embeddings = nn.Parameter(torch.zeros(1, self.num_scales + 1, hidden_size))  # +1 for cls token

        init_embedding_param(self.scale_embeddings)

    def forward(self, scale):
        scale = torch.clamp(scale, 0, self.num_scales - 1) + 1  # +1 offset for cls token
        scale = scale.to(dtype=torch.long)
        scale = self.scale_embeddings[:, scale]
        return scale


class UvPosEmbedding(nn.Module):
    def __init__(self, config):
        super(UvPosEmbedding, self).__init__()

        hidden_size = config["hidden_size"]
        patch_size = config["patch_size"]
        img_dim = config["img_dim"]

        self.width_pos_embeddings = img_dim // patch_size
        num_pos_embeddings = self.width_pos_embeddings ** 2 + 1  # 1 for cls token embedding
        self.positional_embeddings = nn.Parameter(torch.zeros(1, num_pos_embeddings, hidden_size))

        init_embedding_param(self.positional_embeddings)

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
                 use_cls_token,
                 use_patch_embedding,
                 use_pos_embedding,
                 num_extra_tokens,
                 num_scales,
                 ):
        super(Embeddings, self).__init__()

        hidden_size = config["hidden_size"]
        patch_size = config["patch_size"]

        # "RGB" tensor to hidden_size dimension
        self.use_patch_embedding = use_patch_embedding
        if use_patch_embedding:
            self.patch_embeddings = nn.Conv2d(
                in_channels=3,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=patch_size
            )

        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)
        init_embedding_param(self.cls_token)

        # additional tokens
        self.num_extra_tokens = num_extra_tokens
        self.use_extra_tokens = 0 < num_extra_tokens
        if self.use_extra_tokens:
            self.extra_tokens = nn.Parameter(torch.zeros(1, num_extra_tokens, hidden_size), requires_grad=True)
            init_embedding_param(self.extra_tokens)

        self.use_tokens = self.use_cls_token or self.use_extra_tokens
        self.num_tokens = use_cls_token + num_extra_tokens

        self.use_pos_embedding = use_pos_embedding
        if use_pos_embedding:
            self.positional_embeddings = UvPosEmbedding(config)

        self.use_scale_embedding = 1 < num_scales
        if self.use_scale_embedding:
            self.scale_embeddings = ScaleEmbedding(num_scales, config)

        self.dropout = nn.Dropout(DROPOUT_EMBEDDINGS)

    def forward_tokens(self, B):
        tokens = []
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, 1, -1)
            # NOTE: the original implementation injects positional embedding into CLS as well
            # https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
            # this seems weird but we keep this since we use the provided pretrained weights
            if self.use_pos_embedding:
                cls_tokens = cls_tokens + self.positional_embeddings.forward_cls_token()
            # NOTE: we do not inject scale embeddings for CLS
            # if self.use_scale_embedding:
            #     cls_tokens = cls_tokens + self.scale_embeddings.forward_cls_token()
            tokens.append(cls_tokens)
        if self.use_extra_tokens:
            # NOTE: we do not inject pos and scale embeddings for extra tokens
            extra_tokens = self.extra_tokens.expand(B, self.num_extra_tokens, -1)
            tokens.append(extra_tokens)
        return torch.cat(tokens, dim=1)  # format: [CLS, T1, ..., Tn] for K extra tokens

    def forward(self, x, pos, scales=None):
        if len(x.shape) == 5:
            B, N, C, P, P = x.shape

            # compute patch embeddings
            x = x.view(B * N, C, P, P)
            x = self.patch_embeddings(x)
        else:  # B, N, H
            B, N = x.shape[:2]

        x = x.view(B, N, -1)

        # add positional embeddings
        if self.use_pos_embedding:
            pos = pos.view(B * N, 2)
            pos = self.positional_embeddings(pos)
            pos = pos.view(B, N, -1)
            x = x + pos

        # add scale embeddings
        if self.use_scale_embedding:
            if scales is None:
                raise ValueError("Model uses scale embedding but scales is passed as None.")

            scales = scales.reshape(B * N)
            scales = self.scale_embeddings(scales)
            scales = scales.view(B, N, -1)
            x = x + scales

        # prepend cls and extra tokens
        if self.use_tokens:
            tokens = self.forward_tokens(B)
            x = torch.cat((tokens, x), dim=1)

        x = self.dropout(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 config,
                 use_patch_embedding=True,
                 use_pos_embedding=True,
                 use_cls_token=True,
                 use_classifier=False,
                 use_layer_scale=False,
                 num_keep_layers=-1,
                 num_extra_tokens=0,
                 num_classes=1000,
                 num_adapters=0,
                 num_scales=0,
                 path_drop_prob=0.,
                 pretrained=True,
                 return_layers=False,
                 return_attention=False,
                 ):
        super(VisionTransformer, self).__init__()

        self.hidden_size = config["hidden_size"]
        self.use_cls_token = use_cls_token
        self.num_extra_tokens = num_extra_tokens
        self.use_extra_tokens = 0 < num_extra_tokens
        self.use_classifier = use_classifier and self.use_tokens
        self.use_layer_scale = use_layer_scale
        self.use_adapters = 0 < num_adapters

        self.embeddings = Embeddings(
            config,
            use_cls_token=use_cls_token,
            use_patch_embedding=use_patch_embedding,
            use_pos_embedding=use_pos_embedding,
            num_extra_tokens=num_extra_tokens,
            num_scales=num_scales,
        )

        self.use_tokens = self.use_cls_token or self.use_extra_tokens
        self.num_tokens = self.embeddings.num_tokens

        self.encoder = Encoder(
            config=config,
            num_keep_layers=num_keep_layers,
            num_adapters=num_adapters,
            num_tokens=self.num_tokens,
            use_layer_scale=use_layer_scale,
            path_drop_prob=path_drop_prob,
            return_layers=return_layers,
            return_attention=return_attention
        )

        if self.use_classifier:
            if not self.use_tokens or 1 < self.num_tokens:
                raise ValueError("VisionTransformer with Classifier requires CLS token.")
            self.classifier = Linear(self.hidden_size, num_classes)

        if pretrained:
            vit_weights_path = config["vit_weights_path"]
            print("ViT: Loading pretrained transformer from path:", vit_weights_path)
            self.load_from(np.load(vit_weights_path), use_patch_embedding, use_pos_embedding)
        else:
            self.apply(self._init_weights)

    def forward(self, x, pos, scales, tokens_only=True, adapter_num=-1):
        x = self.embeddings(x, pos, scales)  # this will transform BxNxCxPxP into (B, num_tokens+N, H)
        x, attn_weights, hidden_states = self.encoder.forward(x, adapter_num)
        # select tokens (B, num_tokens, H) or tokens+patches (B, num_tokens+N, H)
        if self.use_tokens:
            if tokens_only or self.use_classifier:
                x = x[:, :self.num_tokens]  # remove patches for the final layer output
                for i in range(len(hidden_states)):  # remove patches for layer-wise activations (hidden states)
                    hidden_states[i] = hidden_states[i][:, :self.num_tokens]
                if self.use_classifier:
                    # TODO verify if classifier actually works (very low priority; not needed for IQA)
                    x = x[:, 0]  # select the first CLS token only
                    x = self.classifier(x)  # predict class label
        return x, attn_weights, hidden_states

    def load_from(self, weights, use_patch_embedding, use_pos_embedding):
        with torch.no_grad():
            if use_patch_embedding:
                self.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"]))
                self.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            if self.use_tokens:
                cls_token_data = np2th(weights["cls"])  # original pretrained CLS token data
                if self.use_cls_token:
                    self.embeddings.cls_token.copy_(cls_token_data)
                # if self.use_extra_tokens:
                #     self.embeddings.extra_tokens.copy_(cls_token_data.expand(1, self.num_extra_tokens, -1))

            if use_pos_embedding:
                self.embeddings.positional_embeddings.load_from(weights)

            self.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            for bname, block in self.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.use_classifier:
                self.classifier.weight.copy_(np2th(weights["head/kernel"]).t())
                self.classifier.bias.copy_(np2th(weights["head/bias"]).t())

    # from DeiT: https://github.com/facebookresearch/deit/blob/main/models_v2.py#L171
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=INIT_NORM_STD)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

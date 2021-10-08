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
    )
    return vit_l16_config


def get_vit_config(variant):
    _b16 = "B16"
    _l16 = "L16"
    if variant == _b16:
        vit_config = get_b16_config()
    elif variant == _l16:
        vit_config = get_l16_config()
    else:
        raise ValueError("ViT: Unsupported variant [{}], pick from [{}, {}].".format(
            variant, _b16, _l16
        ))
    return vit_config


# logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def pjoin(*args):
    return "/".join(args)


def np2th(weights):
    """Possibly convert HWIO to OIHW."""
    if weights.ndim == 4:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


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


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()

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
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

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
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class UvPosEmbedding(nn.Module):
    def __init__(self, config, img_dim=384):  # 384
        super(UvPosEmbedding, self).__init__()

        hidden_size = config["hidden_size"]
        patch_size = config["patch_size"]

        self.width_pos_embeddings = img_dim // patch_size
        self.num_pos_embeddings = self.width_pos_embeddings ** 2 + 1
        self.positional_embeddings = nn.Parameter(torch.zeros(1, self.num_pos_embeddings, hidden_size))

    def forward(self, pos):
        pos = (pos + 1.) / (2. + 1e-6)  # rescale to [0-1[, 1 exclusive
        pos = torch.floor(pos * self.width_pos_embeddings)
        pos = (pos[:, 0] * self.width_pos_embeddings + pos[:, 1])
        pos = pos.to(dtype=torch.long)
        pos = self.positional_embeddings[:, pos]
        return pos

    def load_from(self, weights):
        with torch.no_grad():
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_base = self.positional_embeddings

            # check if resizing is necessary
            if posemb.size() != posemb_base.size():
                print("Need to resize positional embeddings")

                # apply rescaling

                ntok_new = posemb_base.size(1)

                posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                ntok_new -= 1

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


class GaussianFourierEmbedding(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_out,
                 gaussian_scale=0.25,
                 ):
        super(GaussianFourierEmbedding, self).__init__()

        assert channels_out % 2 == 0, \
            "GaussianFourierEmbedding channels_out must be divisible by 2."

        B_gauss = gaussian_scale * torch.randn(channels_out // 2, channels_in)
        B_gauss -= B_gauss / 2  # move to [-amp, amp] range
        pi = torch.tensor(np.pi)

        self.register_buffer("B_gauss", B_gauss)
        self.register_buffer("pi", pi)

    # Fourier feature mapping
    def forward(self, x):
        x = (2.0 * self.pi * x) @ self.B_gauss.T
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x

    def load_from(self, weights):
        print("WARNING: GaussianFourierEmbedding.load_from() was called.")


class Embeddings(nn.Module):
    def __init__(self,
                 config,
                 dropout=0.2,
                 img_dim=384,
                 use_fourier_embeddings=False,
                 ):
        super(Embeddings, self).__init__()

        hidden_size = config["hidden_size"]
        patch_size = config["patch_size"]

        # "RGB" tensor to hidden_size dimension
        self.patch_embeddings = nn.Conv2d(in_channels=3,
                                          out_channels=hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)

        if use_fourier_embeddings:
            self.positional_embeddings = GaussianFourierEmbedding(channels_in=2, channels_out=hidden_size)
        else:
            self.positional_embeddings = UvPosEmbedding(config, img_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size), requires_grad=True)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos):
        B, N, C, P, P = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)

        x = x.view(B * N, C, P, P)
        x = self.patch_embeddings(x)
        x = x.view(B, N, -1)  # BxNxH

        pos = pos.view(B * N, 2)
        pos = self.positional_embeddings(pos)
        pos = pos.view(B, N, -1)
        x = x + pos  # BxNxH

        x = torch.cat((cls_token, x), dim=1)
        x = self.dropout(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, num_keep_layers=1, use_fourier_embeddings=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.use_fourier_embeddings = use_fourier_embeddings
        self.embeddings = Embeddings(config, use_fourier_embeddings=use_fourier_embeddings)
        self.encoder = Encoder(config, num_keep_layers, vis)

    def forward(self, x, pos):
        x = self.embeddings(x, pos)  # this will reshape BxNxCxPxP into B x N+1 x H
        x, attn_weights = self.encoder(x)
        x = x[:, 0]
        return x

    def load_from(self, weights):
        with torch.no_grad():
            self.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"]))
            self.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            if not self.use_fourier_embeddings:
                self.embeddings.positional_embeddings.load_from(weights)

            for bname, block in self.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

import numpy as np
from torch import nn
import torch

from modules.Attention.ViT.transformer import VisionTransformer
from modules.RCAN.rcan_channel_attention import ResidualGroup, ResidualGroupXY


class VTAMIQ(nn.Module):
    def __init__(self,
                 vit_config,
                 vit_variant="B16",
                 vit_load_pretrained=True,
                 vit_num_keep_layers=-1,
                 vit_use_scale_embedding=False,
                 num_residual_groups=4,
                 num_rcabs_per_group=4,
                 dropout=0.25,
                 is_full_reference=False,
                 use_diff_embedding=True,
                 ):
        super(VTAMIQ, self).__init__()

        print("VTAMIQ: Using ViT-{} transformer.".format(vit_variant))

        self.use_diff_embedding = use_diff_embedding
        self.is_full_reference = is_full_reference
        self.transformer = VisionTransformer(vit_config,
                                             num_keep_layers=vit_num_keep_layers,
                                             use_scale_embedding=vit_use_scale_embedding)

        # transformer parameters
        hidden_size = vit_config["hidden_size"]
        vit_weights_path = vit_config["vit_weights_path"]

        if vit_load_pretrained:
            print("VTAMIQ: Loading pretrained transformer from path:", vit_weights_path)
            self.transformer.load_from(np.load(vit_weights_path))

        if self.use_diff_embedding:
            # uses a residual group with special multiplicative connection
            # between the reference image f_ref and the difference signal (f_ref - f_dist)
            # thus incorporating f_ref in the first step of the difference modulation stage
            print("VTAMIQ: Using difference embedding.")
            self.diff_embedding = ResidualGroupXY(hidden_size, num_rcabs_per_group, reduction=16)

        self.diff_net = nn.Sequential(
            *[ResidualGroup(hidden_size, num_rcabs_per_group, reduction=16) for _ in range(num_residual_groups)],
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1, bias=True)
        )

        self.q_predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
        )

    def set_freeze_state(self, freeze_state,
                         diffnet=True, encoder=True, embed_patch=True, embed_pos=True, embed_scale=True):
        print("VTAMIQ: Setting freeze state to", freeze_state)

        requires_grad = not freeze_state

        def set_grad(layer, requires_grad):
            for param in layer.parameters():
                param.requires_grad = requires_grad

        if diffnet:
            set_grad(self.diff_net, requires_grad)

        if encoder:
            set_grad(self.transformer.encoder, requires_grad)

        if embed_patch:
            self.transformer.embeddings.cls_token.requires_grad = requires_grad
            set_grad(self.transformer.embeddings.patch_embeddings, requires_grad)

        if embed_pos:
            set_grad(self.transformer.embeddings.positional_embeddings, requires_grad)

        if embed_scale and self.transformer.embeddings.use_scale_embedding:
            set_grad(self.transformer.embeddings.scale_embeddings, requires_grad)

    def forward_fr(self, patches, patches_pos, patches_scales):
        patches_ref, patches_dist = patches

        B, N, C, P, P = patches_ref.shape

        feats_ref = self.transformer(patches_ref, patches_pos, patches_scales)
        feats_dist = self.transformer(patches_dist, patches_pos, patches_scales)
        feats = feats_ref - feats_dist

        feats = feats.view(B, -1, 1, 1)  # B x H x 1 x 1

        if self.use_diff_embedding:
            feats_ref = feats_ref.view(B, -1, 1, 1)  # B x H x 1 x 1
            feats = self.diff_embedding(feats, feats_ref)

        feats = self.diff_net(feats)
        feats = feats.view(B, -1)  # B x H
        q = self.q_predictor(feats)
        q = q.flatten()

        return q

    def forward_nr(self, patches, patches_pos, patches_scales):
        patches_ref = patches[0]

        B, N, C, P, P = patches_ref.shape

        feats = self.transformer(patches_ref, patches_pos, patches_scales)

        feats = feats.view(B, -1, 1, 1)  # B x H x 1 x 1
        feats = self.diff_net(feats)

        feats = feats.view(B, -1)  # B x H
        q = self.q_predictor(feats)
        q = q.flatten()

        return q

    def forward(self, patches, patches_pos, patches_scales):
        if self.is_full_reference:
            return self.forward_fr(patches, patches_pos, patches_scales)
        else:
            return self.forward_nr(patches, patches_pos, patches_scales)

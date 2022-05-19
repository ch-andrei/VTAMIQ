import numpy as np
from torch import nn
import torch


from modules.Attention.ViT.transformer import VisionTransformer, get_vit_config, VIT_VARIANT_B16, VIT_VARIANT_L16
from modules.RCAN.rcan_channel_attention import ResidualGroup, ResidualGroupXY


def set_grad(layer, requires_grad):
    for param in layer.parameters():
        param.requires_grad = requires_grad


# preprocessing module for quality predictions used to remap Q from VTAMIQ for preference judgement (as in PieAPP)
class QMapper(nn.Module):
    def __init__(self, hidden=4, non_lin_act=False, sigmoid=False, batch_norm=False):
        super(QMapper, self).__init__()
        self.q_map = nn.Sequential(
            nn.Linear(1, hidden),
            nn.PReLU() if non_lin_act else nn.Sequential(),
            nn.BatchNorm1d(hidden) if batch_norm else nn.Sequential(),
            nn.Linear(hidden, 1),
            nn.Sigmoid() if sigmoid else nn.Sequential(),
        )

    def forward(self, x):
        # x = torch.stack([q1, q2], dim=1)
        x = x.view(-1, 1)
        x = self.q_map(x)
        return x.flatten()


class VisionTransformerBackbone(nn.Module):
    def __init__(self,
                 variant=VIT_VARIANT_B16,
                 load_pretrained=True,
                 num_keep_layers=-1,
                 use_patch_embedding=True,
                 use_pos_embedding=True,
                 use_scale_embedding=False,
                 use_cls_token=True,
                 return_attention=False,
                 ):

        super(VisionTransformerBackbone, self).__init__()

        vit_config = get_vit_config(variant)
        self.transformer = self.get_transformer(
            vit_config,
            num_keep_layers,
            use_patch_embedding,
            use_pos_embedding,
            use_scale_embedding,
            load_pretrained,
            use_cls_token,
            return_attention,
        )

        self.hidden_size = vit_config["hidden_size"]

    @staticmethod
    def get_transformer(config,
                        num_keep_layers,
                        use_patch_embedding,
                        use_pos_embedding,
                        use_scale_embedding,
                        load_pretrained,
                        use_cls_token,
                        return_attention=False,
                        ):
        return VisionTransformer(
            config,
            num_keep_layers=num_keep_layers,
            use_patch_embedding=use_patch_embedding,
            use_pos_embedding=use_pos_embedding,
            use_scale_embedding=use_scale_embedding,
            use_cls_token=use_cls_token,
            pretrained=load_pretrained,
            vis=return_attention
        )

    def set_freeze_state(self, freeze_state, freeze_dict):
        requires_grad = not freeze_state

        if freeze_dict["freeze_encoder"]:
            set_grad(self.transformer.encoder, requires_grad)

        if freeze_dict["freeze_embeddings_patch"]:
            self.transformer.embeddings.cls_token.requires_grad = requires_grad
            set_grad(self.transformer.embeddings.patch_embeddings, requires_grad)

        if freeze_dict["freeze_embeddings_pos"] and self.transformer.embeddings.use_pos_embedding:
            set_grad(self.transformer.embeddings.positional_embeddings, requires_grad)

        if freeze_dict["freeze_embeddings_scale"] and self.transformer.embeddings.use_scale_embedding:
            set_grad(self.transformer.embeddings.scale_embeddings, requires_grad)


class VTAMIQ(VisionTransformerBackbone):
    def __init__(self,
                 vit_variant,
                 vit_load_pretrained=True,
                 vit_num_keep_layers=-1,
                 vit_use_scale_embedding=False,
                 num_residual_groups=4,
                 num_rcabs_per_group=4,
                 dropout=0.1,
                 is_full_reference=False,
                 use_diff_embedding=False,
                 **kwargs
                 ):
        return_attention = kwargs.pop("return_attention", False)

        super(VTAMIQ, self).__init__(
            variant=vit_variant,
            load_pretrained=vit_load_pretrained,
            num_keep_layers=vit_num_keep_layers,
            use_pos_embedding=True,
            use_scale_embedding=vit_use_scale_embedding,
            use_cls_token=True,
            return_attention=return_attention,
        )

        if 0 < len(kwargs):
            print("WARNING: VTAMIQ has unused kwargs={}".format(kwargs))

        self.use_diff_embedding = use_diff_embedding
        self.is_full_reference = is_full_reference
        self.dropout = dropout

        if self.use_diff_embedding:
            # uses a residual group with special multiplicative connection
            # between the reference image f_ref and the difference signal (f_ref - f_dist)
            # thus incorporating f_ref in the first step of the difference modulation stage
            print("VTAMIQ: using difference embedding.")
            self.diff_embedding = ResidualGroupXY(self.hidden_size, num_rcabs_per_group, reduction=16)

        self.diff_net = nn.Sequential(
            *[ResidualGroup(self.hidden_size, num_rcabs_per_group, reduction=16) for _ in range(num_residual_groups)],
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, bias=True)
        )

        self.q_predictor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 4),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 4, 1),
        )

    def set_freeze_state(self,
                         freeze_state,
                         freeze_dict,
                         ):
        print("VTAMIQ: Setting freeze state to", freeze_state)

        super().set_freeze_state(freeze_state, freeze_dict)

        requires_grad = not freeze_state

        if freeze_dict["freeze_diffnet"]:
            set_grad(self.diff_net, requires_grad)

        if freeze_dict["freeze_q_predictor"]:
            set_grad(self.q_predictor, requires_grad)

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
        # q = q.flatten()

        return q

    def forward_nr(self, patches, patches_pos, patches_scales):
        patches_ref = patches[0]

        B, N, C, P, P = patches_ref.shape

        feats = self.transformer(patches_ref, patches_pos, patches_scales)

        feats = feats.view(B, -1, 1, 1)  # B x H x 1 x 1
        feats = self.diff_net(feats)

        feats = feats.view(B, -1)  # B x H
        q = self.q_predictor(feats)
        # q = q.flatten()

        return q

    def forward(self, patches, patches_pos, patches_scales):
        if self.is_full_reference:
            return self.forward_fr(patches, patches_pos, patches_scales)
        else:
            return self.forward_nr(patches, patches_pos, patches_scales)

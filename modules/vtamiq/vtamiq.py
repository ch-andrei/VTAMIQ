from torch import nn
import torch

from modules.VisionTransformer.backbone import VisionTransformerBackbone
from modules.RCAN.channel_attention import ResidualGroup
from modules.VisionTransformer.transformer import LayerScale

from modules.utils import set_grad
from utils.misc.miscelaneous import check_unused_kwargs


def get_quality_decoder(
        dim, num_rgs, num_rcabs, ca_reduction, rg_path_drop=0., use_ms_cam=False, use_local=False):
    return nn.Sequential(
        *[
            ResidualGroup(
                dim, num_rcabs, reduction=ca_reduction, path_drop_prob=rg_path_drop,
                use_bn=False, use_ms_cam=use_ms_cam, use_local=use_local, input1d=True
            )
            for _ in range(num_rgs)
        ],
        nn.Conv1d(dim, dim, kernel_size=1),
    )


class VTAMIQ(VisionTransformerBackbone):
    def __init__(
            self,
            vit_config=None,

            # calibration network params
            calibrate=True,
            diff_scale=True,
            num_rgs=4,
            num_rcabs=4,
            rg_path_drop=0.1,
            ca_reduction=8,

            # quality predictor params
            predictor_dropout=0.,

            # misc params
            return_features=False,

            **kwargs
    ):
        if vit_config is None:
            vit_config = {}
        check_unused_kwargs("VTAMIQ", **kwargs)
        vit_config.pop("use_classifier", None)  # ViT should not return class labels, only features
        super().__init__(
            use_classifier=False,  # always disable classifier
            **vit_config,
            **kwargs
        )

        self.token_num = 0  # ViT has 1 CLS token and N register tokens; which token to use (0 -> CLS, 1 -> 1st reg)

        hidden_size = self.vit_hidden_size

        self.diff_scale = LayerScale(hidden_size, init_values=1.0) if diff_scale else nn.Sequential()

        if calibrate:
            self.quality_decoder = get_quality_decoder(
                hidden_size, num_rgs, num_rcabs, ca_reduction,
                rg_path_drop=rg_path_drop)
        else:
            self.quality_decoder = nn.Sequential()

        self.predictor_dropout = predictor_dropout
        self.q_predictor = nn.Sequential(
            nn.Dropout(predictor_dropout),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.PReLU(),
            nn.Dropout(predictor_dropout),
            nn.Linear(hidden_size // 4, 1),
        )

        self.return_features = return_features

    def set_freeze_state(self, freeze_state, freeze_dict):
        print("VTAMIQ: Setting freeze state to", freeze_state)

        super().set_freeze_state(freeze_state, freeze_dict["freeze_dict_vit"])

        requires_grad = not freeze_state

        if freeze_dict["freeze_quality_decoder"]:
            set_grad(self.quality_decoder, requires_grad)

        if freeze_dict["freeze_q_predictor"]:
            set_grad(self.q_predictor, requires_grad)

    def forward(self, patches, pos, scales):
        patches_ref, patches_dist = patches
        pos_ref, pos_dist = pos
        scales_ref, scales_dist = scales

        # feats shape: B x (num_tokens + num_patches) x hidden_size
        feats_ref, _, _ = self.forward_vit(patches_ref, pos_ref, scales_ref, tokens_only=True)
        feats_dist, _, _ = self.forward_vit(patches_dist, pos_dist, scales_dist, tokens_only=True)
        B = patches_ref.shape[0]

        feats_ref = torch.permute(feats_ref, (0, 2, 1))
        feats_dist = torch.permute(feats_dist, (0, 2, 1))

        cls_ref = feats_ref[..., self.token_num]
        cls_dist = feats_dist[..., self.token_num]
        # [..., self.token_num] to select only the IQA token (can be CLS token or extra_token)

        cls_diff = self.diff_scale(cls_ref - cls_dist)  # CLS + extra tokens

        # calibrate difference vector
        cls_diff = self.quality_decoder(cls_diff.view(B, -1, 1))  # .view(B, -1, 1) reshapes to B x H x 1

        cls_diff = cls_diff.view(B, -1)  # B x H
        q = self.q_predictor(cls_diff).flatten()

        return q, None

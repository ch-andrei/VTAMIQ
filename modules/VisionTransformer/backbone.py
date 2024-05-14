from torch import nn
from modules.VisionTransformer.transformer import VisionTransformer, get_vit_config, \
    VIT_VARIANT_B8, VIT_VARIANT_B16, VIT_VARIANT_L16
from modules.utils import set_grad
from utils.misc.miscelaneous import check_unused_kwargs


class VisionTransformerBackbone(nn.Module):
    @property
    def vit_hidden_size(self):
        return self.transformer.hidden_size

    @property
    def vit_num_layers(self):
        return len(self.transformer.encoder.layers)

    def __init__(
            self,
            variant=VIT_VARIANT_B16,  # choose from [VIT_VARIANT_B8, VIT_VARIANT_B16, VIT_VARIANT_L16]
            use_patch_embedding=True,
            use_pos_embedding=True,
            use_cls_token=True,
            use_classifier=True,
            use_layer_scale=False,
            pretrained=True,
            num_keep_layers=-1,
            num_adapters=0,
            num_scales=0,
            num_extra_tokens=0,
            path_drop_prob=0.0,
            return_layers=False,
            return_attention=False,
            **kwargs
    ):
        check_unused_kwargs("VisionTransformerBackbone", **kwargs)
        super().__init__()
        self.transformer = VisionTransformer(
            config=get_vit_config(variant),
            use_patch_embedding=use_patch_embedding,
            use_pos_embedding=use_pos_embedding,
            use_cls_token=use_cls_token,
            use_classifier=use_classifier,
            use_layer_scale=use_layer_scale,
            num_keep_layers=num_keep_layers,
            num_extra_tokens=num_extra_tokens,
            num_adapters=num_adapters,
            num_scales=num_scales,
            path_drop_prob=path_drop_prob,
            pretrained=pretrained,
            return_layers=return_layers,
            return_attention=return_attention,
        )

    def forward_vit(self, patches, patches_pos, patches_scale, tokens_only=True, adapter_num=None):
        if adapter_num is None or adapter_num < 0:
            # default case: if transformer has adapters, use them
            adapter_num = 0 if self.transformer.use_adapters else -1
        x, attn_weights, hidden_states = self.transformer.forward(
            patches, patches_pos, patches_scale, tokens_only=tokens_only, adapter_num=adapter_num)
        return x, attn_weights, hidden_states

    def set_freeze_state(self, freeze_state, freeze_dict):
        requires_grad = not freeze_state
        freeze_all = freeze_dict is None  # when freeze_dict=None, allow freeze for all layers

        # encoder modules
        if freeze_all or freeze_dict["freeze_encoder"]:
            # freeze everything then unfreeze individual modules
            set_grad(self.transformer.encoder, requires_grad)

            # unfreeze encoder layer adapters, always enable
            if not freeze_dict["freeze_encoder_adapters"] and self.transformer.use_adapters:
                for layer in self.transformer.encoder.layers:
                    for adapter1, adapter2 in layer.adapters:
                        set_grad(adapter1, True)
                        set_grad(adapter2, True)

            # unfreeze encoder layer LayerScale, always enable
            if not freeze_dict["freeze_encoder_layerscale"] and self.transformer.use_layer_scale:
                for layer in self.transformer.encoder.layers:
                    set_grad(layer.ls1, True)
                    set_grad(layer.ls2, True)

        # embeddings (patch, pos, scale) modules
        if freeze_all or freeze_dict["freeze_embeddings_cls_token"]:
            try:
                self.transformer.embeddings.cls_token.requires_grad = requires_grad
            except AttributeError:
                print("VisionTransformerBackbone.set_freeze_state(): "
                      "Could not modify requires_grad for self.transformer.embeddings.cls_token")

        if freeze_all or freeze_dict["freeze_embeddings_extra_tokens"]:
            try:
                self.transformer.embeddings.extra_tokens.requires_grad = requires_grad
            except AttributeError:
                print("VisionTransformerBackbone.set_freeze_state(): "
                      "Could not modify requires_grad for self.transformer.embeddings.extra_tokens")

        if freeze_all or freeze_dict["freeze_embeddings_patch"]:
            set_grad(self.transformer.embeddings.patch_embeddings, requires_grad)

        if freeze_all or (freeze_dict["freeze_embeddings_pos"] and self.transformer.embeddings.use_pos_embedding):
            set_grad(self.transformer.embeddings.positional_embeddings, requires_grad)

        if freeze_all or (freeze_dict["freeze_embeddings_scale"] and self.transformer.embeddings.use_scale_embedding):
            set_grad(self.transformer.embeddings.scale_embeddings, requires_grad)

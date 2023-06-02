
import tensorflow as tf
import tfimm
from tfimm.models import ModelConfig, keras_serializable, register_model
from tfimm.architectures.vit import ViTConfig
from collections import OrderedDict


@keras_serializable
class ViT_mae(tfimm.architectures.vit.ViT):
    """ Vision Transformer with support for global average pooling
    """
    cfg_class = ViTConfig
    def __init__(self, cfg: ViTConfig, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        self.global_pool = True

    def forward_features(self, x, training=False, return_features=False):
        features = OrderedDict()
        batch_size = tf.shape(x)[0]

        x, grid_size = self.patch_embed(x, return_shape=True)
        cls_token = tf.repeat(self.cls_token, repeats=batch_size, axis=0)
        if not self.cfg.distilled:
            x = tf.concat((cls_token, x), axis=1)
        else:
            dist_token = tf.repeat(self.dist_token, repeats=batch_size, axis=0)
            x = tf.concat((cls_token, dist_token, x), axis=1)
        if not self.cfg.interpolate_input:
            x = x + self.pos_embed
        else:
            pos_embed = interpolate_pos_embeddings(
                self.pos_embed,
                src_grid_size=self.cfg.grid_size,
                tgt_grid_size=grid_size,
                nb_tokens=self.cfg.nb_tokens,
            )
            x = x + pos_embed
        x = self.pos_drop(x, training=training)
        features["patch_embedding"] = x

        for j, block in enumerate(self.blocks):
            x = block(x, training=training, return_features=return_features)
            if return_features:
                x, block_features = x
                features[f"block_{j}/attn"] = block_features["attn"]
            features[f"block_{j}"] = x

        features["features_all"] = x
        if self.global_pool:
            x = tf.math.reduce_mean(x[:,1:,:], axis=1) # global pool without class token
            x = self.norm(x, training=training)
        else:
            x = self.norm(x, training=training)

            if self.cfg.distilled:
                # Here we diverge from timm and return both outputs as one tensor. That way
                # all models always have one output by default
                x = x[:, :2]
            elif self.cfg.representation_size:
                x = self.pre_logits(x[:, 0])
            else:
                x = x[:, 0]
        features["features"] = x
        return (x, features) if return_features else x


@register_model
def vit_large_patch16_224_mae(**kwargs):
    """
    ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source
    https://github.com/google-research/vision_transformer.
    """
    cfg = ViTConfig(
        name="vit_large_patch16_224_mae",
        url=None,
        patch_size=16,
        embed_dim=1024,
        nb_blocks=24,
        nb_heads=16,
    )
    return ViT_mae, cfg

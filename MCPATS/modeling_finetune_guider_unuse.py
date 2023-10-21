import math
from functools import partial
from pickle import TRUE
import numpy as np
from scipy.special import comb
import random
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path,to_3tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from modeling_finetune import Patch, Embed, get_sinusoid_encoding_table, MAWS, Block

from dataset import stand

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 2, 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

class VisionTransformer_cls_unuse(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=tuple([150,180,150]), 
                 patch_size=30, 
                 in_chans=1, 
                 num_classes=2, 
                 embed_dim=512, 
                 depth=8,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 use_mean_pooling=True,
                 vis=True,
                 feature_fusion=True,
                 selected_token_num=6):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.vis = vis
        self.feature_fusion = feature_fusion
        self.selected_token_num = selected_token_num # num tokens for selection after one layer 

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches

        self.patch = Patch(img_size=img_size, patch_size=patch_size)           
        num_patches = self.patch.num_patches
        self.embed = Embed(in_dim=patch_size ** 3, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches + 1, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        if self.feature_fusion:
            self.ff_token_select = MAWS()
        else:
            pass

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, vis=self.vis,finetune=True)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.ffnorm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None                               
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        attn_weights = []
        contributions = []
        x = self.patch(x)
        x = self.embed(x)
        B, _, _ = x.shape
        tokens = [[] for i in range(B)]

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        for blk in self.blocks:
            x, weights, contribution = blk(x)
            if self.feature_fusion:
                selected_num, selected_inx = self.ff_token_select(weights,contribution)
                B_1 =  selected_inx.shape[0]
                for i in range(B_1):
                    tokens[i].extend(x[i, selected_inx[i,:self.selected_token_num]])
            if self.vis:
                attn_weights.append(weights)
                contributions.append(contribution)

        if self.feature_fusion:
            tokens = [torch.stack(token) for token in tokens]
            tokens = torch.stack(tokens).squeeze(1)
            concat = torch.cat((x[:,0].unsqueeze(1), tokens), dim=1)  # concat cls token and selected tokens
            x = self.ffnorm(concat)
        else:
            x = self.norm(x)

        if self.fc_norm is not None:
            x = x[:, 1:]
            return self.fc_norm(x.mean(1)), attn_weights
        else:
            return x[:, 0], attn_weights

    def forward(self, x):
        x, attn_weights = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def vit_base_patch30_guider_unuse(pretrained=False, **kwargs):
    model = VisionTransformer_cls_unuse(
        patch_size=30, embed_dim=512, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, vis=True,
        feature_fusion=True, selected_token_num=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch30_cls_guider_token2(pretrained=False, **kwargs):
    model = VisionTransformer_cls_unuse(
        patch_size=30, embed_dim=512, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, vis=True,
        feature_fusion=True, selected_token_num=2, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_patch30_cls_guider_token3(pretrained=False, **kwargs):
    model = VisionTransformer_cls_unuse(
        patch_size=30, embed_dim=512, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, vis=True,
        feature_fusion=True, selected_token_num=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_patch30_cls_guider_token4(pretrained=False, **kwargs):
    model = VisionTransformer_cls_unuse(
        patch_size=30, embed_dim=512, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, vis=True,
        feature_fusion=True, selected_token_num=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_patch30_cls_guider_token5(pretrained=False, **kwargs):
    model = VisionTransformer_cls_unuse(
        patch_size=30, embed_dim=512, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, vis=True,
        feature_fusion=True, selected_token_num=5, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def vit_base_patch30_cls_guider_token3_withoutff(pretrained=False, **kwargs):
    model = VisionTransformer_cls_unuse(
        patch_size=30, embed_dim=512, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, vis=True,
        feature_fusion=False, selected_token_num=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
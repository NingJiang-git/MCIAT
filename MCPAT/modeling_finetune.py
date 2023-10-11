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

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 2, 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

def innerproduct(x):
        """点积"""
        adj = torch.matmul(x, x.transpose(1,2)) / x.shape[2]
        return torch.sigmoid(adj)

## functions for image distorting
def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_rows, img_cols, img_deps = x.shape
    num_block = 10
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 block_noise_size_z))
        image_temp[noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                      noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

## END ################################################################################



class MAWS(nn.Module):
    # mutual attention weight selection
    def __init__(self):
        super(MAWS, self).__init__()

    def forward(self, x, contributions):
        length = x.size()[1]

        contributions = contributions.mean(1)
        weights = x[:,:,0,:].mean(1)
        scores = contributions*weights
        max_inx = torch.argsort(scores, dim=1,descending=True)
        return None, max_inx  

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        # self._init_weights() 

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Patch(nn.Module):
    """ Image to Patch
    """
    def __init__(self, img_size=tuple([150,180,150]), patch_size=30):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        num_patches = (img_size[2] // patch_size[2])*(img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

    def forward(self, x, **kwargs):
        B, C, X,Y,Z = x.shape
        # FIXME look at relaxing size constraints
        assert X == self.img_size[0] and Y == self.img_size[1] and Z == self.img_size[2], \
            f"Input image size ({X}*{Y}*{Z}) doesn't match model ({self.img_size[0]}*{self.img_size[1]*self.img_size[2]})."
        x = rearrange(x,'B C (a r) (b s) (c t) -> (B C) (a b c) (r s t)', a=self.patch_shape[0], b=self.patch_shape[1], c=self.patch_shape[2], 
                                                                        r=self.patch_size[0], s=self.patch_size[1], t=self.patch_size[2])
        return x

class Embed(nn.Module):
    """ Patch to Embedding
    """
    def __init__(self, in_dim=8000,embed_dim=512):
        super().__init__()
        in_dim = in_dim
        embed_dim = embed_dim
        self.proj = nn.Linear(in_dim,embed_dim)
    def forward(self, x, **kwargs):
        x = self.proj(x)        
        return x



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=tuple([150,180,150]), patch_size=30, in_chans=1, embed_dim=512):
        super().__init__()
        # img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        num_patches = (img_size[2] // patch_size[2])*(img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, X,Y,Z = x.shape
        # FIXME look at relaxing size constraints
        assert X == self.img_size[0] and Y == self.img_size[1] and Z == self.img_size[2], \
            f"Input image size ({X}*{Y}*{Z}) doesn't match model ({self.img_size[0]}*{self.img_size[1]*self.img_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 

    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, vis=True, finetune=False):
        super().__init__()
        self.vis = vis
        self.finetune = finetune
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn_scores = attn.clone()
        attn_scores = attn_scores.softmax(dim=-2)
        attn_scores = attn_scores[:,:,:,0]
        
        attn = attn.softmax(dim=-1)
        weights = attn if self.vis else None
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.finetune:
            return x, weights, attn_scores
        else:
            return x



class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, vis = True, finetune=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim, vis=vis, finetune=finetune)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # droppath or dropout
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.finetune = finetune

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x_1 = x.clone()
            if self.finetune:
                x, weights, contribution = self.attn(self.norm1(x))
            else:
                x = self.attn(self.norm1(x))
            x = x_1 + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_1 = x.clone()
            if self.finetune:
                x, weights, contribution = self.attn(self.norm1(x))
            else:
                x = self.attn(self.norm1(x))            
            x = self.gamma_1 * x
            x = x_1 + self.drop_path(x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        if self.finetune:
            return x, weights, contribution
        else:
            return x



class Last_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_classes=2,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim, vis=False, finetune=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # droppath or dropout
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.head = nn.Linear(dim, num_classes)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x_1 = x.clone()
            x = self.attn(self.norm1(x))
            x = x_1 + self.drop_path(x)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x_1 = x.clone()
            x = self.attn(self.norm1(x))            
            x = self.gamma_1 * x
            x = x_1 + self.drop_path(x)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        
        x = self.norm3(x)
        x = x[:, 0]
        x = self.head(x)
        return x

class VisionTransformer(nn.Module):
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
            return self.fc_norm(x.mean(1)), attn_weights
        else:
            return x[:, 0], attn_weights

    def forward(self, x):
        x, attn_weights = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def vit_base_patch30(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=30, embed_dim=512, depth=8, num_heads=12, mlp_ratio=4, qkv_bias=True, vis=True,
        feature_fusion=True, selected_token_num=1, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

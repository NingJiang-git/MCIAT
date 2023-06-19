import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from einops import rearrange
import nibabel as nib

from modeling_finetune import _cfg, Block, Patch, Embed, PatchEmbed, get_sinusoid_encoding_table, MAWS, innerproduct, nonlinear_transformation, local_pixel_shuffling
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_base_patch30', 
]

class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=tuple([150,180,150]), patch_size=30, in_chans=1, num_classes=0, embed_dim=512, depth=8,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False, vis = False, feature_fusion=False, selected_token_num=6,
                 nonlinear_rate=0.9, local_rate=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.vis = vis
        self.feature_fusion = feature_fusion
        self.selected_token_num = selected_token_num
        self.nonlinear_rate = nonlinear_rate
        self.local_rate = local_rate

        # self.patch_embed = PatchEmbed(    
        #                             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)           

        self.patch = Patch(img_size=img_size, patch_size=patch_size)           
        num_patches = self.patch.num_patches
        self.embed = Embed(in_dim=patch_size ** 3, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb: 
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            # self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
            self.pos_embed = get_sinusoid_encoding_table(num_patches + 1, embed_dim) # use guider token, dim_1 of self.pos_embed must be num_patches + 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if self.feature_fusion:
            self.ff_token_select = MAWS()
        else:
            pass

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, vis=self.vis, finetune=False)
            for i in range(depth)])

        self.norm =  norm_layer(embed_dim)
        self.ffnorm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() 

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02) 

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights) 


    def _init_weights(self, m): 
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight) 
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

    def forward_features(self, x, mask):
        attn_weights = []
        contributions = []

        ## branch restore ################################################################
        x_1 = x.clone()
        x_1 = x_1.cpu().numpy()
        x_1 = nonlinear_transformation(x_1, self.nonlinear_rate)
        x_1 = torch.from_numpy(x_1).float()
        x_1 =x_1.cuda()
        ## save distorted
        img = x_1[0,::].squeeze()
        img_stand = nib.load('./CAMCAN/sub-CC110033_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz')
        affine = img_stand.affine.copy()
        hdr = img_stand.header.copy()
        img = img.cpu()
        img = img.detach().numpy()
        img = nib.Nifti1Image(img,affine,hdr)
        nib.save(img,'./mtask_distort/distort.nii.gz')

        ###################################################################################        
        x  = self.patch(x)  
        B_orig, _ , C_orig = x.shape
        x = self.embed(x)

        cls_tokens = self.cls_token.expand(B_orig, -1, -1)  ########## B_orig = batch_size ############################
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        pos_1 = self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, _, C = x.shape
        # prepare for mask and reconstruction
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        # branch restore #############################################################
        x_1 = self.patch(x_1)
        x_1 = self.embed(x_1)
        mask_without_cls = mask[:,1:151].clone()
        pos_1_without_cls = pos_1[:,1:151,:].clone()
        x_1 = x_1 + pos_1_without_cls
        x_1_vis = x_1[~mask_without_cls].reshape(B, -1, C)
        # x_1_vis is the distorted vis patches #########################################

        for blk in self.blocks:
            x_vis = blk(x_vis)
            x_1_vis = blk(x_1_vis)
        x_vis = self.norm(x_vis)
        x_1_vis = self.norm(x_1_vis)
        return x_vis, x_1_vis
 

    def forward(self, x, mask):
        x, x_distorted = self.forward_features(x, mask)
        x = self.head(x)
        x_distorted = self.head(x_distorted)

        return x, x_distorted


class PretrainViTDecoder_task_recon(nn.Module): # task_recon : input(x_vis + mask token) ; output(x_mask)
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=30, num_classes=27000, embed_dim=512, depth=1,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=150,
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 1 * patch_size ** 3 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, finetune=False)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() 
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
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

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)
        x_mask = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        return x_mask

class PretrainViTDecoder_task_restore(nn.Module): # task_restore : input(x_vis_distorted) ; output(x_vis_orig_prediction)
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=30, num_classes=27000, embed_dim=512, depth=1,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=150,
                 ):
        super().__init__()
        self.num_classes = num_classes
        assert num_classes == 1 * patch_size ** 3 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, finetune=False)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity() 
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
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

    def forward(self, x):
        for blk in self.blocks:
            # x = blk(x)
            x = blk(x)
        x = self.head(self.norm(x)) # return all
        return x


class PretrainViTDecoder_task_age(nn.Module): # task_age : input(x_vis) ; output(age_prediction)
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=30, num_classes=1, embed_dim=512, depth=1,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=150,
                 ):
        super().__init__()
        self.num_classes = num_classes
        # assert num_classes == 1 * patch_size ** 3 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, finetune=False)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim , num_classes) if num_classes > 0 else nn.Identity() 

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
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
        

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x.mean(1))
        x = self.head(x) # return age
        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=tuple([150,180,150]), 
                 patch_size=30, 
                 encoder_in_chans=1, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=512, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=27000, 
                 decoder_embed_dim=512, 
                 decoder_depth=1,
                 decoder_num_heads=6, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 vis = False, 
                 feature_fusion=False, 
                 selected_token_num=12,
                 nonlinear_rate=0.9, 
                 local_rate=0.5):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            vis=vis,
            feature_fusion=feature_fusion,
            selected_token_num=selected_token_num,
            nonlinear_rate=nonlinear_rate,
            local_rate=local_rate)

        self.decoder_recon = PretrainViTDecoder_task_recon(
            patch_size=patch_size, 
            num_patches=self.encoder.patch.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)

        self.decoder_restore = PretrainViTDecoder_task_restore(
            patch_size=patch_size, 
            num_patches=self.encoder.patch.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)

        self.decoder_age = PretrainViTDecoder_task_age(
            patch_size=patch_size, 
            num_patches=self.encoder.patch.num_patches,
            num_classes=1, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)

        self.encoder_to_decoder_recon_1 = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.encoder_to_decoder_recon_2 = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.encoder_to_decoder_restore = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim)) 

        # self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch.num_patches, decoder_embed_dim)
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch.num_patches + 1, decoder_embed_dim)


        trunc_normal_(self.mask_token, std=.02) 


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):

        x_vis, x_vis_distorted = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = x_vis[:,1:37,:]

        x_vis_1_1 = self.encoder_to_decoder_recon_1(x_vis) # [B, N_vis, C_d]
        x_vis_1_2 = self.encoder_to_decoder_recon_2(x_vis)
        x_inner_1_1 = innerproduct(x_vis_1_1)
        x_inner_1_2 = innerproduct(x_vis_1_2)

        x_vis_dis = self.encoder_to_decoder_restore(x_vis_distorted)
        B, N, C = x_vis_1_1.shape
        
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach() 

        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_vis_without_cls = pos_emd_vis[:,1:37,:].clone()
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)

        x_full_1_1 = torch.cat([x_vis_1_1 + pos_emd_vis_without_cls, self.mask_token + pos_emd_mask], dim=1) 
        x_full_1_2 = torch.cat([x_vis_1_2 + pos_emd_vis_without_cls, self.mask_token + pos_emd_mask], dim=1) 
        x_vis_dis = x_vis_dis + pos_emd_vis_without_cls
        x_vis = x_vis + pos_emd_vis_without_cls
       
        x_1_1 = self.decoder_recon(x_full_1_1, pos_emd_mask.shape[1])     
        x_1_2 = self.decoder_recon(x_full_1_2, pos_emd_mask.shape[1]) 
        x_recon = x_1_1 - x_1_2

        x_vis_restore = self.decoder_restore(x_vis_dis)
        x_age = self.decoder_age(x_vis)

        return x_recon, x_inner_1_1 , x_inner_1_2, x_vis_restore, x_age



@register_model
def pretrain_base_patch30(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=tuple([150,180,150]),
        patch_size=30, 
        encoder_embed_dim=512, 
        encoder_depth=8, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_num_classes=27000,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        vis = False, 
        feature_fusion=False, 
        selected_token_num=12,
        nonlinear_rate=0.9, 
        local_rate=0.5, 
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
import math
import sys
# from tkinter.ttk import LabeledScale
from typing import Iterable
import torch
import os
import torch.nn as nn
import nibabel as nib
from torch.autograd import Variable
from torchvision.transforms.functional import affine
import utils
from einops import rearrange

import numpy as np

EPS = 1e-15


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    model_D: torch.nn.Module, optimizer_D: torch.optim.Optimizer, 
                    epoch: int, loss_scaler, loss_scaler_D, max_norm: float = 0, patch_size: int = 30, batch_size: int = 8, 
                    log_writer=None, lr_scheduler=None, start_steps=None,save=False,
                    lr_schedule_values=None, wd_schedule_values=None):

    model.train()
    model_D.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print('epoch:',epoch)
    
    loss_func = nn.MSELoss()
    loss_adversarial = nn.MSELoss()
  
    for step, data_train in enumerate(data_loader):
        it = start_steps + step 
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        data_dic = data_train           
        images, ages = Variable(data_dic['image']), Variable(data_dic['label'])
        images = images.type(torch.cuda.FloatTensor)

        ages = ages.data.cpu().numpy()
        ages = Variable(torch.from_numpy(ages)).float()
        ages = ages.cuda()
        # ages = ages.squeeze()

        num_patches = 5*6*5
        mask_ratio = 0.76
        num_mask = int(mask_ratio * num_patches) 

        bool_masked_pos = np.hstack([
            np.zeros(num_patches - num_mask),
            np.ones(num_mask),
        ])
        np.random.shuffle(bool_masked_pos)
        masked_pos_without_cls = bool_masked_pos.copy()

        bool_masked_pos = np.insert(bool_masked_pos,0,0)

        bool_masked_pos = torch.from_numpy(bool_masked_pos).float()
        masked_pos_without_cls = torch.from_numpy(masked_pos_without_cls).float()
        
        pos = masked_pos_without_cls.clone() 

        bool_masked_pos = bool_masked_pos.type(torch.cuda.FloatTensor)
        bool_masked_pos = bool_masked_pos.to(torch.bool)
        bool_masked_pos = bool_masked_pos.expand(batch_size,-1)       

        masked_pos_without_cls = masked_pos_without_cls.type(torch.cuda.FloatTensor)
        masked_pos_without_cls = masked_pos_without_cls.to(torch.bool)
        masked_pos_without_cls = masked_pos_without_cls.expand(batch_size,-1)          

        images_patch = rearrange(images, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1=patch_size, p2=patch_size,p3=patch_size)
        ## prepare to concat full images fed into discriminator 
        images_full = images.clone().half()
        images_full = rearrange(images_full, 'b c (x p1) (y p2) (z p3) -> b (x y z) (p1 p2 p3 c)', p1=patch_size, p2=patch_size,p3=patch_size)

        if save==True:
            img = images_patch[0,::]
            img = rearrange(img,'n (a b c) -> n a b c',a=30,b=30,c=30)
   
        # --------------------------------------------------------
        # ADDING 0927
        # --------------------------------------------------------
        with torch.cuda.amp.autocast():
            recons , inner_1 ,inner_2, restores, age_preds = model(images, bool_masked_pos)
            print('step',step)

            ## prepare labels for discriminator
            Tensor = torch.cuda.FloatTensor
            valid = Variable(Tensor(images.size(0), 1).fill_(1.0), requires_grad=False).float()
            fake = Variable(Tensor(images.size(0), 1).fill_(0.0), requires_grad=False).float()

            loss_inner_1 = -torch.log(inner_1 +
                              EPS).mean()

            loss_inner_2 = -torch.log(1 -
                              inner_2 +
                              EPS).mean()        
            age_preds = age_preds.squeeze()

            loss_age = loss_func(input=age_preds,target=ages)

            ## concat full images : outputs of decoder_recon and decoder_restore
            images_real = images_full.clone()
            images_real = rearrange(images_real,'B (x y z) (a b c) -> B (x a) (y b) (z c)',x=5,y=6,z=5,a=30,b=30,c=30)
            images_real = images_real.unsqueeze(dim=1).float()

            images_full[:,pos==1,:] = recons
            images_full[:,pos==0,:] = restores

            loss_recon = loss_func(input=images_full, target=images_patch)

            images_fake = images_full.clone()
            ## prepare for input of discriminator
            images_fake = rearrange(images_fake,'B (x y z) (a b c) -> B (x a) (y b) (z c)',x=5,y=6,z=5,a=30,b=30,c=30)
            images_fake = images_fake.unsqueeze(dim=1).float()

            g_adv_loss = loss_adversarial(model_D(images_fake), valid)
    
            g_loss = 0.1*loss_age + 0.79*loss_recon + 0.005*loss_inner_1 + 0.005*loss_inner_2 + 0.1*g_adv_loss


            print('loss_recon',loss_recon)
            print('loss_inner_1',loss_inner_1)
            print('loss_inner_2',loss_inner_2)
            print('loss_age',loss_age)
            print('g_adv_loss',g_adv_loss)
            print('g_loss_total',g_loss)


        loss_recon_value = loss_recon.item()
        loss_inner_1_value = loss_inner_1.item()
        loss_inner_2_value = loss_inner_2.item()
        loss_age_value = loss_age.item()
        adv_loss_value = g_adv_loss.item()
        g_loss_value = g_loss.item()


        ## save and vis
        if save==True:
            img_orig_vis = img
            img_new_vis = images_full[0,:]
            img_1 = img_orig_vis.clone()
            recon = recons[0,::]
            recon = rearrange(recon,'n (a b c) -> n a b c',a=30,b=30,c=30)
            recon = recon.type(torch.cuda.FloatTensor)
            mask = torch.zeros_like(recon)

            img_orig_vis = rearrange(img_orig_vis,'(x y z) a b c -> (x a) (y b) (z c)',x=5,y=6,z=5)
            img_orig_vis = img_orig_vis.cpu()
            img_orig_vis = img_orig_vis.detach().numpy()            

            # save origin
            img_stand = nib.load('./CAMCAN/sub-CC110033_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz')
            affine = img_stand.affine.copy()
            hdr = img_stand.header.copy()
            origin_img = nib.Nifti1Image(img_orig_vis,affine,hdr)
            nib.save(origin_img,'./mtask_orig/origin_%s.nii.gz'%epoch)

            # save mask
            img_1[pos==1,:,:,:] = mask
            img_1 = rearrange(img_1,'(x y z) a b c -> (x a) (y b) (z c)',x=5,y=6,z=5)
            img_1 = img_1.cpu()
            img_1 = img_1.detach().numpy()
            mask_img = nib.Nifti1Image(img_1,affine,hdr)
            nib.save(mask_img,'./mtask_mask/mask_%s.nii.gz'%epoch)

            # save new
            img_new_vis = rearrange(img_new_vis,'(x y z) (a b c) -> (x a) (y b) (z c)',x=5,y=6,z=5,a=30,b=30,c=30)
            img_new_vis = img_new_vis.cpu()
            img_new_vis = img_new_vis.detach().numpy()
            new_img = nib.Nifti1Image(img_new_vis,affine,hdr)
            nib.save(new_img,'./mtask_new/new_%s.nii.gz'%epoch)            


        ## generator step
        if not math.isfinite(g_loss_value): 
            print("Loss is {}, stopping training".format(g_loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

        grad_norm = loss_scaler(g_loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize() 

        metric_logger.update(loss_recon=loss_recon_value)
        metric_logger.update(loss_age=loss_age_value)
        metric_logger.update(adv_loss=adv_loss_value)
        metric_logger.update(g_loss=g_loss_value)
        metric_logger.update(loss_scale=loss_scale_value)

        ## discriminator step 
        optimizer_D.zero_grad()
        is_second_order_D = hasattr(optimizer_D, 'is_second_order') and optimizer_D.is_second_order
        
        fake_loss = loss_adversarial(model_D(images_fake.detach()),fake)       
        real_loss = loss_adversarial(model_D(images_real),valid)

        d_loss = 0.5 * (real_loss + fake_loss)
        print('d_loss',d_loss)
       
        grad_norm_D = loss_scaler_D(d_loss, optimizer_D, clip_grad=max_norm,
                                parameters=model_D.parameters(), create_graph=is_second_order_D)
        loss_scale_value_D = loss_scaler_D.state_dict()["scale"]

        d_loss_value = d_loss.item()
        metric_logger.update(d_loss=d_loss_value)
        metric_logger.update(loss_scale_D=loss_scale_value_D)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        for group in optimizer_D.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        for group in optimizer_D.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        metric_logger.update(grad_norm_D=grad_norm_D)

        if log_writer is not None:
            log_writer.update(loss_recon=loss_recon_value, head="recon-loss")
            log_writer.update(loss_age=loss_age_value, head="age-loss")
            log_writer.update(adv_loss=adv_loss_value, head="adv-loss")
            log_writer.update(g_loss=g_loss_value, head="g-loss")
            log_writer.update(d_loss=d_loss_value, head="d-loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

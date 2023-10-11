# --------------------------------------------------------
# Based on mae code bases
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/pengzhiliang/MAE-pytorch
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
from timm.models import create_model
from optim_factory import create_optimizer

from train import train_one_epoch

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils

import modeling_pretrain

from torch.utils.data import DataLoader,Dataset,DistributedSampler,RandomSampler, SequentialSampler
import torchvision
import torchvision.utils
from dataset import sMRI_Dataset
import discriminator
import torch.distributed as dist


def get_args():
    parser = argparse.ArgumentParser('pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=200, type=int)

    parser.add_argument("--gpuid", default=[0], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")

    # Model parameters
    parser.add_argument('--model', default='pretrain_base_patch30', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--model_D', default='ResNet_10', type=str, metavar='MODEL',
                        help='Name of model_D to train')

    parser.add_argument('--mask_ratio', default=0.76, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=[150,180,150], type=tuple,
                        help='images input size for backbone')

    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
                        

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--up_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

 
    # Dataset parameters
    parser.add_argument('--root_dir', default='./CAMCAN', type=str,
                        help='dataset path')
    parser.add_argument('--data_file', default='CAMCAN_DATA_NEW.txt', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir_G', default='./pretrained_ViT_model',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_dir_D', default='./pretrained_Discriminator',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True) 
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # set these parameters in file modeling_pretrain.py

    # # feature fusion parameters (default false in pre-training) 
    # parser.add_argument('--vis', default=False)
    # parser.add_argument('--feature_fusion', default=False)
    # parser.add_argument('--selected_token_num', default=12, type=int)

    # # image_distort parameters
    # parser.add_argument('--nonlinear_rate', default=0.9, type=float)
    # parser.add_argument('--local_rate', default=0.5, type=float)

    return parser.parse_args()

def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model

def get_model_D(args):
    print(f"Creating model_D: {args.model_D}")
    model_D = create_model(
        args.model_D,
        pretrained=False,
    )

    return model_D


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # create model
    model = get_model(args)
    patch_size = model.encoder.patch.patch_size
    print("Patch size = %s" % str(patch_size))  
    args.window_size = (args.input_size[0] // patch_size[0], args.input_size[1] // patch_size[1], args.input_size[2] // patch_size[2])
    args.patch_size = patch_size

    # creat discriminator model
    model_D = get_model_D(args)

    dataset_train = sMRI_Dataset(root_dir = args.root_dir,data_file = args.data_file)

    num_tasks = 1
    num_training_steps_per_epoch = len(dataset_train) // args.batch_size // num_tasks
    sampler_train = RandomSampler(dataset_train)
    print("Sampler_train = %s" % str(sampler_train))
       
    log_writer = None

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker 
    )

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    model_D.to(device)
    model_D_without_ddp = model_D


    total_batch_size = args.batch_size * utils.get_world_size()
    args.lr = args.lr * total_batch_size / 4       

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    

    model = torch.nn.DataParallel(model, device_ids=args.gpuid)
    model_without_ddp = model.module

    model_D = torch.nn.DataParallel(model_D, device_ids=args.gpuid)
    model_D_without_ddp = model_D.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    optimizer_D = create_optimizer(
        args, model_D_without_ddp)

    loss_scaler = NativeScaler()
    loss_scaler_D = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    ## load vit model G
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, model_dir=args.output_dir_G)
    ## load model D
    utils.auto_load_model(
        args=args, model=model_D, model_without_ddp=model_D_without_ddp, optimizer=optimizer_D, loss_scaler=loss_scaler_D, model_dir=args.output_dir_D)

    print(f"Start training for {args.epochs} epochs")

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch) # for tensorboard

        if  (epoch+1) % 20 == 0:
            train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, model_D, optimizer_D, epoch, loss_scaler, loss_scaler_D, args.clip_grad,
            patch_size=patch_size[0], batch_size=args.batch_size,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            save=True,
        ) 
        else:
            train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, model_D, optimizer_D, epoch, loss_scaler, loss_scaler_D, args.clip_grad,
            patch_size=patch_size[0], batch_size=args.batch_size,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            save=False,
        )           

        if args.output_dir_G:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                ## save vit model G
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_dir=args.output_dir_G)
                ## save model D
                utils.save_model(
                    args=args, model=model_D, model_without_ddp=model_D_without_ddp, optimizer=optimizer_D,
                    loss_scaler=loss_scaler_D, epoch=epoch, model_dir=args.output_dir_D)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir_G and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir_G, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir_G:
        Path(opts.output_dir_G).mkdir(parents=True, exist_ok=True)
    if opts.output_dir_D:
        Path(opts.output_dir_D).mkdir(parents=True, exist_ok=True)
    main(opts)

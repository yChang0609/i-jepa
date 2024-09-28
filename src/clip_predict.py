import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import sys
import yaml

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet_tiny

from src.helper import (
    load_encoder,
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

# Directly encoder 
from einops import rearrange
from src.models.VAE.encoder_decoder import Encoder

# -- MineCLIP
import hashlib
import hydra
from omegaconf import OmegaConf
from mineclip import MineCLIP

# --
log_timings = True
log_freq = 10
checkpoint_freq = 150
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


class DirectlyEncoder(nn.Module):
    def __init__(self, image_size, out_dim):
        self.encoder = Encoder(
                in_channels = 3, 
                in_feature_width =  image_size[0],
                final_feature_width = 4,
                stem_channels=256, 
                num_repeat=2
            )
        
        self.final_layer = nn.Linear(
            self.encoder.last_channels*self.encoder.final_feature_width*self.encoder.final_feature_width,
            out_dim
            )
    def forward(self, images):
        batch_size = images[0]
        x = self.encoder(images)
        x = rearrange(x, "B C H W  -> B (C H W)", C=self.encoder.last_channels, H=self.encoder.final_feature_width)
        return self.final_layer(x)


class Predictor(nn.Module):
    def __init__(self, image_size, pred_dim, pre_train=None, freeze=True):
        super().__init__()
        self.have_pretrain = False
        if pre_train == None:
            encode_model = DirectlyEncoder(image_size, pred_dim)
        else :
            self.have_pretrain = True
            encode_model = pre_train
            if freeze:
                for param in encode_model.parameters():
                    param.requires_grad = False
            self.output_layer = nn.Sequential(
                    nn.AvgPool1d(kernel_size=pre_train.embed_dim),
                    nn.BatchNorm1d(pre_train.embed_dim),
                    nn.Linear(pre_train.embed_dim, pred_dim),
            )
        self.encode_model = encode_model

    def forward(self, images):
        x = self.encode_model(images)
        if self.have_pretrain:
            x = self.output_layer(x)
        return x

def resize_and_crop(img, target_size=(160, 256)):
    # [batch, channels, height, width]
    _, _, h, w = img.shape
    target_h, target_w = target_size
    
    # Step 1
    resized_img = F.interpolate(img, size=(h, target_w), mode='bilinear', align_corners=False)
    
    # Step 2
    start_h = (h - target_h) // 2  # 中心開始的高度
    cropped_img = resized_img[:, :, start_h:start_h + target_h, :]  # 裁剪高度
    
    return cropped_img

def main(args, mount_path, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    conv_channels = args['meta']['conv_channels']
    conv_strides = args['meta']['conv_strides']

    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    root_path =  os.path.join(mount_path, root_path)
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- Trianing setting 
    clip_pred_type = args['training']['cilp_pred']['predictor_type']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    folder =  os.path.join(mount_path, folder)
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    # dump = os.path.join(folder, 'params-ijepa.yaml')
    # with open(dump, 'w') as f:
    #     yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    pretrain_latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    pretrain_load_path =  os.path.join(folder, r_file) if r_file is not None else pretrain_latest_path

    tag = "clip_predict"
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    pred_save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    pred_latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'pred_loss'),
                           ('%.2e', 'mem'),
                           ('%d', 'time (ms)'))
    
    # -- init model
    @hydra.load_clip(config_name="conf", config_path=f"{mount_path}/logs/mineclip", version_base="1.1")
    def load_clip(cfg):
        OmegaConf.set_struct(cfg, False)
        ckpt = cfg.pop("ckpt")
        OmegaConf.set_struct(cfg, True)
        assert (
            hashlib.md5(open(ckpt.path, "rb").read()).hexdigest() == ckpt.checksum
        ), "broken ckpt"
        model = MineCLIP(**cfg).to(device)
        model.load_ckpt(ckpt.path, strict=True)
        return model

    clip_model = load_clip()

    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        conv_channels = conv_channels,
        conv_strides = conv_strides)
    del predictor

    encoder, _ = load_encoder(
        device=device,
        r_path=pretrain_load_path,
        encoder=encoder)
    
    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    dataset_imgnet, train_loader, test_loader = make_imagenet_tiny(
            transform=transform,
            batch_size=batch_size,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)

    predictor = Predictor(
        image_size=[224,224], 
        pred_dim=clip_model.clip_model.vision_model.output_dim,
        pre_train=(None if clip_pred_type=="Directly" else encoder),
        freeze=(False if clip_pred_type=="Directly" else True),
        )
    
    optimizer = torch.optim.AdamW(predictor.parameters())
    criterion = nn.MSELoss()
    predictor.to(device)
    
    def save_checkpoint(epoch):
        save_dict = {
            'predictor':predictor.state_dict(),
            'opt': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
        }
        if rank == 0:
            torch.save(save_dict, pred_latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, pred_save_path.format(epoch=f'{epoch + 1}'))

    start_epoch = 0
    num_epochs = 100 if num_epochs > 100 else num_epochs
    torch.cuda.empty_cache() 
    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        testing_set_loss_meter = AverageMeter()

        for itr, udata in enumerate(train_loader):
            imgs, _ = udata
            imgs = imgs.to(device)

            def train_step():
                optimizer.zero_grad()

                with torch.no_grad():
                    resize_images = resize_and_crop(imgs)
                    clip_features = clip_model.encode_video(resize_images)#[batch, 512]

                outputs = predictor(imgs)
                
                loss = criterion(outputs, clip_features)
                loss.backward()
                optimizer.step()
                grad_stats = grad_logger(predictor.named_parameters())

                return (float(loss), grad_stats)
            
            (loss, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)
 

            # -- Logging
            def log_stats():
                mem = torch.cuda.max_memory_allocated() / 1024.**2
                csv_logger.log(epoch + 1, itr, loss, mem, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   mem,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss), 'loss is nan'

        predictor.eval()
        with torch.no_grad():
            for imgs, _ in test_loader:
                imgs = imgs.to(device)

                resize_images = resize_and_crop(imgs)
                clip_features = clip_model.encode_video(resize_images)#[batch, 512]

                outputs = predictor(imgs)

                loss = criterion(outputs, clip_features)
                testing_set_loss_meter.update(loss)
        predictor.train()
        logger.info(f'Test set Accuracy[top-1/top-5]: {testing_set_loss_meter.avg}')

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)

if __name__ == "__main__":
    main()

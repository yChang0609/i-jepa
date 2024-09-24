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

from math import sqrt
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

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
from src.datasets.imagenet1k import make_imagenet1k

from src.helper import (
    load_encoder,
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

from src.models.VAE.categorical_vae import CategoricalVAE
from src.models.VAE.vae import VAE
from src.models.vision_transformer import VisionTransformer
from einops import rearrange, repeat, reduce
from PIL import Image

# --
log_timings = True
log_freq = 10
checkpoint_freq = 100
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class JEPAbaseVAE(nn.Module):
    def __init__(self, pre_train:VisionTransformer, use_amp, vae_type='normal', freeze = True):
        super(JEPAbaseVAE, self).__init__()
        self.jepa = pre_train
        if freeze:
            for param in self.jepa.parameters():
                param.requires_grad = False

        ## -- VAE differen code
        if vae_type == 'normal':
            vae = VAE(z_dim=4096, 
                      in_channels=self.jepa.embed_dim, 
                      in_feature_width=sqrt(self.jepa.patch_embed.num_patches), 
                      use_amp=use_amp) 
        if vae_type == 'categorical':
            vae = CategoricalVAE(stoch_dim=32, # 32 * 32
                                in_channels=self.jepa.embed_dim, 
                                in_feature_width=sqrt(self.jepa.patch_embed.num_patches), 
                                dyanmic_hidden_dim=self.jepa.embed_dim, 
                                use_amp=use_amp)
        assert not vae==None
        self.vae = vae
    

    def load_model(self, r_path):
        try:
            checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
            epoch = checkpoint['epoch']
            
            # -- loading encoder
            pretrained_dict = checkpoint['vae']
            for k, v in pretrained_dict.items():
                self.vae.state_dict()[k[len("module."):]].copy_(v)
            logger.info(f'loaded pretrained encoder from epoch {epoch}')

            logger.info(f'read-path: {r_path}')
            del checkpoint

        except Exception as e:
            logger.info(f'Encountered exception when loading checkpoint {e}')
            epoch = 0

        return epoch

class EmbDecoder(nn.Module):
    def __init__(self, emb_channel, in_size, recon_image_width):
        super().__init__()
        backbone = []
        channels = emb_channel 
        feat_width = 4
        original_in_channels = 3
        self.in_size = in_size
        szie = in_size
        while True:
            if szie == recon_image_width//2: 
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels//2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))
            szie *=2

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=original_in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        x = rearrange(x, "B (H W) C  -> B C H W",H=self.in_size)
        obs_hat = self.backbone(x)
        return obs_hat


def main(args, mount_path, vae_type , resume_preempt=False):

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
    kl_weight = args['training']['vae']['kl_weight']
    num_sample = args['training']['vae']['num_sample']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    folder =  os.path.join(mount_path, folder)
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    img_folder = os.path.join(folder, f'vae-{vae_type}_images')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

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
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path =  os.path.join(folder, r_file) if r_file is not None else latest_path

    tag = f"vae-{vae_type}"
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.2e', 'mem'),
                           ('%d', 'time (ms)'),
                           ('%.5f', 'visual_loss'),
                           ('%d', 'visual_time (ms)'))

    # -- init model
    jepa_encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        conv_channels = conv_channels,
        conv_strides = conv_strides)

    jepa_encoder, _ = load_encoder(
        device=device,
        r_path=load_path,
        encoder=jepa_encoder)
    
    del predictor
    torch.cuda.empty_cache() 

    

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    dataset, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(unsupervised_loader)
    
    logger.info("Create Model")
    jepa_vae = JEPAbaseVAE(jepa_encoder, use_amp=use_bfloat16, vae_type=vae_type)
    vae_optimizer = torch.optim.AdamW(jepa_vae.vae.parameters())
    vae_criterion = nn.MSELoss()
    jepa_vae.jepa.to(device)
    jepa_vae.vae.to(device)

    emb_decoder = EmbDecoder(jepa_encoder.embed_dim, sqrt(jepa_encoder.patch_embed.num_patches), crop_size)
    visual_optimizer = torch.optim.AdamW(emb_decoder.parameters())
    visual_criterion = nn.MSELoss()
    emb_decoder.to(device)
    
    logger.info(jepa_vae)
    logger.info(emb_decoder)
    
    def save_checkpoint(epoch):
        save_dict = {
            'vae':jepa_vae.vae.state_dict(),
            'opt': vae_optimizer.state_dict(),
            'epoch': epoch,
            'loss': vae_loss_meter.avg,
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    start_epoch = 0
    num_epochs = 100 if num_epochs > 100 else num_epochs
    # -- TRAINING LOOP

    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        vae_loss_meter = AverageMeter()
        recon_loss_meter = AverageMeter()
        kl_loss_meter = AverageMeter()
        time_meter = AverageMeter()

        visual_loss_meter = AverageMeter()
        visual_time_meter = AverageMeter()
        
        for itr, udata in enumerate(unsupervised_loader):
            imgs, _ = udata
            imgs = imgs.to(device)

            def vae_train_step():
                vae_optimizer.zero_grad()
                # -- JEPA and VAE
                emb = jepa_vae.jepa(imgs)
                output = jepa_vae.vae(emb)
                recon, gt = output[0], output[1]


                # -- Loss computing
                print(recon.shape)
                print(gt.shape)
                recon_loss = vae_criterion(recon, gt)
                ## -- VAE differen code
                if vae_type == 'normal':
                    mu, logvar = output[2],output[3]
                    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

                if vae_type == 'categorical':
                    logits = output[2]
                    q_p = F.softmax(logits, dim=-1) # Convert the categorical codes into probabilities
                    # KL divergence between gumbel-softmax distribution
                    eps = 1e-7

                    # Entropy of the logits
                    h1 = q_p * torch.log(q_p + eps)

                    # Cross entropy with the categorical distribution
                    h2 = q_p * np.log(1. / jepa_vae.vae.stoch_dim + eps)
                    kld_loss = torch.mean(torch.sum(h1 - h2, dim =(1,2)), dim=0)
                
                loss = recon_loss + kl_weight*kld_loss
                loss.backward()
                vae_optimizer.step()
                return float(loss), float(recon_loss), float(kl_weight*kld_loss)
            
            loss_list, etime = gpu_timer(vae_train_step)
            vae_loss_meter.update(loss_list[0])
            recon_loss_meter.update(loss_list[1])
            kl_loss_meter.update(loss_list[2])
            time_meter.update(etime)


            def visual_train_step():
                visual_optimizer.zero_grad()
                with torch.no_grad():
                    emb = jepa_vae.jepa(imgs)
                hat_imgs = emb_decoder(emb)
                loss = visual_criterion(hat_imgs, imgs)
                loss.backward()
                visual_optimizer.step()
                return float(loss)
            
            visual_loss, visual_etime = gpu_timer(visual_train_step)
            visual_loss_meter.update(visual_loss)
            visual_time_meter.update(visual_etime)

            # -- Logging
            def log_stats():
                mem = torch.cuda.max_memory_allocated() / 1024.**2
                csv_logger.log(
                    epoch + 1, itr, 
                    loss_list[0], mem, etime,
                    visual_loss, visual_etime)
                if (itr % log_freq == 0) or np.isnan(loss_list[0]) or np.isinf(loss_list[0]):
                    logger.info('[%d, %5d] VAE loss: %.3f (Recon: %.3f, KL: %.3f)'
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   vae_loss_meter.avg,
                                   recon_loss_meter.avg,
                                   kl_loss_meter.avg,
                                   mem,
                                   time_meter.avg))
                    logger.info('[%d, %5d] visual loss: %.3f '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   visual_loss_meter.avg,
                                   visual_time_meter.avg))
            log_stats()
            assert not np.isnan(loss_list[0]), 'loss is nan'

        with torch.no_grad():
            def tensor2pil(tensor):
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
                images = tensor * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
                images = torch.clamp(images, 0, 1)
                image = images.permute(1, 2, 0).numpy()
                image_np = (image * 255).astype(np.uint8)
                return Image.fromarray(image_np)

            for images, _ in unsupervised_loader:
                idx = torch.randint(0, images.size(0), (1,)).item() 
                random_images = images.to(device)
                emb = jepa_vae.jepa(random_images)
                z_dis = jepa_vae.vae.encode(emb)
                
                het_emb_list = [jepa_vae.vae.decode(jepa_vae.vae.sample(z_dis)) for _ in range(num_sample) ]
                
                hat_jepa_images = emb_decoder(emb)
                hat_vae_images_list = [emb_decoder(het_emb_list[i]) for i in range(num_sample)]
  
                
                random_image_np = random_images[idx].cpu()
                hat_jepa_image_np = hat_jepa_images[idx].cpu()
                hat_vae_image_np_list = [hat_vae_images_list[i][idx].cpu() for i in range(num_sample)]

                image_save_path = os.path.join(img_folder, f'vae_{epoch+1}-image')
                os.mkdir(image_save_path)
                image = tensor2pil(random_image_np)
                image.save(image_save_path+'/orig_imag.png')
                image = tensor2pil(hat_jepa_image_np)
                image.save(image_save_path+'/hat_jepa_image.png')
                for i in range(num_sample):
                    image = tensor2pil(hat_vae_image_np_list[i])
                    image.save(image_save_path+f'/hat_vae_image_{i+1}.png')

                break  
        
        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % vae_loss_meter.avg)
        save_checkpoint(epoch+1)

if __name__ == "__main__":
    main()

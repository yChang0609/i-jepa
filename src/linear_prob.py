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

def compute_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class LinearProbe(nn.Module):
    def __init__(self, pre_train, num_classes, freeze = True):
        super(LinearProbe, self).__init__()
        self.pre_train = pre_train
        if freeze:
            for param in pre_train.parameters():
                param.requires_grad = False

        self.linear = nn.Sequential(
            nn.BatchNorm1d(pre_train.embed_dim),
            nn.Linear(pre_train.embed_dim, num_classes),
            # nn.ReLU(),
        )

    def forward(self, image):
        x = self.pre_train(image)
        x = x.mean(dim = 1)
        return self.linear(x)

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

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']
    folder =  os.path.join(mount_path, folder)
    
    if not os.path.exists(folder):
        os.makedirs(folder)

    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
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

    l_tag = "linear_probe"
    l_log_file = os.path.join(folder, f'{l_tag}_r{rank}.csv')
    l_save_path = os.path.join(folder, f'{l_tag}' + '-ep{epoch}.pth.tar')
    l_latest_path = os.path.join(folder, f'{l_tag}-latest.pth.tar')
    l_test_acc_file = os.path.join(folder, f'{l_tag}_test_acc_r{rank}.csv')

    # -- make csv_logger
    csv_logger = CSVLogger(l_log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'acc_1'),
                           ('%.5f', 'acc_5'),
                           ('%.2e', 'mem'),
                           ('%d', 'time (ms)'))
    acc_logger = CSVLogger(l_test_acc_file,
                        ('%d', 'epoch'),
                        ('%.5f', 'acc_1'),
                        ('%.5f', 'acc_5'))
    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        conv_channels = conv_channels,
        conv_strides = conv_strides)
    target_encoder = copy.deepcopy(encoder)

    encoder, _ = load_encoder(
        device=device,
        r_path=load_path,
        encoder=encoder)
    
    del predictor
    del target_encoder
    torch.cuda.empty_cache() 

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
    
    linear_probe = LinearProbe(encoder, len(dataset_imgnet.classes), freeze = True)
    optimizer = torch.optim.AdamW(linear_probe.parameters())
    criterion = nn.CrossEntropyLoss()
    linear_probe.to(device)
    
    def save_checkpoint(epoch):
        save_dict = {
            'linear_probe':linear_probe.state_dict(),
            'opt': optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
        }
        if rank == 0:
            torch.save(save_dict, l_latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, l_save_path.format(epoch=f'{epoch + 1}'))

    start_epoch = 0
    num_epochs = 100
    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        loss_meter = AverageMeter()
        time_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()

        for itr, udata in enumerate(train_loader):
            imgs, labels = udata
            imgs, labels = imgs.to(device), labels.to(device)

            def train_step():
                optimizer.zero_grad()
                outputs = linear_probe(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                grad_stats = grad_logger(linear_probe.named_parameters())

                acc1, acc5 = compute_accuracy(outputs, labels, topk=(1, 5))
                # correct = 0
                # total = 0
                # _, predicted = torch.max(outputs.data, 1)
                # total += labels.size(0)
                # correct += (predicted == labels).sum().item()
                # accuracy = 100 * correct / total
                return (float(loss), acc1, acc5, grad_stats)
            
            (loss, acc1, acc5, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)
            acc1_meter.update(acc1)
            acc5_meter.update(acc5)

            # -- Logging
            def log_stats():
                mem = torch.cuda.max_memory_allocated() / 1024.**2
                csv_logger.log(epoch + 1, itr, loss, acc1, acc5, mem, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                '[Accuracy Top-1:%.2f %%]'
                                '[Accuracy Top-5:%.2f %%]'
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   acc1_meter.avg,
                                   acc5_meter.avg,
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

        linear_probe.eval()
        test_acc1_meter = AverageMeter()
        test_acc5_meter = AverageMeter()
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = linear_probe(imgs)
                acc1, acc5 = compute_accuracy(outputs, labels, topk=(1, 5))
                test_acc1_meter.update(acc1)
                test_acc5_meter.update(acc5)
            
        linear_probe.train()
        acc_logger.log(epoch+1, test_acc1_meter.avg, test_acc5_meter.avg)
        logger.info(f'Test set Accuracy[top-1/top-5]: {test_acc1_meter.avg}% / {test_acc5_meter.avg}%')

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)

if __name__ == "__main__":
    main()

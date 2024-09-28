# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed

class TrainMode:
    jepa = "jepa"
    jepa_linear_prob = "jepa_linear_prob"
    vit_cls = "vit_cls"
    vae = "vae"
    cat_vae = "cat_vae"
    ae = "ae"
    clip_pred = "clip_pred"

parser = argparse.ArgumentParser()
parser.add_argument(
    '--train', type=str,
    help=f'decision training mode \
    {TrainMode.jepa}, \
    {TrainMode.jepa_linear_prob}, \
    {TrainMode.vit_cls}, \
    {TrainMode.vae}, \
    {TrainMode.cat_vae},\
    {TrainMode.ae},\
    {TrainMode.clip_pred}')

parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


def process_main(rank, train_mode, fname, world_size, devices):
    # -- enviroment var
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])
    mount_path_env = os.getenv('MOUNT_PATH', "./")
    
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    # -- Training
    if train_mode == TrainMode.jepa or train_mode == TrainMode.vit_cls :    
        params = None
        with open(fname, 'r') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
            logger.info('loaded params...')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)

    # -- Evaluation
    if train_mode == TrainMode.jepa_linear_prob \
        or train_mode == TrainMode.vae \
        or train_mode == TrainMode.cat_vae \
        or train_mode == TrainMode.ae \
        or train_mode == TrainMode.clip_pred:
        params = None
        yaml_flie = os.path.join(fname,'params-ijepa.yaml')
        with open(yaml_flie, 'r') as y_file:
            params = yaml.load(y_file, Loader=yaml.FullLoader)
            logger.info('loaded params...')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    
    ## --Select work
    if train_mode == TrainMode.jepa:
        from src.train import main as jepa_main
        jepa_main(args=params, mount_path=mount_path_env)
    elif train_mode == TrainMode.jepa_linear_prob:
        from src.linear_prob import main as linear_prob_main
        linear_prob_main(args=params, mount_path=mount_path_env)
    elif train_mode == TrainMode.vit_cls:
        from src.trian_vit_cls import main as vit_main
        vit_main(args=params, mount_path=mount_path_env)
    elif train_mode == TrainMode.clip_pred:
        from src.clip_predict import main as clip_pred_main
        clip_pred_main(args=params, mount_path=mount_path_env)
    elif train_mode == TrainMode.vae  \
        or train_mode == TrainMode.cat_vae  \
        or train_mode == TrainMode.ae     :
        if train_mode == TrainMode.vae:
            vae_type = 'normal'
        elif train_mode == TrainMode.cat_vae:
            vae_type = 'categorical'
        elif train_mode == TrainMode.ae:
            vae_type = 'auto_encoder'
        from src.train_vae import main as vae_main
        vae_main(args=params, mount_path=mount_path_env, vae_type=vae_type)

if __name__ == '__main__':
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.train, args.fname, num_gpus, args.devices)
        ).start()

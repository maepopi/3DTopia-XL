import os
import sys
import io

import torch
import numpy as np
from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from dva.io import load_from_config
from dva.ray_marcher import RayMarcher
from dva.utils import to_device

import logging
from time import time
logger = logging.getLogger("cache_conditioner.py")

def main(config):
    logging.basicConfig(level=logging.INFO)

    rank = 0
    seed = 42 
    device = 0
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    is_master = rank == 0
    
    dataset = load_from_config(config.dataset)
    conditioner = load_from_config(config.model.conditioner)
    conditioner = conditioner.to(device)
    
    # computing values for the given viewpoints
    rm = RayMarcher(
        config.image_height,
        config.image_width,
        **config.rm,
    ).to(device)

    loader = DataLoader(
        dataset,
        batch_size=config.train.get("batch_size", 4),
        pin_memory=True,
        num_workers=config.train.get("n_workers", 1),
        drop_last=False,
    )

    conditioner.eval()
    iteration = 0
    for b, batch in enumerate(loader):
        logger.info(f"Iteration {iteration}")
        batch = to_device(batch, device)
        bs = batch['gt'].shape[0]
        with torch.no_grad():
            y = conditioner(batch, rm, amp=False, precision_dtype=torch.float32)
        for bidx in range(bs):
            fitted_param = y[bidx, ...].clone()
            folder = batch['folder'][bidx]
            key = batch['key'][bidx]
            fitted_param_url = "./data/obj-2048-518reso-dino-cond/{}{}.pt".format(folder, key)
            torch.save(fitted_param, fitted_param_url)
        iteration += 1


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # manually enable tf32 to get speedup on A100 GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # set config
    config = OmegaConf.load(str(sys.argv[1]))
    config_cli = OmegaConf.from_cli(args_list=sys.argv[2:])
    if config_cli:
        logger.info("overriding with following values from args:")
        logger.info(OmegaConf.to_yaml(config_cli))
        config = OmegaConf.merge(config, config_cli)

    main(config)

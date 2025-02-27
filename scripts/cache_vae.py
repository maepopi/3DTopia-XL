import os
import sys
import io

import torch
import numpy as np
from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from dva.io import load_from_config
from dva.utils import to_device

import logging
from time import time
logger = logging.getLogger("cache_vae.py")

def main(config):
    logging.basicConfig(level=logging.INFO)

    rank = 0
    seed = 42 
    device = 0
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    is_master = rank == 0
    
    dataset = load_from_config(config.dataset)
    vae = load_from_config(config.model.vae)
    vae_state_dict = torch.load(config.model.vae_checkpoint_path, map_location='cpu')
    vae.load_state_dict(vae_state_dict['model_state_dict'])
    
    vae = vae.to(device)
    loader = DataLoader(
        dataset,
        batch_size=config.train.get("batch_size", 4),
        pin_memory=True,
        num_workers=config.train.get("n_workers", 1),
        drop_last=False,
    )

    vae.eval()
    iteration = 0
    for b, batch in enumerate(loader):
        logger.info(f"Iteration {iteration}")
        batch = to_device(batch, device)
        bs = batch['gt'].shape[0]
        with torch.no_grad():
            latent = vae.encode(batch['gt'].reshape(bs*config.model.num_prims, config.model.dim_feat, config.model.prim_shape, config.model.prim_shape, config.model.prim_shape)).parameters
            latent = latent.reshape(bs, config.model.num_prims, *latent.shape[1:]).detach()
        
        for bidx in range(bs):
            fitted_param = latent[bidx, ...].clone()
            folder = batch['folder'][bidx]
            key = batch['key'][bidx]
            fitted_param_url = "./data/klvae_2048_scaleup_cache/vae-{}{}.pt".format(folder, key)
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

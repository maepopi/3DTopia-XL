import os
import sys
import io

import torch
import numpy as np
from omegaconf import OmegaConf

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from dva.ray_marcher import RayMarcher
from dva.io import load_from_config
from dva.losses import process_losses
from dva.utils import to_device
from dva.visualize import visualize_primvolume

import logging
import time
logger = logging.getLogger("train_ae.py")

def main(config):
    dist.init_process_group("nccl")
    logging.basicConfig(level=logging.INFO)

    rank = int(os.environ["RANK"])
    assert rank == dist.get_rank()
    device = int(os.environ["LOCAL_RANK"])
    seed = config.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    logger.info(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    is_master = rank == 0

    if is_master:
        os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
        OmegaConf.save(config, f"{config.output_dir}/config.yml")
        logger.info(f"saving results to {config.output_dir}")
        logger.info(f"starting training with the config: {OmegaConf.to_yaml(config)}")
    
    amp = config.train.amp
    scaler = torch.cuda.amp.GradScaler() if amp else None
    dataset = load_from_config(config.dataset)
    if not config.dataset.incl_srt:
        assert config.model.vae.in_channels == config.model.dim_feat
    model = load_from_config(config.model.vae)
    if config.checkpoint_path:
        state_dict = torch.load(config.checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict['model_state_dict'])
    iteration = 0
    model = DDP(model.to(device), device_ids=[device])

    # computing values for the given viewpoints
    rm = RayMarcher(
        config.image_height,
        config.image_width,
        **config.rm,
    ).to(device)

    loss_fn = load_from_config(config.loss).to(device)
    optimizer = load_from_config(config.optimizer, params=model.parameters())
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=config.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.train.get("batch_size", 4),
        pin_memory=True,
        sampler=train_sampler,
        num_workers=config.train.get("n_workers", 1),
        drop_last=True,
    )

    model.train()
    for epoch in range(config.train.n_epochs):
        train_sampler.set_epoch(epoch)
        if is_master:
            ts = time.time()
        for b, batch in enumerate(loader):
            with torch.cuda.amp.autocast(enabled=amp):
                batch = to_device(batch, device)
                if is_master:
                    te = time.time()
                    data_time = te - ts
                    ts = te
                bs = batch['gt'].shape[0]
                batch['gt'] = batch['gt'].reshape(bs * config.model.num_prims, config.model.vae.in_channels, config.model.prim_shape, config.model.prim_shape, config.model.prim_shape)

                preds = {}
                recon, posterior = model(batch['gt'])
                preds['recon'] = recon
                preds['posterior'] = posterior

                loss, loss_dict = loss_fn(batch, preds, iteration)
                _loss_dict = process_losses(loss_dict)

                if is_master:
                    te = time.time()
                    model_time = te - ts
                    ts = te

                if torch.isnan(loss):
                    loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                    logger.warning(f"some of the losses is NaN, skipping: {loss_str}")
                    continue

                optimizer.zero_grad()
                if amp:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            if is_master:
                te = time.time()
                bp_time = te - ts
                ts = te

            if is_master and iteration % config.train.log_every_n_steps == 0:
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                logger.info(f"epoch={epoch}, iter={iteration}[data={data_time:.3f}|model={model_time:.3f}|bp={bp_time:.3f}]: {loss_str}")

            if iteration % config.train.summary_every_n_steps == 0:
                if is_master:
                    if config.dataset.incl_srt:
                        recon_srt_param = torch.mean(recon[:, 0:4, ...], dim=[2,3,4]).reshape(bs, config.model.num_prims, 4)
                        # invert normalization
                        recon_srt_param[..., 0:1] = recon_srt_param[..., 0:1] / 10. + 0.05
                        recon_feat_param = recon[:, 4:, ...]
                    else:
                        recon_srt_param = batch['input_param'][:, :, :4].reshape(bs, config.model.num_prims, 4)
                        recon_feat_param = recon
                    # invert normalization
                    recon_feat_param[:, 0:1, ...] /= 5.
                    recon_feat_param[:, 1:, ...] = (recon_feat_param[:, 1:, ...] + 1) / 2.
                    recon_feat_param = recon_feat_param.reshape(bs, config.model.num_prims, -1)
                    recon_param = torch.concat([recon_srt_param, recon_feat_param], dim=-1)
                    visualize_primvolume("{}/{:06d}_recon.jpg".format(config.output_dir, iteration), batch, recon_param, rm, device)
                    visualize_primvolume("{}/{:06d}_gt.jpg".format(config.output_dir, iteration), batch, batch['input_param'], rm, device)
                    logger.info(f"saving checkpoint after {iteration} steps")
                    params = {
                        "model_state_dict": model.module.state_dict(),
                        "epoch": epoch,
                        "iteration": iteration,
                        "optimizer": optimizer.state_dict(),
                    }
                    torch.save(params, f"{config.output_dir}/checkpoints/latest.pt")
                dist.barrier()

            iteration += 1

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
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

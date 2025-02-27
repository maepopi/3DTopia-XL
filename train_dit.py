import os
import sys
import io

import torch
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from copy import deepcopy

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from dva.ray_marcher import RayMarcher
from dva.io import load_from_config
from dva.losses import process_losses
from dva.utils import to_device
from dva.visualize import visualize_primvolume
from models.diffusion import create_diffusion
from models.vae3d_dib_correct import DiagonalGaussianDistribution

import logging
from time import time
logger = logging.getLogger("train_dit.py")


def get_grad_norm(model):
    grads = [
        param.grad.detach().flatten()
        for param in model.parameters()
        if param.grad is not None
    ]
    norm = torch.cat(grads).norm()
    return norm

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


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
    scaler = None
    precision = config.train.get("precision", 'tf32')
    if precision == 'tf32':
        assert not amp, "Precision in tf32, cannot turn on amp training"
        precision_dtype = torch.float32
    elif precision == 'fp16':
        assert amp
        precision_dtype = torch.float16
        scaler = torch.cuda.amp.GradScaler()
    elif precision == 'bf16':
        assert amp
        scaler = torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=100)
        precision_dtype = torch.bfloat16
    else:
        raise ValueError
    dataset = load_from_config(config.dataset)
    model = load_from_config(config.model.generator)
    vae = load_from_config(config.model.vae)
    conditioner = load_from_config(config.model.conditioner)
    vae_state_dict = torch.load(config.model.vae_checkpoint_path, map_location='cpu')
    vae.load_state_dict(vae_state_dict['model_state_dict'])
    
    if config.checkpoint_path:
        state_dict = torch.load(config.checkpoint_path, map_location='cpu')
        if "ema" in state_dict:
            loaded_key = 'ema'
        else:
            loaded_key = 'model_state_dict'
        model.load_state_dict(state_dict[loaded_key])
    
    iteration = 0
    vae = vae.to(device)
    conditioner = conditioner.to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[device])
    diffusion = create_diffusion(**config.diffusion)  # default: 1000 steps, linear noise schedule

    # computing values for the given viewpoints
    rm = RayMarcher(
        config.image_height,
        config.image_width,
        **config.rm,
    ).to(device)

    optimizer = load_from_config(config.optimizer, params=model.parameters())
    scheduler = load_from_config(config.scheduler, optimizer=optimizer)
    
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

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()
    vae.eval()
    conditioner.eval()

    perchannel_norm = False
    if "latent_mean" in config.model:
        latent_mean = torch.Tensor(config.model.latent_mean)[None, None, :].to(device)
        latent_std = torch.Tensor(config.model.latent_std)[None, None, :].to(device)
        assert latent_mean.shape[-1] == config.model.generator.in_channels
        perchannel_norm = True
    start_time = time()
    log_steps = 0
    logger.info(f"Starting Training at rank={rank}")
    for epoch in range(config.train.n_epochs):
        train_sampler.set_epoch(epoch)
        for b, batch in enumerate(loader):
            batch = to_device(batch, device)
            bs = batch['gt'].shape[0]
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=precision_dtype, enabled=amp):
                    if 'vae_cache' in batch:
                        g_param = batch['vae_cache'].reshape(bs*config.model.num_prims, *batch['vae_cache'].shape[2:])
                        if not perchannel_norm:
                            latent = DiagonalGaussianDistribution(g_param).sample().mul_(config.model.latent_nf)
                        else:
                            latent = DiagonalGaussianDistribution(g_param).sample()
                    else:
                        if not perchannel_norm:
                            latent = vae.encode(batch['gt'].reshape(bs*config.model.num_prims, config.model.dim_feat, config.model.prim_shape, config.model.prim_shape, config.model.prim_shape)).sample().mul_(config.model.latent_nf)
                        else:
                            latent = vae.encode(batch['gt'].reshape(bs*config.model.num_prims, config.model.dim_feat, config.model.prim_shape, config.model.prim_shape, config.model.prim_shape)).sample()
                input_srt_param = batch['input_param'].clone()[:, :, 0:4]
                if not perchannel_norm:
                    input_srt_param[:, :, 0:1] = (input_srt_param[:, :, 0:1] - 0.05) * 10
                x = torch.concat([input_srt_param, latent.reshape(bs, config.model.num_prims, -1)], dim=-1)
                if perchannel_norm:
                    x = ((x - latent_mean) / latent_std) * config.model.latent_nf
                # probably have pytorch3d raymarching instead of conditioner, thus cannot do amp
                y = conditioner(batch, rm, amp, precision_dtype)
            
            t = torch.randint(0, diffusion.num_timesteps, (bs,), device=device)
            model_kwargs = dict(y=y, precision_dtype=precision_dtype, enable_amp=amp)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss_total"].mean()
            _loss_dict = process_losses(loss_dict)

            if torch.isnan(loss):
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                logger.warning(f"some of the losses is NaN, skipping: {loss_str}")
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if iteration % config.train.log_every_n_steps == 0:
                grad_norm = get_grad_norm(model).cpu().item()
                current_scale = scaler._scale.cpu().item() if scaler is not None else 0
            optimizer.zero_grad()
            update_ema(ema, model.module)
            scheduler.step()

            log_steps += 1
            if iteration % config.train.log_every_n_steps == 0:
                torch.cuda.synchronize()
                end_time = time()
                sec_per_step = (end_time - start_time) / log_steps
                if is_master:
                    loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                    logger.info(f"epoch={epoch}, iter={iteration}[{sec_per_step:.3f}sec/step]: grad_norm:{grad_norm:.6f} scaler:{current_scale} {loss_str}")
                log_steps = 0
                start_time = time()

            if iteration % config.train.summary_every_n_steps == 0:
                if is_master:
                    with torch.no_grad():
                        inf_bs = min(8, config.train.batch_size)
                        inf_x = torch.randn(inf_bs, *x.shape[1:]).to(x).float()
                        # Using condition in current training batch
                        model_kwargs = dict(y=y[:inf_bs, ...].float())
                        samples = diffusion.p_sample_loop(
                            ema.forward, inf_x.shape, inf_x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
                        )
                        recon_param = samples.reshape(inf_bs, config.model.num_prims, -1)
                        if perchannel_norm:
                            recon_param = recon_param / config.model.latent_nf * latent_std + latent_mean
                        recon_srt_param = recon_param[:, :, 0:4]
                        recon_feat_param = recon_param[:, :, 4:] # [8, 2048, 64]
                        recon_feat_param_list = []
                        # one-by-one to avoid oom
                        for inf_bidx in range(inf_bs):
                            if not perchannel_norm:
                                decoded = vae.decode(recon_feat_param[inf_bidx, ...].reshape(1*config.model.num_prims, *latent.shape[-4:]) / config.model.latent_nf)
                            else:
                                decoded = vae.decode(recon_feat_param[inf_bidx, ...].reshape(1*config.model.num_prims, *latent.shape[-4:]))
                            recon_feat_param_list.append(decoded.detach())
                        recon_feat_param = torch.concat(recon_feat_param_list, dim=0)
                        # invert normalization
                        if not perchannel_norm:
                            recon_srt_param[:, :, 0:1] = (recon_srt_param[:, :, 0:1] / 10) + 0.05
                        recon_feat_param[:, 0:1, ...] /= 5.
                        recon_feat_param[:, 1:, ...] = (recon_feat_param[:, 1:, ...] + 1) / 2.
                        recon_feat_param = recon_feat_param.reshape(inf_bs, config.model.num_prims, -1)
                        recon_param = torch.concat([recon_srt_param, recon_feat_param], dim=-1)
                        recon_param_wgt_srt = torch.concat([batch['input_param'][:inf_bs, :, 0:4], recon_feat_param], dim=-1)
                        visualize_primvolume("{}/{:06d}_recon.jpg".format(config.output_dir, iteration), batch, recon_param, rm, device)
                        visualize_primvolume("{}/{:06d}_recon_GTsrt.jpg".format(config.output_dir, iteration), batch, recon_param_wgt_srt, rm, device)
                        visualize_primvolume("{}/{:06d}_gt.jpg".format(config.output_dir, iteration), batch, batch['input_param'][:inf_bs], rm, device)
                        logger.info(f"saving checkpoint after {iteration} steps")
                        params = {
                            "model_state_dict": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "iteration": iteration,
                        }
                        torch.save(params, f"{config.output_dir}/checkpoints/latest.pt")
                        torch.cuda.empty_cache()
                dist.barrier()

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

import os
import sys

import torch
import numpy as np
from omegaconf import OmegaConf

import torch.distributed as dist
from torch.utils.data import DataLoader

from dva.ray_marcher import RayMarcher
from dva.io import load_from_config
from dva.losses import process_losses
from dva.utils import to_device
from dva.visualize import render_primsdf, visualize_primsdf_box

import logging

device = torch.device("cuda")

logger = logging.getLogger("train_fitting.py")

def main(config):
    dist.init_process_group("nccl")

    logging.basicConfig(level=logging.INFO)

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    os.makedirs(f"{config.output_dir}/checkpoints", exist_ok=True)
    OmegaConf.save(config, f"{config.output_dir}/config.yml")
    logger.info(f"saving results to {config.output_dir}")
    logger.info(f"starting training with the config: {OmegaConf.to_yaml(config)}")

    dataset = load_from_config(config.dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=config.train.get("batch_size", 4),
        pin_memory=False,
        sampler=train_sampler,
        num_workers=config.train.get("n_workers", 1),
        drop_last=True,
        worker_init_fn=lambda _: np.random.seed(),
    )
    
    model = load_from_config(config.model, mesh_obj=dataset.mesh_obj, f_sdf=dataset.f_sdf, geo_fn=dataset.geo_fn_list, asset_list=dataset.asset_list)
    model = model.to(device)

    # computing values for the given viewpoints
    rm = RayMarcher(
        config.image_height,
        config.image_width,
        **config.rm,
    ).to(device)

    loss_fn = load_from_config(config.loss).to(device)
    optimizer = load_from_config(config.optimizer, params=model.parameters())
    iteration = 0
    model.train()
    # stage 1, optimizing SDF only
    while True:
        if iteration >= config.train.shape_fit_steps:
            model.eval()
            visualize_primsdf_box("{}/{:06d}_box_nosampling.png".format(config.output_dir, iteration), model, rm, device)
            render_primsdf("{}/{:06d}_rendering.png".format(config.output_dir, iteration), model, rm, device)
            model.train()
            break
        for b, batch in enumerate(loader):
            batch = to_device(batch, device)
            for k, v in batch.items():
                batch[k] = v.reshape(config.train.batch_size * config.dataset.chunk_size, *v.shape[2:])

            if local_rank == 0 and batch is None:
                logger.info(f"batch {b} is None, skipping")
                continue

            if local_rank == 0 and iteration >= config.train.shape_fit_steps:
                logger.info(f"stopping after {config.train.shape_fit_steps}")
                break
            
            batch['pts'].requires_grad_(True)
            preds = model(batch['pts'])
            preds['prim_scale'] = (1 / model.scale.reshape(1, model.num_prims, 1).repeat(1, 1, 3))

            loss, loss_dict = loss_fn(batch, preds, iteration)
            _loss_dict = process_losses(loss_dict)

            if torch.isnan(loss):
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                logger.warning(f"some of the losses is NaN, skipping: {loss_str}")
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if local_rank == 0 and iteration % config.train.log_every_n_steps == 0:
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                logger.info(f"iter={iteration}: {loss_str}")

            if (
                local_rank == 0
                # and iteration
                and iteration % config.train.summary_every_n_steps == 0
            ):
                logger.info(
                    f"saving summary to {config.output_dir} after {iteration} steps"
                )

            iteration += 1

        pass

    # stage 2, optimizing texture
    optimizer_tex = load_from_config(config.optimizer, params=[model.feat_param])
    while True:
        if iteration >= config.train.tex_fit_steps:
            if (local_rank == 0):
                logger.info(f"Texture Optimization Done: saving checkpoint after {iteration} steps")
                model.eval()
                visualize_primsdf_box("{}/{:06d}_box_nosampling.png".format(config.output_dir, iteration), model, rm, device)
                render_primsdf("{}/{:06d}_rendering.png".format(config.output_dir, iteration), model, rm, device)
                model.train()
                if config.train.save_fp16:
                    model = model.half()
                params = {
                    "model_state_dict": model.state_dict(),
                }
                torch.save(params, f"{config.output_dir}/checkpoints/tex-{iteration:06d}.pt")
            break
        for b, batch in enumerate(loader):
            batch = to_device(batch, device)
            for k, v in batch.items():
                batch[k] = v.reshape(config.train.batch_size * config.dataset.chunk_size, *v.shape[2:])

            if local_rank == 0 and batch is None:
                logger.info(f"batch {b} is None, skipping")
                continue

            if local_rank == 0 and iteration >= config.train.tex_fit_steps:
                logger.info(f"stopping after {config.train.tex_fit_steps}")
                break
            
            preds = model(batch['tex_pts'])
            preds['prim_scale'] = (1 / model.scale.reshape(1, model.num_prims, 1).repeat(1, 1, 3))

            loss, loss_dict = loss_fn(batch, preds, iteration)
            _loss_dict = process_losses(loss_dict)

            if torch.isnan(loss):
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                logger.warning(f"some of the losses is NaN, skipping: {loss_str}")
                continue

            optimizer_tex.zero_grad()
            loss.backward()
            optimizer_tex.step()

            if local_rank == 0 and iteration % config.train.log_every_n_steps == 0:
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in _loss_dict.items()])
                logger.info(f"iter={iteration}: {loss_str}")
            iteration += 1
        pass


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # set config
    config = OmegaConf.load(str(sys.argv[1]))
    config_cli = OmegaConf.from_cli(args_list=sys.argv[2:])
    if config_cli:
        logger.info("overriding with following values from args:")
        logger.info(OmegaConf.to_yaml(config_cli))
        config = OmegaConf.merge(config, config_cli)

    main(config)

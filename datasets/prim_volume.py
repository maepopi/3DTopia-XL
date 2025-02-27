import io
import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset


import logging

logger = logging.getLogger(__name__)

class DCTManifoldDataset(Dataset):
    def __init__(
        self,
        manifold_url_template,
        obj_name_list_path,
        num_prims,
        dim_feat,
        prim_shape,
        incl_srt=False,
        device='cpu',
        **kwargs,
    ):
        super().__init__()
        assert os.path.exists(obj_name_list_path)
        with open(obj_name_list_path, 'r') as f:
            obj_name_list = f.readlines()
        self.manifold_url_template = manifold_url_template
        self.obj_list = obj_name_list
        self.num_prims = num_prims
        self.dim_feat = dim_feat
        self.prim_shape = prim_shape
        assert not incl_srt
        self.incl_srt = incl_srt
        self.device = device
    
    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, index):
        sample = {}
        obj_meta = self.obj_list[index]
        folder, key = obj_meta[:-1].split("/")
        sample['folder'] = folder
        sample['key'] = key
        manifold_obj_path = self.manifold_url_template.format(folder=folder, key=key)
        try:
            ckpt = torch.load(manifold_obj_path, map_location=self.device)
            weights_dict = ckpt['model_state_dict']
            srt_param = weights_dict['srt_param']
            feat_param = weights_dict['feat_param']
            if torch.isnan(srt_param).any() or torch.isnan(feat_param).any():
                raise ValueError
        except:
            srt_param = torch.zeros(self.num_prims, 4)
            feat_param = torch.zeros(self.num_prims, self.dim_feat * self.prim_shape ** 3)
        srt_param = srt_param.float()
        feat_param = feat_param.float()
        feat_param = feat_param.reshape(self.num_prims, self.dim_feat, self.prim_shape ** 3)
        feat_param[:, 1:, :] = torch.clip(feat_param[:, 1:, :], min=0.0, max=1.0)
        feat_param = feat_param.reshape(self.num_prims, self.dim_feat * self.prim_shape ** 3)
        sample['input_param'] = torch.concat([srt_param, feat_param], dim = -1)

        fft_feat_param = torch.view_as_real(torch.fft.fft(feat_param)) # [nprims, 6*8*8*8, 2]
        
        fft_feat_param = fft_feat_param.reshape(self.num_prims, self.dim_feat, self.prim_shape, self.prim_shape, self.prim_shape, 2).permute(0, 1, 5, 2, 3, 4).reshape(self.num_prims, self.dim_feat * 2, self.prim_shape, self.prim_shape, self.prim_shape)
        normalized_feat_param = fft_feat_param / 20.
        # [nprims, 12, 8, 8, 8]
        sample['gt'] = normalized_feat_param
        return sample

class ManifoldDataset(Dataset):
    def __init__(
        self,
        manifold_url_template,
        obj_name_list_path,
        num_prims,
        dim_feat,
        prim_shape,
        incl_srt=False,
        device='cpu',
        **kwargs,
    ):
        super().__init__()
        assert os.path.exists(obj_name_list_path)
        with open(obj_name_list_path, 'r') as f:
            obj_name_list = f.readlines()
        self.manifold_url_template = manifold_url_template
        self.obj_list = obj_name_list
        self.num_prims = num_prims
        self.dim_feat = dim_feat
        self.prim_shape = prim_shape
        self.incl_srt = incl_srt
        self.device = device
    
    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, index):
        sample = {}
        obj_meta = self.obj_list[index]
        folder, key = obj_meta[:-1].split("/")
        sample['folder'] = folder
        sample['key'] = key
        manifold_obj_path = self.manifold_url_template.format(folder=folder, key=key)
        try:
            ckpt = torch.load(manifold_obj_path, map_location=self.device)
            weights_dict = ckpt['model_state_dict']
            srt_param = weights_dict['srt_param']
            feat_param = weights_dict['feat_param']
            if torch.isnan(srt_param).any() or torch.isnan(feat_param).any():
                raise ValueError
        except:
            srt_param = torch.zeros(self.num_prims, 4)
            feat_param = torch.zeros(self.num_prims, self.dim_feat * self.prim_shape ** 3)
        srt_param = srt_param.float()
        feat_param = feat_param.float()
        feat_param = feat_param.reshape(self.num_prims, self.dim_feat, self.prim_shape ** 3)
        feat_param[:, 1:, :] = torch.clip(feat_param[:, 1:, :], min=0.0, max=1.0)
        feat_param = feat_param.reshape(self.num_prims, self.dim_feat * self.prim_shape ** 3)
        # sample['srt_param'] = srt_param
        # sample['feat_param'] = feat_param
        sample['input_param'] = torch.concat([srt_param, feat_param], dim = -1)

        normalized_srt_param = srt_param.clone()
        normalized_srt_param[:, 0:1] = (normalized_srt_param[:, 0:1] - 0.05) * 10 # heuristic normalization
        normalized_srt_param = normalized_srt_param[..., None, None, None].repeat(1, 1, self.prim_shape, self.prim_shape, self.prim_shape)

        # [nprims, 6, 8, 8, 8]
        normalized_feat_param = feat_param.clone().reshape(self.num_prims, self.dim_feat, self.prim_shape, self.prim_shape, self.prim_shape)
        # sdf heuristic normalization
        normalized_feat_param[:, 0:1, ...] *= 5
        # color, mat normalization [0, 1] -> [-1, 1]
        normalized_feat_param[:, 1:, ...] = normalized_feat_param[:, 1:, ...] * 2. - 1.

        # [nprims, 10, 8, 8, 8]
        if self.incl_srt:
            sample['gt'] = torch.concat([normalized_srt_param, normalized_feat_param], dim = 1)
        else:
            sample['gt'] = normalized_feat_param
        return sample

class VAECacheManifoldDataset(Dataset):
    def __init__(
        self,
        manifold_url_template,
        vaecache_url_template,
        obj_name_list_path,
        num_prims,
        dim_feat,
        prim_shape,
        incl_srt=False,
        device='cpu',
        **kwargs,
    ):
        super().__init__()
        assert os.path.exists(obj_name_list_path)
        with open(obj_name_list_path, 'r') as f:
            obj_name_list = f.readlines()
        self.manifold_url_template = manifold_url_template
        self.vaecache_url_template = vaecache_url_template
        self.obj_list = obj_name_list
        self.num_prims = num_prims
        self.dim_feat = dim_feat
        self.prim_shape = prim_shape
        self.incl_srt = incl_srt
        self.device = device
    
    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, index):
        sample = {}
        obj_meta = self.obj_list[index]
        folder, key = obj_meta[:-1].split("/")
        sample['folder'] = folder
        sample['key'] = key
        manifold_obj_path = self.manifold_url_template.format(folder=folder, key=key)
        vaecache_path = self.vaecache_url_template.format(folder=folder, key=key)
        try:
            ckpt = torch.load(manifold_obj_path, map_location=self.device)
            weights_dict = ckpt['model_state_dict']
            srt_param = weights_dict['srt_param']
            feat_param = weights_dict['feat_param']
            if torch.isnan(srt_param).any() or torch.isnan(feat_param).any():
                raise ValueError
            
            vae_ckpt = torch.load(vaecache_path, map_location=self.device)
            if torch.isnan(vae_ckpt).any():
                raise ValueError
        except:
            srt_param = torch.zeros(self.num_prims, 4)
            feat_param = torch.zeros(self.num_prims, self.dim_feat * self.prim_shape ** 3)
            vae_ckpt = torch.zeros(self.num_prims, 2, 4, 4, 4)
        sample['vae_cache'] = vae_ckpt.float()
        srt_param = srt_param.float()
        feat_param = feat_param.float()
        feat_param = feat_param.reshape(self.num_prims, self.dim_feat, self.prim_shape ** 3)
        feat_param[:, 1:, :] = torch.clip(feat_param[:, 1:, :], min=0.0, max=1.0)
        feat_param = feat_param.reshape(self.num_prims, self.dim_feat * self.prim_shape ** 3)
        # sample['srt_param'] = srt_param
        # sample['feat_param'] = feat_param
        sample['input_param'] = torch.concat([srt_param, feat_param], dim = -1)

        normalized_srt_param = srt_param.clone()
        normalized_srt_param[:, 0:1] = (normalized_srt_param[:, 0:1] - 0.05) * 10 # heuristic normalization
        normalized_srt_param = normalized_srt_param[..., None, None, None].repeat(1, 1, self.prim_shape, self.prim_shape, self.prim_shape)

        # [nprims, 6, 8, 8, 8]
        normalized_feat_param = feat_param.clone().reshape(self.num_prims, self.dim_feat, self.prim_shape, self.prim_shape, self.prim_shape)
        # sdf heuristic normalization
        normalized_feat_param[:, 0:1, ...] *= 5
        # color, mat normalization [0, 1] -> [-1, 1]
        normalized_feat_param[:, 1:, ...] = normalized_feat_param[:, 1:, ...] * 2. - 1.

        # [nprims, 10, 8, 8, 8]
        if self.incl_srt:
            sample['gt'] = torch.concat([normalized_srt_param, normalized_feat_param], dim = 1)
        else:
            sample['gt'] = normalized_feat_param
        return sample

class AllCacheManifoldDataset(Dataset):
    def __init__(
        self,
        manifold_url_template,
        vaecache_url_template,
        cond_url_template,
        obj_name_list_path,
        num_prims,
        dim_feat,
        prim_shape,
        incl_srt=False,
        device='cpu',
        **kwargs,
    ):
        super().__init__()
        assert os.path.exists(obj_name_list_path)
        with open(obj_name_list_path, 'r') as f:
            obj_name_list = f.readlines()
        self.manifold_url_template = manifold_url_template
        self.cond_url_template = cond_url_template
        self.vaecache_url_template = vaecache_url_template
        self.obj_list = obj_name_list
        self.num_prims = num_prims
        self.dim_feat = dim_feat
        self.prim_shape = prim_shape
        self.incl_srt = incl_srt
        self.device = device
    
    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, index):
        sample = {}
        obj_meta = self.obj_list[index]
        folder, key = obj_meta[:-1].split("/")
        sample['folder'] = folder
        sample['key'] = key
        manifold_obj_path = self.manifold_url_template.format(folder=folder, key=key)
        cond_obj_path = self.cond_url_template.format(folder=folder, key=key)
        vaecache_path = self.vaecache_url_template.format(folder=folder, key=key)
        try:
            ckpt = torch.load(manifold_obj_path, map_location=self.device)
            weights_dict = ckpt['model_state_dict']
            srt_param = weights_dict['srt_param']
            feat_param = weights_dict['feat_param']
            if torch.isnan(srt_param).any() or torch.isnan(feat_param).any():
                raise ValueError

            cond_ckpt = torch.load(cond_obj_path, map_location=self.device)
            if torch.isnan(cond_ckpt).any():
                raise ValueError

            vae_ckpt = torch.load(vaecache_path, map_location=self.device)
            if torch.isnan(vae_ckpt).any():
                raise ValueError
        except:
            srt_param = torch.zeros(self.num_prims, 4)
            feat_param = torch.zeros(self.num_prims, self.dim_feat * self.prim_shape ** 3)
            vae_ckpt = torch.zeros(self.num_prims, 2, 4, 4, 4)
            if "views" in cond_obj_path:
                cond_ckpt = torch.zeros(1370 * 4, 768)
            else:
                cond_ckpt = torch.zeros(1370, 768)

        sample['vae_cache'] = vae_ckpt.float()
        sample['cond'] = cond_ckpt.float()
        srt_param = srt_param.float()
        feat_param = feat_param.float()
        feat_param = feat_param.reshape(self.num_prims, self.dim_feat, self.prim_shape ** 3)
        feat_param[:, 1:, :] = torch.clip(feat_param[:, 1:, :], min=0.0, max=1.0)
        feat_param = feat_param.reshape(self.num_prims, self.dim_feat * self.prim_shape ** 3)
        # sample['srt_param'] = srt_param
        # sample['feat_param'] = feat_param
        sample['input_param'] = torch.concat([srt_param, feat_param], dim = -1)

        normalized_srt_param = srt_param.clone()
        normalized_srt_param[:, 0:1] = (normalized_srt_param[:, 0:1] - 0.05) * 10 # heuristic normalization
        normalized_srt_param = normalized_srt_param[..., None, None, None].repeat(1, 1, self.prim_shape, self.prim_shape, self.prim_shape)

        # [nprims, 6, 8, 8, 8]
        normalized_feat_param = feat_param.clone().reshape(self.num_prims, self.dim_feat, self.prim_shape, self.prim_shape, self.prim_shape)
        # sdf heuristic normalization
        normalized_feat_param[:, 0:1, ...] *= 5
        # color, mat normalization [0, 1] -> [-1, 1]
        normalized_feat_param[:, 1:, ...] = normalized_feat_param[:, 1:, ...] * 2. - 1.

        # [nprims, 10, 8, 8, 8]
        if self.incl_srt:
            sample['gt'] = torch.concat([normalized_srt_param, normalized_feat_param], dim = 1)
        else:
            sample['gt'] = normalized_feat_param
        return sample
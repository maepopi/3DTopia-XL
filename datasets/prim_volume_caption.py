import io
import os
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
import open_clip


import logging
logger = logging.getLogger(__name__)

class VAECacheManifoldDataset(Dataset):
    def __init__(
        self,
        manifold_url_template,
        vaecache_url_template,
        obj_name_list_path,
        caption_list_path,
        num_prims,
        dim_feat,
        prim_shape,
        lan_model_spec="ViT-L-14",
        incl_srt=False,
        device='cpu',
        **kwargs,
    ):
        super().__init__()
        assert os.path.exists(obj_name_list_path)
        assert os.path.exists(caption_list_path)
        with open(obj_name_list_path, 'r') as f:
            obj_name_list = f.readlines()
        with open(caption_list_path, 'r') as f:
            caption_list = f.readlines()
        self.manifold_url_template = manifold_url_template
        self.vaecache_url_template = vaecache_url_template
        self.obj_list = obj_name_list
        self.caption_list = caption_list
        # we assume the order of object is same in obj_list and caption_list
        assert len(self.obj_list) == len(self.caption_list), "len(obj_list)={} is not equal to len(caption_list)={}".format(len(self.obj_list), len(self.caption_list))
        self.num_prims = num_prims
        self.dim_feat = dim_feat
        self.prim_shape = prim_shape
        self.incl_srt = incl_srt
        self.tokenizer = open_clip.get_tokenizer(lan_model_spec)
        self.device = device
    
    def __len__(self):
        return len(self.obj_list)

    def __getitem__(self, index):
        sample = {}
        obj_meta = self.obj_list[index]
        caption_meta = self.caption_list[index]
        folder, key = obj_meta[:-1].split("/")
        caption_key, caption = caption_meta[:-1].split("@", 1)
        assert caption_key == key
        sample['folder'] = folder
        sample['key'] = key
        sample['caption_raw'] = caption
        sample['caption_token'] = self.tokenizer([caption])[0]
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
import torch
import trimesh
import numpy as np
from torch.utils.data.dataset import Dataset
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from utils.mesh import Mesh
from utils.meshutils import scale_to_unit_cube, rotation_matrix
from dva.geom import GeometryModule
import cubvh
import os

import logging

logger = logging.getLogger(__name__)

class SampleSDFTexMatMesh(Dataset):
    def __init__(
        self,
        mesh_file_path,
        glb_f=None,
        num_surface_samples=300000,
        num_near_samples=200000,
        sample_std=0.01,
        chunk_size=1024,
        is_train=True,
        **kwargs,
    ):
        super().__init__()
        if isinstance(mesh_file_path, str) and os.path.exists(mesh_file_path):
            assert mesh_file_path.endswith("glb")
        if glb_f is not None:
            _data = trimesh.load(glb_f, file_type='glb')
        else:
            _data = trimesh.load(mesh_file_path)
        self.chunk_size = chunk_size
        device = "cpu"
        # always convert scene to mesh, and apply all transforms...
        if isinstance(_data, trimesh.Scene):
            # print(f"[INFO] load trimesh: concatenating {len(_data.geometry)} meshes.")
            _concat = []
            # loop the scene graph and apply transform to each mesh
            scene_graph = _data.graph.to_flattened() # dict {name: {transform: 4x4 mat, geometry: str}}
            for k, v in scene_graph.items():
                name = v['geometry']
                if name in _data.geometry and isinstance(_data.geometry[name], trimesh.Trimesh):
                    transform = v['transform']
                    _concat.append(_data.geometry[name].apply_transform(transform))
            # we do not concatenate here
            _mesh_list = _concat
        else:
            _mesh_list = [_data]
        
        _asset_list = []
        _valid_mesh_list = []
        max_xyz_list = []
        min_xyz_list = []
        sampling_weights = []
        for each_mesh in _mesh_list:
            _asset = Mesh.parse_trimesh_data(each_mesh, device=device)
            # clean faces have less than 3 connected components
            tmp_mesh = trimesh.Trimesh(_asset.v, _asset.f, process=False)
            cc = trimesh.graph.connected_components(tmp_mesh.face_adjacency, min_len=3)
            if not len(cc) > 0:
                if _asset.v.shape[0] > 5:
                    _valid_mesh_list.append(each_mesh)
                    _asset_list.append(_asset)
                    sampling_weights.append(tmp_mesh.area.item())
                    max_xyz_list.append(_asset.v.max(0)[0])
                    min_xyz_list.append(_asset.v.min(0)[0])
                else:
                    logger.info(f"Less than 3 connected components found! Drop trimesh element with vertices shape:{tmp_mesh.vertices.shape}")
                continue
            _valid_mesh_list.append(each_mesh)
            cc_mask = np.zeros(len(tmp_mesh.faces), dtype=np.bool_)
            cc_mask[np.concatenate(cc)] = True
            tmp_mesh.update_faces(cc_mask)
            # remove unreferenced vertices, update vertices and texture coordinates accordingly
            referenced = np.zeros(len(tmp_mesh.vertices), dtype=bool)
            referenced[tmp_mesh.faces] = True
            inverse = np.zeros(len(tmp_mesh.vertices), dtype=np.int64)
            inverse[referenced] = np.arange(referenced.sum())
            tmp_mesh.update_vertices(mask=referenced, inverse=inverse)
            # update texture coordinates
            updated_vt = _asset.vt[referenced, :].clone()
            _asset.vt = updated_vt
            # renormalize vertices to unit cube after outliers removal
            _asset.v = torch.from_numpy(tmp_mesh.vertices).float()
            _asset.f = torch.from_numpy(tmp_mesh.faces).long()
            _asset.ft = torch.from_numpy(tmp_mesh.faces).long()
            _asset_list.append(_asset)
            # use sum of face area as weights
            sampling_weights.append(tmp_mesh.area.item())
            max_xyz_list.append(_asset.v.max(0)[0])
            min_xyz_list.append(_asset.v.min(0)[0])
        # scale to unit cube
        global_max_xyz, _ = torch.stack(max_xyz_list).max(0)
        global_min_xyz, _ = torch.stack(min_xyz_list).min(0)
        bb_centroid = (global_max_xyz + global_min_xyz) / 2.
        global_scale_max = (global_max_xyz - global_min_xyz).max()
        for ast in _asset_list:
            zero_mean_pts = ast.v.clone() - bb_centroid
            ast.v = zero_mean_pts * (1.8 / global_scale_max)

        self.asset_list = _asset_list
        _merged_mesh = trimesh.util.concatenate(_valid_mesh_list)
        _merged_vertices = torch.from_numpy(_merged_mesh.vertices).to(torch.float32)
        _merged_vertices = scale_to_unit_cube(_merged_vertices)
        _merged_mesh.vertices = _merged_vertices
        _merged_faces = torch.from_numpy(_merged_mesh.faces).to(torch.long)
        self.mesh_obj = _merged_mesh
        self.f_sdf = cubvh.cuBVH(_merged_vertices.cuda(), _merged_faces.cuda())
        self.mesh_p3d_obj = Meshes([_merged_vertices], [_merged_faces])
        surface_samples = sample_points_from_meshes(self.mesh_p3d_obj, num_surface_samples)
        near_samples = sample_points_from_meshes(self.mesh_p3d_obj, num_near_samples)
        near_samples = near_samples + torch.rand_like(near_samples) * sample_std
        self.sampled_points = torch.concat([surface_samples, near_samples], dim=1)[0]
        self.sampled_sdf = self.f_sdf.signed_distance(self.sampled_points, return_uvw=False, mode='raystab')[0].cpu()[..., None] * (-1)

        # instantiation of geometry function
        self.geo_fn_list = []
        sampling_weights = torch.Tensor(sampling_weights)
        self.sampling_weights = sampling_weights / torch.sum(sampling_weights)
        self.num_sampled_pts = (self.sampling_weights * (num_surface_samples + num_near_samples)).to(torch.int)
        if not torch.sum(self.num_sampled_pts) == (num_surface_samples + num_near_samples):
            diff = num_surface_samples + num_near_samples - torch.sum(self.num_sampled_pts)
            self.num_sampled_pts[-1] += diff
        sampled_gt_tex = []
        sampled_gt_pts = []
        sampled_gt_mat = []
        for idx, ast in enumerate(_asset_list):
            topology = {
                "v": ast.v.to(torch.float32),
                "vi": ast.f.to(torch.long),
                "vti": ast.ft.to(torch.long),
                "vt": ast.vt.to(torch.float32),
                "n_verts": ast.v.shape[0],
            }
            # assert ast.albedo.shape[:2] == ast.metallicRoughness.shape[:2]
            geo_fn = GeometryModule(
                v=topology['v'],
                vi=topology['vi'],
                vt=topology['vt'],
                vti=topology['vti'],
                impaint=False,
                uv_size=ast.albedo.shape[:2],
            )
            self.geo_fn_list.append(geo_fn)
            num_sampled_pts = self.num_sampled_pts[idx]
            if num_sampled_pts == 0:
                continue
            sampled_texture, sampled_pts = geo_fn.rand_sample_3d_uv(num_sampled_pts, ast.albedo)
            sampled_material, _ = geo_fn.sample_uv_from_3dpts(sampled_pts, ast.metallicRoughness)
            sampled_gt_tex.append(torch.from_numpy(sampled_texture))
            sampled_gt_pts.append(torch.from_numpy(np.array(sampled_pts, dtype=np.float32)))
            sampled_gt_mat.append(torch.from_numpy(sampled_material[..., -2:]))
        
        self.sampled_tex = torch.concat(sampled_gt_tex, dim=0)
        self.sampled_tex_points = torch.concat(sampled_gt_pts, dim=0)
        self.sampled_mat = torch.concat(sampled_gt_mat, dim=0)
        self.idx_list = np.arange(self.sampled_sdf.shape[0])
        assert self.sampled_tex.shape[0] == self.sampled_tex_points.shape[0]
        assert self.sampled_sdf.shape[0] == self.sampled_points.shape[0]
        assert self.sampled_points.shape[0] == self.sampled_tex_points.shape[0]

    def __len__(self):
        return self.sampled_sdf.shape[0]

    def __getitem__(self, index):
        idxs = np.random.choice(self.idx_list, self.chunk_size)
        sample = {}
        sample['pts'] = self.sampled_points[idxs, :]
        sample['sdf'] = self.sampled_sdf[idxs, :]
        sample['tex_pts'] = self.sampled_tex_points[idxs, :]
        sample['tex'] = self.sampled_tex[idxs, :]
        sample['mat'] = self.sampled_mat[idxs, :]
        return sample
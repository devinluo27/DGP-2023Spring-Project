import torch
import numpy as np
from tqdm import tqdm

import srt.utils.visualize as vis
from srt.utils.common import mse2psnr, reduce_dict, gather_all
from srt.utils import nerf
from srt.utils.common import get_rank, get_world_size

import os
import os.path as osp
import math
from collections import defaultdict
from vis.marching_cube import marching_cube_infertime

class SRTTrainer_sdf:
    rand_sdf_value = torch.randn(size=(2, 5000, 1))
    def __init__(self, model, optimizer, cfg, device, out_dir, sdf_kwargs={}, infertime=False, truncate_sdf=False):
        self.model = model
        self.optimizer = optimizer
        self.config = cfg
        self.device = device
        self.out_dir = out_dir
        self.sdf_kwargs = sdf_kwargs
        self.infertime = infertime
        self.infertime_res = 64
        self.truncate_sdf = sdf_kwargs.get('truncate_sdf', False)
        self.sdf_thres = torch.tensor(sdf_kwargs.get('sdf_thres', 0.25)).to(self.device)
        print(f'[SRTTrainer_sdf] self.truncate_sdf {self.truncate_sdf}; self.sdf_thres {self.sdf_thres}')
        # self.render_kwargs = render_kwargs
        # if 'num_coarse_samples' in cfg['training']:
            # self.render_kwargs['num_coarse_samples'] = cfg['training']['num_coarse_samples']
        # if 'num_fine_samples' in cfg['training']:
            # self.render_kwargs['num_fine_samples'] = cfg['training']['num_fine_samples']

    def evaluate(self, val_loader, **kwargs):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        self.model.eval()
        eval_lists = defaultdict(list)

        loader = val_loader if get_rank() > 0 else tqdm(val_loader)
        sceneids = []

        for data in loader:
            sceneids.append(data['sceneid'])
            eval_step_dict = self.eval_step(data, **kwargs)

            for k, v in eval_step_dict.items():
                # print(k, v.shape)
                eval_lists[k].append(v)

        sceneids = torch.cat(sceneids, 0).cuda()
        sceneids = torch.cat(gather_all(sceneids), 0)

        print(f'Evaluated {len(torch.unique(sceneids))} unique scenes.')

        eval_dict = {k: torch.cat(v, 0) for k, v in eval_lists.items()}
        eval_dict = reduce_dict(eval_dict, average=True)  # Average across processes
        eval_dict = {k: v.mean().item() for k, v in eval_dict.items()}  # Average across batch_size
        print('Evaluation results:')
        print("eval_dict", eval_dict)
        return eval_dict




    def evaluate_infertime(self, val_loader, **kwargs):
        ''' Performs an evaluation in separate script.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        self.model.eval()
        eval_lists = defaultdict(list)

        loader = val_loader if get_rank() > 0 else tqdm(val_loader)
        sceneids = []

        for data in loader:
            sceneids.append(data['sceneid'])
            # get xyz and pred_sdf
            eval_step_dict = self.eval_step_infertime(data, **kwargs)

            for k, v in eval_step_dict.items():
                print(k, v.shape)
                eval_lists[k].append(v)

        sceneids = torch.cat(sceneids, 0).cuda()
        sceneids = torch.cat(gather_all(sceneids), 0)

        print(f'Evaluated {len(torch.unique(sceneids))} unique scenes.')

        eval_dict = {k: torch.cat(v, 0) for k, v in eval_lists.items()}
        eval_dict = reduce_dict(eval_dict, average=True)  # Average across processes
        eval_dict = {k: v.mean().item() for k, v in eval_dict.items()}  # Average across batch_size
        print('Evaluation results:')
        print(eval_dict)
        return eval_dict

    
    def eval_step_infertime(self, data, full_scale=False):
        with torch.no_grad():
            loss, loss_terms = self.compute_loss_infertime(data, 1000000)
        
        pred_sdf = loss_terms['pred_sdf']
        mse = loss_terms['mse']
        psnr = mse2psnr(mse)

        print('outdir', self.out_dir)
        print('scene_paths', data['scene_paths'])
        # osp.join(self.out_dir, data['sceneid'], )
        # scene_paths is a list return by listdirs
        mesh_path = osp.join(self.out_dir,data['scene_paths'][0],'mesh.stl')

        # for k,v in data.items():
            # print(k, v.shape)
        # exit()
        marching_cube_infertime(pred_sdf, self.infertime_res, write_path=mesh_path)
        return {'psnr': psnr, 'mse': mse, 'pred_sdf': pred_sdf, **loss_terms}


    @torch.no_grad()
    def compute_loss_infertime(self, data, it):
        print('compute_loss_infertime')
        res = self.infertime_res
        device = self.device

        input_images = data.get('input_images').to(device)
        lin_coords = np.linspace(-1.0, 1.0, res)
        x_coords, y_coords, z_coords = np.meshgrid(lin_coords, lin_coords, lin_coords)
        xyz = np.stack([x_coords, y_coords, z_coords], axis=-1).reshape(1, -1, 3).astype(np.float32)
        xyz = torch.tensor(xyz).to(device)

        # xyz = data.get('xyz_sdf')[:, :, :3].to(device)
        # sdf_value = data.get('xyz_sdf')[:, :, 3:].to(device)

        # input_camera_pos = data.get('input_camera_pos').to(device)
        # input_rays = data.get('input_rays').to(device)
        # target_pixels = data.get('target_pixels').to(device)

        # input_images torch.Size([2, 6, 3, 128, 128]) xyz torch.Size([2, 5000, 3]), torch.Size([2, 5000, 1])
        # print('input_images', input_images.shape, 'xyz', xyz.shape, sdf_value.shape)
        z = self.model.encoder(input_images)

        # target_camera_pos = data.get('target_camera_pos').to(device)
        # target_rays = data.get('target_rays').to(device)

        loss = 0.
        loss_terms = dict()
        n_sdf = res**3


        # shape: 1, nsdf, 3
        batch = 20000
        n_loops = n_sdf//batch + 1
        pred_sdf = []
        # divide the grid into loops, prevent out of memory
        for i in range(n_loops):
            start = i*batch
            end = min((i+1)*batch, n_sdf)
            pred_temp, extras = self.model.decoder(z, xyz[:, start:end, :])
            pred_sdf.append(pred_temp)
            # print(pred_temp.shape)

        pred_sdf = torch.cat(pred_sdf, dim=1)
        # print(pred_sdf.shape)

        # [2,5000,1]; [2,5000,1]
        assert pred_sdf.shape[1] == n_sdf
        print('pred_sdf', pred_sdf.shape)

        # if it % 1000 == 0:
        #     print('xyz', xyz[0,200:210])
        #     print('pred_sdf', pred_sdf[0,200:210], '\nsdf_value', sdf_value[0,200:210])

        # loss = loss + ((pred_sdf - sdf_value)**2).mean((1, 2))
        # loss_terms['mse'] = loss
        loss_terms['mse'] = torch.tensor([0,], dtype=torch.float32, device=self.device)


        if self.infertime:
            loss_terms['pred_sdf'] = pred_sdf

        return loss, loss_terms

    def train_step(self, data, it):
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_terms = self.compute_loss(data, it)
        loss = loss.mean(0)
        if math.isnan(loss) or math.isinf(loss):
            loss = torch.tensor(0)
        loss_terms = {k: v.mean(0).item() for k, v in loss_terms.items()}
        loss.backward()
        self.optimizer.step()
        return loss.item(), loss_terms

    def compute_loss(self, data, it):
        device = self.device

        input_images = data.get('input_images').to(device)
        xyz = data.get('xyz_sdf')[:, :, :3].to(device)
        sdf_value = data.get('xyz_sdf')[:, :, 3:].to(device)

        ## maybe interesting to use camera poses
        # input_camera_pos = data.get('input_camera_pos').to(device)
        # input_rays = data.get('input_rays').to(device)
        # target_pixels = data.get('target_pixels').to(device)

        # input_images torch.Size([2, 6, 3, 128, 128]) xyz torch.Size([2, 5000, 3]), torch.Size([2, 5000, 1])
        # print('input_images', input_images.shape, 'xyz', xyz.shape, sdf_value.shape)
        z = self.model.encoder(input_images)

        # target_camera_pos = data.get('target_camera_pos').to(device)
        # target_rays = data.get('target_rays').to(device)

        loss = 0.
        loss_terms = dict()

        # pred_sdf, extras = self.model.decoder(z, xyz, **self.render_kwargs)
        pred_sdf, extras = self.model.decoder(z, xyz)

        # [2,5000,1]; [2,5000,1]
        # print('pred_sdf', pred_sdf.shape, 'sdf_value', sdf_value.shape)

        # TODO overfit 0
        # sdf_value[...] = self.rand_sdf_value

        if it % 1000 == 0:
            print('xyz', xyz[0,200:210])
            print('pred_sdf', pred_sdf[0,200:210], '\nsdf_value', sdf_value[0,200:210])
        
        # use truncate_sdf to compute loss
        if self.truncate_sdf:
        # if False:
            # loss = torch.tensor(0., requires_grad=True)
            # diff = torch.clamp(sdf_value, min=-self.sdf_thres, max=self.sdf_thres)
            # - torch.clamp(pred_sdf, min=-self.sdf_thres, max=self.sdf_thres)

            pred_clamp = torch.min(torch.max(-self.sdf_thres, pred_sdf), self.sdf_thres)
            gt_clamp = torch.min(torch.max(-self.sdf_thres, sdf_value), self.sdf_thres)
            diff = pred_clamp - gt_clamp

            # print('diff', diff, diff.requires_grad)
            mse_tmp = torch.square(diff).mean((1, 2))
            # print('mse_tmp', mse_tmp, mse_tmp.requires_grad)
            loss = loss + mse_tmp

        else:
            loss = loss + ((pred_sdf - sdf_value)**2).mean((1, 2))

        loss_terms['mse'] = loss

        # if 'coarse_img' in extras:
        #     coarse_loss = ((extras['coarse_img'] - target_pixels)**2).mean((1, 2))
        #     loss_terms['coarse_mse'] = coarse_loss
        #     loss = loss + coarse_loss

        return loss, loss_terms

    def eval_step(self, data, full_scale=False):
        with torch.no_grad():
            loss, loss_terms = self.compute_loss(data, 1000000)

        mse = loss_terms['mse']
        psnr = mse2psnr(mse)
        return {'psnr': psnr, 'mse': mse, **loss_terms}
    
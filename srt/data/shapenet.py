import numpy as np
import imageio
import yaml
from torch.utils.data import Dataset

import os

from srt.utils.nerf import transform_points

class ShapeNetDataset(Dataset):
    def __init__(self, path, mode, points_per_item=2048, max_len=None,
                 canonical_view=True, full_scale=False, config_dict=None
                ):
        """ Loads the NMR dataset as found at
        https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
        Hosted by Niemeyer et al. (https://github.com/autonomousvision/differentiable_volumetric_rendering)
        Args:
            path (str): Path to dataset.
            mode (str): 'train', 'val', or 'test'.
            points_per_item (int): Number of target points per scene.
            max_len (int): Limit to the number of entries in the dataset.
            canonical_view (bool): Return data in canonical camera coordinates (like in SRT), as opposed
                to world coordinates.
            full_scale (bool): Return all available target points, instead of sampling.
        """
        self.path = path
        self.mode = mode
        self.points_per_item = points_per_item
        self.max_len = max_len
        self.canonical = canonical_view
        self.full_scale = full_scale
        self.imgs_per_scene = config_dict['imgs_per_scene']
        self.imgs_sample_per_scene = config_dict['imgs_sample_per_scene']
        # self.sdf_per_scene = config_dict['sdf_per_scene']
        # print(self.path)
        # exit()
        if self.mode == 'train':
            self.sdf_sample_per_scene = config_dict['sdf_sample_per_scene']
        else:
            # === 20000 when eval
            val_path = self.path + '_val'
            if os.path.isdir(val_path):
                self.path = val_path
                print('use separate val path\n')
            # 2x sample when evaluating
            self.sdf_sample_per_scene = config_dict['sdf_sample_per_scene'] * 2

        self.imgs_idx_arr = np.arange(self.imgs_sample_per_scene)
        self.posneg_equal_sampling = True

        self.positive_ratio = 0.5
        self.n_positive = int(self.sdf_sample_per_scene * self.positive_ratio)
        self.n_neg = self.sdf_sample_per_scene - self.n_positive
        print("config_dict", config_dict)
        self.truncate_sdf = config_dict.get('truncate_sdf', False)


        print('[ShapeNetDataset] path', path)

        # reproducibilityss
        self.scene_paths = sorted(os.listdir(self.path))
        self._load_sdf()


        # with open(os.path.join(path, 'metadata.yaml'), 'r') as f:
        #     metadata = yaml.load(f, Loader=yaml.CLoader)

        # class_ids = [entry['id'] for entry in metadata.values()]

        # self.scene_paths = []
        # for class_id in class_ids:
        #     with open(os.path.join(path, class_id, f'softras_{mode}.lst')) as f:
        #         cur_scene_ids = f.readlines()
        #     cur_scene_ids = [scene_id.rstrip() for scene_id in cur_scene_ids if len(scene_id) > 1]
        #     cur_scene_paths = [os.path.join(class_id, scene_id) for scene_id in cur_scene_ids]
        #     self.scene_paths.extend(cur_scene_paths)
            
        self.num_scenes = len(self.scene_paths)
        print(f'ShapeNetDataset {mode} dataset loaded: {self.num_scenes} scenes.')


    def _load_sdf(self):
        """load all sdf into dict"""
        # key: '1d99f...', value: a numpy arr (20000, 4)
        self.sdf_dict = {}
        self.sdf_positive_dict = {}
        self.sdf_neg_dict = {}

        if self.mode == 'infer_no_gt':
            raise FileNotFoundError()
        else:
            for p in self.scene_paths:
                sdf_path = os.path.join(self.path, p, 'xyzsdf.npy')
                self.sdf_dict[p] = np.load(sdf_path).astype(np.float32)
                if getattr(self, "sdf_per_scene", False):
                    self.sdf_per_scene = self.sdf_dict[p].shape[0]
                else:
                    assert self.sdf_per_scene == self.sdf_dict[p].shape[0], "number of points doesn't match"
                # TODO Truncate SDF?
                # if self.truncate_sdf:
                    # self.sdf_dict[p] 
                self.sdf_positive_dict[p] = self.sdf_dict[p][self.sdf_dict[p][:, 3] > 0]
                self.sdf_neg_dict[p] = self.sdf_dict[p][self.sdf_dict[p][:, 3] <= 0]

                print('self.sdf_positive_dict[p]', len(self.sdf_positive_dict[p]), self.sdf_per_scene) # 10000, v2 even sampling pos/neg


    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return self.num_scenes


    def __getitem__(self, idx):
        
        # randomly select 6 out of 8 images
        scene_idx = idx
        load_imgs_idx = np.random.choice(self.imgs_idx_arr, size=self.imgs_sample_per_scene, replace=False)

        # get file path to a scene
        scene_path = os.path.join(self.path, self.scene_paths[scene_idx])
        # load 6/8 images
        images = [np.asarray(imageio.imread(
            os.path.join(scene_path, f'{i:02d}.png'))) for i in load_imgs_idx]
        # print('image', images[0].max()) max: 255
        images = np.stack(images, 0).astype(np.float32) / 255.
        # print('images', images.shape) images (6, 128, 128, 3)
        input_image = np.transpose(images, (0, 3, 1, 2))
        # print('input_image', input_image.shape) # input_image (6, 3, 128, 128)


        
        # sample SDF original: [20000, 4] -> [5000?, 4]
        if self.posneg_equal_sampling:
            pos = self.sdf_positive_dict[self.scene_paths[scene_idx]]
            # print('pos', pos.shape) # 870, 4

            xyz_sdf_pos = np.random.choice(pos.shape[0], size=self.n_positive, replace=True)
            pos = pos[xyz_sdf_pos, :]

            neg = self.sdf_neg_dict[self.scene_paths[scene_idx]]
            xyz_sdf_neg = np.random.choice(neg.shape[0], size=self.n_neg, replace=False)
            neg = neg[xyz_sdf_neg, :]

            xyz_sdf = np.concatenate([pos, neg], axis=0)
            np.random.shuffle(xyz_sdf)
            
            # print('xyz_sdf', xyz_sdf.shape) # (5000, 4)
            

        else:
            load_sdf_idx = np.random.choice(self.sdf_per_scene, size=self.sdf_sample_per_scene, replace=False)
            xyz_sdf = self.sdf_dict[self.scene_paths[scene_idx]][load_sdf_idx, :]




        result = {
            'input_images':      input_image,              # [6, 3, h, w]
            'xyz_sdf': xyz_sdf,
            'sceneid':       idx,                             # int
            'scene_paths': self.scene_paths[idx], # path name
        }


        return result

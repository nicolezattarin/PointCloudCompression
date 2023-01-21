import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils import readpcd
from utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, transform


class CustomData(Dataset):
    def __init__(self, root, npts, train=True,  nfiles=None):
        super(CustomData, self).__init__()
        dirname = 'train_data' if train else 'val_data'
        path = os.path.join(root, dirname)
        self.train = train

        self.files = [os.path.join(path, item) for item in sorted(os.listdir(path))]
        if nfiles is not None:
            self.files = self.files[:nfiles]
            
        # only if point cloud is not empty
        self.files = [item for item in self.files if readpcd(item).has_points()]
        self.npts = npts

    def __getitem__(self, item):
        file = self.files[item]
        ref_cloud = readpcd(file, rtype='npy')
        ref_cloud = random_select_points(ref_cloud, m=self.npts)
        ref_cloud = pc_normalize(ref_cloud)
       
        return ref_cloud
        
    def __len__(self):
        return len(self.files)
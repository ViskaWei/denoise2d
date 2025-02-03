import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torch

class HPFDataset(Dataset):
    def __init__(self, root_dir, crop_size, mode='train', transform=False):
        super(HPFDataset, self).__init__()
        
        self.root_dir = root_dir
        self.transform = transform
        self.crop_size = crop_size
        self.mode = mode
        
        self.paths = list(Path(root_dir).glob('*.dat'))
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        data = np.fromfile(path, dtype='uint16')
        data = data.reshape(1004, 1344, 35)
        
        x, y = 0, 0
        
        if self.mode == 'train':
            x = np.random.randint(0, 1344 - self.crop_size)
            y = np.random.randint(0, 1004 - self.crop_size)
        
        img = data[y:y+self.crop_size, x:x+self.crop_size, :]
        
        if self.transform:
            flip_h = np.random.rand() > 0.5
            flip_v = np.random.rand() > 0.5
            rotate = np.random.rand() > 0.5
            
            if flip_h:
                img = np.flip(img, axis=0)
                
            if flip_v:
                img = np.flip(img, axis=1)
                
            if rotate:
                k = np.random.randint(0, 4)
                img = np.rot90(img, k=k, axes=(0, 1))
        
        img = img.float()
        img /= 2**16 - 1
        
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        noise = torch.randn_like(img) * 0.001
        noisy_image = img + noise
        
        return noisy_image, img
    
    
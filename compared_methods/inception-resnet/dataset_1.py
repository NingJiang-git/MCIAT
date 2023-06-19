from pickle import FALSE
import nibabel as nib
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import random



class MyDataset(Dataset):
    """labeled sMRI datasets."""
    
    def __init__(self, root_dir, data_file):
        """
        Args:
            root_dir (string): Directory of all the sMRIs.
            data_file (string): File name of the train/val/test split file.
        """
        self.root_dir = root_dir
        self.data_file = data_file
        
    
    def __len__(self):
        return sum(1 for line in open(self.data_file))
    
    def __getitem__(self, idx):
        df = open(self.data_file)
        lines = df.readlines()
        df.close()
        lst = lines[idx].split()
        img_name = lst[0]
        img_label = lst[1]
        image_path = os.path.join(self.root_dir, img_name)
        image = nib.load(image_path)
        image = np.squeeze(image.get_data())
        image = image[50:110,100:160,60:120]


        img_label = int(img_label)
        img_label = np.array(img_label)
        img_label = torch.from_numpy(img_label)
        image = image[np.newaxis,:]
        image = torch.from_numpy(image).float()

        image = stand(image)

        sample = {'image': image, 'label': img_label}
        
        return sample


def stand(x):
    y = torch.mean(x)
    z = torch.std(x)
    x = (x - y) / z
    return x


import os
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import fnmatch

class dataset_loader(data.Dataset):
    def __init__(self):
        self.event_noise_dir = "CSDD_dataset/Event_voxel"
        self.blur_img_dir = "CSDD_dataset/NIR_blurry"
        self.sharp_img_dir = "CSDD_dataset/NIR_clean"
        self.names = [file_name[:-4] for file_name in fnmatch.filter(os.listdir(self.event_noise_dir), '*.npy')] 
        self.names.sort()
        self.to_tensor = transforms.ToTensor()             
    def __getitem__(self, index):
        name = self.names[index]
        event_noise = np.load(os.path.join(self.event_noise_dir, name + '.npy'))      
        blur = Image.open(os.path.join(self.blur_img_dir,name + '.png')).convert('L')
        sharp = Image.open(os.path.join(self.sharp_img_dir,name + '.png')).convert('L') 

        event_noise = self.to_tensor(event_noise)
        blur = self.to_tensor(blur)
        sharp = self.to_tensor(sharp)        

        return {'event_noise':event_noise,'blur':blur,'sharp':sharp,'name':name}                     
    def __len__(self):
        return len(self.names)     
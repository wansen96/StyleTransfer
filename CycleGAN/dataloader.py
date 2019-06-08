# This file is used for loading style images and landscape dataset
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt
import random


# dataset loading function
class GANTransDataset(td.Dataset):
    def __init__(self, root_dir, mode='landscape', image_size = (512,512)):
        super(GANTransDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)
    
    def __len__(self):
        return len(self.files)
    
    def __repr__(self):
        return "GANTransDataset(mode={}, image_size={})". \
            format(self.mode, self.image_size)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        img = Image.open(img_path).convert('RGB')
        transform = tv.transforms.Compose([
            tv.transforms.Resize(self.image_size), # resize to image_size
            tv.transforms.ToTensor(), # convert to tensor
            tv.transforms.Normalize(mean = [0.5,0.5,0.5], std = [1,1,1]) # normalize to [-1,1])
            #transforms.Lambda(lambda x: x.mul_(255)),
            ])
        img = transform(img)
#         img = img.reshape(1,3,512,512) # revised
        return img

# combine two dataset into one
class GANCombinedDataset(td.Dataset):
    def __init__(self, dataset_A, dataset_B):
        super(GANCombinedDataset,self).__init__()
        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        len_A = len(self.dataset_A)
        len_B = len(self.dataset_B)
        # length of combined dataset equals to the length of smaller dataset
        self.count = 0
        self.is_A_bigger = False
        if  len_A>len_B:
            self.length = len_B
            self.index = list(range(len_A))
            self.is_A_bigger = True
        else:
            self.length = len_A
            self.index = list(range(len_B))
            
    def __len__(self):
        return self.length
        
    
    def __repr__(self):
        return "GANCombinedDataset(Dataset_A:mode{} image_size={}; Dataset_B:mode{} image_size={})". \
            format(self.dataset_A.mode, self.dataset_A.image_size,self.dataset_B.mode, self.dataset_B.image_size)
    
    def __getitem__(self,idx):
        # shuffle to randomly combine A & B every epoch
        if self.count >= self.length:
            self.count = 0
        if self.count == 0:
            random.shuffle(self.index)
        self.count+=1
        if self.is_A_bigger:
            idx_a = self.index[idx]
            idx_b = idx
        else:
            idx_b = self.index[idx]
            idx_a = idx
        
        img_a = self.dataset_A.__getitem__(idx_a)
        img_b = self.dataset_B.__getitem__(idx_b)
        return img_a, img_b

# showing image 
def torchimshow(image, ax=None):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h        
    
# Show the comparison of raw landscape image, style image and GAN transfer image 
def resultshow(rawimg, styleimg, GANimg):
    fig,axes = plt.subplots(ncols = 3, figsize = (7,3))
    torchimshow(rawimg, ax = axes[0])
    torchimshow(styleimg, ax = axes[1])
    torchimshow(GANimg, ax = axes[2])
    
    axes[0].set_title('Land Scape')
    axes[1].set_title('Style')
    axes[2].set_title('GAN Transfer')      

"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import random
import sys
import numpy as np
import torch.utils.data as data
import cv2
from PIL import Image,ImageCms,ImageOps
import os
import os.path
import time
import shutil
#from skimage import exposure


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm','.npy']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

'''
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def new_loder_for_original(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return np.array(img.convert('RGB'))

def exception_reader(path):
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return np.array(img.convert('RGB'))[0,:,:]

    except:
        return np.zeros((4288,2848),dtype=np.uint8)

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
def normalize(arr):
    """
    Linear normalization
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(3):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def aspect_resize(img,r_height):
    width,height=img.size
    aspect_ratio=width*1.0/height
    n_width,n_height=int(r_height*aspect_ratio),r_height
    return img.resize((n_width,n_height),Image.ANTIALIAS)
'''
class BRAF_dataloader(data.Dataset):
    def __init__(self, root,transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        #self.loader = npy_loader
        #self.original_img_loader=new_loder_for_original
        #self.aspect_resize=aspect_resize


    def __getitem__(self, index):
        #print('getting item, ',index)
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        # img_original = self.original_img_loader(path)
        img_p=np.load(path)
        #print(img_p.shape)
        #img_p=np.expand_dims(img_p,0)
        #print(img_p.shape)
        #img_p= np.transpose(img_p,(2,0,1))
        #print path
        target = int(path[-5])
        if self.transform is not None:
            # print(img_p.size)
            #img_p = aspect_resize(img_p,512)
            # print(img_p.size)
            img_p = self.transform(img_p)
            # print(type(img))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_p, target
        # return img_p, target
        
    def __len__(self):
        return len(self.imgs)



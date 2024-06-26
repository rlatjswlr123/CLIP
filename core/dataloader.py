import os
import torchvision
from glob import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def create_logits(x1,x2,logit_scale):                                                   
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale*x1 @ x2.t()
    logits_per_x2 = logit_scale*x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


def get_class_loader(train_path, test_path, train_aug, test_aug, train_batch_size, test_batch_size):
    train_dataset = ImageFolderWithPaths(train_path, transform=train_aug)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True
        )
    
    test_dataset = ImageFolderWithPaths(test_path, transform=test_aug)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=True
        )
    
    return train_dataloader, test_dataloader
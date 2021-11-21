"""Data loader for the bird dataset provided by the TA and modified to fit my repository's architecture

"""
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torchvision.transforms import transforms
from utils.transforms import SemanticSegmentation
import torch
import numpy as np

class BirdsDataloader():
    """
    Creates a dataloader for train and val splits
    """

    def __init__(self, args):
        self.config = args
        self.transform = {"train": transforms.Compose([


            # transforms.Resize((520, 520)),

            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.uint8),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            transforms.ConvertImageDtype(torch.float),
            # T.RandomPerspective(distortion_scale=0.6, p=1.0),

            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

            # transforms.RandomErasing(inplace=True),


            # SemanticSegmentation(),
            transforms.Resize((384, 384)),
        ]),
            "val": transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            ]
        )
        }

        self.train_dataset = datasets.ImageFolder(
            self.config.image_dir + '/train_images', transform=self.transform["train"])
        self.valid_dataset = datasets.ImageFolder(
            self.config.image_dir + '/val_images', transform=self.transform["val"])
        self.len_train_data = len(self.train_dataset)
        self.len_valid_data = len(self.valid_dataset)

        # num_train = len(self.len_train_data)
        # indices = list(range(num_train))
        # np.random.shuffle(indices)
        # split = int(np.floor(self.len_valid_data  * num_train))
        # train_idx, valid_idx = indices[split:], indices[:split]
        # train_sampler = SubsetRandomSampler(train_idx)
        # valid_sampler = SubsetRandomSampler(valid_idx)

        self.train_iterations = (
            self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
        self.valid_iterations = (
            self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)

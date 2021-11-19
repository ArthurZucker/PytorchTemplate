"""Data loader for the bird dataset provided by the TA and modified to fit my repository's architecture

"""
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from utils.transforms import SemanticSegmentation

class BirdsDataloader():
    """
    Creates a dataloader for train and val splits
    """

    def __init__(self, args):
        self.config = args
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            SemanticSegmentation()
        ])

        train_dataset = datasets.ImageFolder(self.config.image_dir + '/train_images',transform=self.transform)
        valid_dataset = datasets.ImageFolder(self.config.image_dir + '/val_images',transform=self.transform)
        self.len_train_data = len(train_dataset)
        self.len_valid_data = len(valid_dataset)

        self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
        self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

        self.train_loader = DataLoader(train_dataset,batch_size=self.config.batch_size, shuffle=True,num_workers=self.config.num_workers)
        self.valid_loader = DataLoader(valid_dataset,batch_size=self.config.batch_size, shuffle=False,num_workers=self.config.num_workers)
    
        
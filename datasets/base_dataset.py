import torch
import torchaudio

import glob
from torch.utils.data import Dataset
from utils.signal_processing import get_rnd_audio,extract_label_bat
from pandas import read_csv
from os import path

class base_dataset(Dataset):
    
    def __init__(self,wav_dir, annotation_file, input_size, transform=None, target_transform=None):
        """
        Initialises the audio dataset
        """
        self.audio_files        = read_csv(annotation_file)
        self.label              = read_csv(annotation_file)
        self.transform          = transform
        self.target_transform   = target_transform
        self.input_size         = input_size
        self.wav_dir            = wav_dir
        
    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.audio_files)
        
    def __getitem__(self,idx):
        """Returns the item for a batch

        """
        raise NotImplementedError
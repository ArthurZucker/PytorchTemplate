from dataclasses import dataclass
from typing import List
import numpy as np
from simple_parsing.helpers import list_field

import os
"""Dataclass allows to have arguments and easily use wandb's weep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""


@dataclass
class hparams:
    """Hyperparameters of Yout Model"""
    # validation frequency 
    validate_every: int = 1
    # num_classes
    num_classes: int = 20
    # feature_extracting or fine_tuning
    feature_extracting: bool = False
    # Learning rate of the Adam optimizer.
    lr: float = 1e-3
    # batch sier
    batch_size : int = 8
    # Use cuda for training
    cuda: bool = True
    # Architecture to choose, available are "denet (to come)", "sincnet (to come)", "leaf (to come)", "yolor (to come)"
    arch: str = "resnet50"
    # Agent to use, the agent has his own trining loop
    agent: str = "BaseAgent"
    # Dataset used for training
    dataloader: str = "BirdsDataloader"
    # output file for kaggle
    outfile: str = "result.csv"
    # path to images in dataset
    image_dir: str = os.path.join(os.getcwd(),"assets/bird_dataset")
    # test directory
    test_dir: str = os.path.join(os.getcwd(),"assets/bird_dataset/test_images/mistery_category")
    # Number of workers used for the dataloader
    num_workers: int  = 16
    # weight_decay
    weight_decay: float = 0.001
    # momentum 
    momentum: float = 0.9
    # seed
    seed: float = np.random.random()
    # gpu_device
    gpu_device : int = 0
    # optimizer
    optimizer: str = "SGD"
    # loss
    loss: str = "CrossEntropy"
    # checkpoint dir
    checkpoint_dir: str = os.path.join(os.getcwd(),"weights/")
    # checkpoint file
    checkpoint_file: str = ""
    # mode
    mode: str = "train"
    # Toggle testing mode, which only runs a few epochs and val
    test_mode: bool = False
    # max epoch tu run
    max_epoch: int = 100
    # async_loading
    async_loading: bool = True
    # activation function
    activation: str = "relu"

    # example of list parameters
    # layer resolutions
    fc_lay: List[int] = list_field(2048, 2048, 2048, 1024, 256, 2)
    fc_drop: List[float] = list_field(0.0, 0.0, 0.0, 0.0, 0.0, 0.1)
    fc_use_laynorm: List[bool] = list_field(
        False, False, False, False, False, False)
    fc_act: List[str] = list_field(
        "leaky_relu", "linear", "leaky_relu", "leaky_relu", "leaky_relu", "softmax")

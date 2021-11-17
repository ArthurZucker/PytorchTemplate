from dataclasses import dataclass
from typing import List

from simple_parsing.helpers import list_field

"""Dataclass allows to have arguments and easily use wandb's weep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""


@dataclass
class hparams:
    """Hyperparameters of Yout Model"""
    # num_classes
    num_classes: int = 20
    # feature_extracting or fine_tuning
    feature_extracting: bool = False
    # Learning rate of the Adam optimizer.
    lr: float = 1e-3
    # batch sier
    batch_size : int = 4
    # Use cuda for training
    cuda: bool = True
    # Architecture to choose, available are "denet (to come)", "sincnet (to come)", "leaf (to come)", "yolor (to come)"
    arch: str = "resnet50"
    # Agent to use, the agent has his own trining loop
    agent: str = "BaseAgent"
    # Dataset used for training
    dataloader: str = "BirdsDataloader"
    # path to images in dataset
    image_dir: str = "/home/arthur/Work/MVA-S1/recvis/hw3/assets/bird_dataset"
    # Number of workers used for the dataloader
    num_workers: int  = 8
    # weight_decay
    weight_decay: float = 0.01
    # momentum 
    momentum: float = 0.9
    # seed
    seed: float = 1235
    # gpu_device
    gpu_device : int = 0
    # optimizer
    optimizer: str = "SGD"
    # loss
    loss: str = "CrossEntropy"
    # checkpoint dir
    checkpoint_dir: str = "./weights"
    # checkpoint file
    checkpoint_file: str = ""
    # mode
    mode: str = "train"
    # Toggle testing mode, which only runs a few epochs and val
    test_mode: bool = True
    # max epoch tu run
    max_epoch: int = 5
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

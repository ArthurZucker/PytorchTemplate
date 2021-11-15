from dataclasses import dataclass
from simple_parsing.helpers import list_field
from typing import List

"""Dataclass allows to have arguments and easily use wandb's weep module.

Each arguments can be preceded by a comment, which will describe it when you call 'main.pu --help
An example of every datatype is provided. Some of the available arguments are also available.
Most notably, the agent, dataset, optimizer and loss can all be specified and automatically parsed
"""

@dataclass
class hparams:
    """Hyperparameters of Yout Model"""
    # Learning rate of the Adam optimizer.
    lr: float = 0.05
    # learning rate scheduler, policy to decrease
    lr_schedule: str = "poly2"
    # Use cuda for training
    cuda: bool = True
    # Architecture to choose, available are "denet (to come)", "sincnet (to come)", "leaf (to come)", "yolor (to come)"
    arch: str = "resnet50"
    #Agent to use, the agent has his own trining loop
    agent: str = "Base"
    # Dataset used for training
    dataset: str = "egyptian"
    # path to images in dataset
    img_dir: str = "/home/arthur/Work/FlyingFoxes/database/EgyptianFruitBats"
    # annotation path 
    annotation_file: str = "/home/arthur/Work/FlyingFoxes/sources/flying_foxes_study/AudioEventDetection/DENet/assets/subsampled_datset.csv"
    # Number of workers used for the dataloader
    # optimizer 
    optimizer: str = "Rmsprop"
    # loss
    loss: str = "NNL"
    # checkpoint dir
    checkpoint_dir: str = "weights"
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
    fc_lay: List[int] = list_field(2048,2048,2048,1024,256,2)
    fc_drop: List[float] = list_field(0.0,0.0,0.0,0.0,0.0,0.1)
    fc_use_laynorm: List[bool] = list_field(False,False,False,False,False,False)
    fc_act: List[str] = list_field("leaky_relu","linear","leaky_relu","leaky_relu","leaky_relu","softmax")

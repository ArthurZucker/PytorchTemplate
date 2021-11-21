import torch.nn as nn
from .resnet50 import Resnet50

class SiameseResnet50(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.net = Resnet50(args)

    
    def forward(self,pos_anchor,neg_anchor,x):
        pass
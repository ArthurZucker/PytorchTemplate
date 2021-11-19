from torchvision.models.segmentation import lraspp_mobilenet_v3_large
import torch.nn as nn
class MobileNet(nn.Module):
    def __init__(self,args=None) -> None:
        super().__init__()
        self.bird_class = 3 
        self.net = lraspp_mobilenet_v3_large(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.net(x)
        return x

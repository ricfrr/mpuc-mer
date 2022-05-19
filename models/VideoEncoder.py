import torch 
import torch.nn as nn
from torchvision import models
import torch
# Choose the `x3d_s` model 

class VideoModel(nn.Module):
    def __init__(self, num_channels=3, model_name=None):
    
        super(VideoModel, self).__init__()
        if model_name is not None:
            model_name = 'x3d_s'
            self.model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
            self.model.blocks[5].activation = nn.Identity() # .ModuleList
            self.model.blocks[5].proj = nn.Linear(in_features=self.model.blocks[5].proj.in_features, out_features=512)
        else:
            self.model = models.video.r2plus1d_18(pretrained=True)
            self.model.fc = nn.Sequential(nn.Dropout(0.1),
                                            nn.Linear(in_features=self.model.fc.in_features, out_features=512))
            if num_channels == 4:
                new_first_layer = nn.Conv3d(in_channels=4,
                                            out_channels=self.model.stem[0].out_channels,
                                            kernel_size=self.model.stem[0].kernel_size,
                                            stride=self.model.stem[0].stride,
                                            padding=self.model.stem[0].padding,
                                            bias=False)
                # copy pre-trained weights for first 3 channels
                new_first_layer.weight.data[:, 0:3] = self.model.stem[0].weight.data
                self.model.stem[0] = new_first_layer
        self.modes = ["clip"]

    def forward(self, x):
        out = self.model(x)
        return out
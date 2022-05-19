import torch 
import pandas as pd 
import numpy as np
from models.attention import Attention_Module

import torch.nn as nn
import torch

class LinearClassifier(nn.Module):
    def __init__(self, num_feat=512, n_modality=4, n_classes=7, attention=False):
        super(LinearClassifier, self).__init__()
        self.attention = attention
        if self.attention:
            self.attention_head = Attention_Module(num_feat*n_modality)
        self.linear1 = torch.nn.Linear(num_feat*n_modality, num_feat)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_feat, n_classes)

    def forward(self, x):
        if self.attention:
            x = self.attention_head(x)
        out = self.linear1(x)
        out_f =  self.relu(out)
        logits =  self.linear2(out_f)
        return logits, out
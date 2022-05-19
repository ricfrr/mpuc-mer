import torch.nn as nn
import torch 
import torch.nn.functional as F

class AudioEncoder(nn.Module):

    def __init__(self, n_mels=128, len_seq = 42, out=8):
        super(AudioEncoder, self).__init__()
        self.c1d_1 =  nn.Conv1d(n_mels,64, 3)
        self.dropout_1 = nn.Dropout(p=0.1) 
        self.maxpool1d = nn.MaxPool1d(2)
        self.c1d_2 =  nn.Conv1d(64,32,3)
        self.dropout_2 = nn.Dropout(p=0.1) 
        self.fc = nn.Linear(32*42,out)
        
    def forward(self, input):
        out = F.relu(self.c1d_1(input.permute(0,2,1))) # [batch, n_mels, len]
        out = self.dropout_1(out)
        out = self.maxpool1d(out)
        out = F.relu(self.c1d_2(out))
        out = self.dropout_2(out)
        return self.fc(out.flatten(1))
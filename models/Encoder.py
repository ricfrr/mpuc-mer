import torchvision
import torch
from models.STGCN import STGCN
from models.TCN import TCN
from models.VideoEncoder import VideoModel
from utils.utils import time_print
import time 

import torch.nn as nn
import yaml
import numpy as np
from models.STGCN import get_normalized_adj
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class Encoder(torch.nn.Module):
    def __init__(self, config_file=None, n_frames=30, cmu_mosei=False,projection_features=False, raw_data=True, prediction_head=False, device="cuda:0"):
        """
        """
        super(Encoder, self).__init__()

        with open(config_file) as f:
            config = yaml.safe_load(f)

        self.cmu_mosei = cmu_mosei
        self.prediction_head = prediction_head
        self.raw_data = raw_data

        num_nodes_ld= 51
        text_size = 300
        visual_size = 35
        acoustic_size = 384
        num_feat_video= 2
        input_sizes = [text_size, visual_size, acoustic_size]

        out_feat = config["model_params"]["out_feat"]
    
        # if true return the preojection features instead of last layer
        self.projection_features = projection_features    
        
        with open(config["model_params"]["adj_matr"], 'rb') as f:
            A = np.load(f)
            if A.sum() != 51**2:
                A = A + np.identity(51)
        self.A_hat_v = torch.Tensor(get_normalized_adj(A)).to(device)        

        self.video = config["dataset"]["video"]
        self.text = config["dataset"]["text"]
        self.audio = config["dataset"]["audio"]
        self.graph = config["dataset"]["graph"]

        if self.graph:
            self.stgcn = STGCN(num_nodes_ld,num_feat_video,n_frames,config["model_params"]["out_feat"], num_classes=out_feat)
            self.proj_ld = torch.nn.Sequential(torch.nn.Linear(out_feat, 1024),torch.nn.ReLU(),torch.nn.Linear(1024, out_feat)) #nn.Linear(256, 128)

        if self.text:
            if self.cmu_mosei:
                self.text_pos_encoder = PositionalEncoding(text_size)
                self.text_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_sizes[0], nhead=5)
                self.trnn1 = torch.nn.TransformerEncoder(self.text_encoder_layer, num_layers=6)
                self.text_linear = torch.nn.Linear(input_sizes[0], out_feat)
                self.proj_text = torch.nn.Sequential(torch.nn.Linear(out_feat, 1024),torch.nn.ReLU(),torch.nn.Linear(1024, out_feat))
            else:
                self.text_pos_encoder = PositionalEncoding(text_size)
                self.text_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_sizes[0], nhead=5)
                self.trnn1 = torch.nn.TransformerEncoder(self.text_encoder_layer, num_layers=6)
                self.text_linear = torch.nn.Linear(input_sizes[0], out_feat)
                self.proj_text = torch.nn.Sequential(torch.nn.Linear(out_feat, 1024),torch.nn.ReLU(),torch.nn.Linear(1024, out_feat))


        if self.video:
            if raw_data:
                # pretrained r2plus1d model
                self.video_encoder =  VideoModel(model_name=None)
            else:
                self.video_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_sizes[1], nhead=5)
                self.video_linear = torch.nn.Linear(input_sizes[1], out_feat)
                self.vrnn1 = torch.nn.TransformerEncoder(self.video_encoder_layer, num_layers=6)
            self.proj_video= torch.nn.Sequential(torch.nn.Linear(out_feat, 1024),torch.nn.ReLU(),torch.nn.Linear(1024, out_feat))
        
        if self.audio:
            if raw_data:
                self.audio_encoder = TCN(in_chan=128, n_blocks=5, n_repeats=2, out_chan=out_feat,cut_out=False) #AudioEncoder(out=out_feat)
            else:
                self.audio_pos_encoder = PositionalEncoding(acoustic_size)
                self.audio_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_sizes[2], nhead=4)
                self.arnn1 = torch.nn.TransformerEncoder(self.audio_encoder_layer, num_layers=6)
                self.audio_linear = torch.nn.Linear(input_sizes[2], out_feat)    
            self.proj_audio = torch.nn.Sequential(torch.nn.Linear(out_feat, 1024),torch.nn.ReLU(),torch.nn.Linear(1024, out_feat))

        if self.prediction_head:
            self.intermodal_predictions = nn.ModuleList([nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()])
            #create a list of linear layers in which in each layer there are the cose
            for mod in self.intermodal_predictions:
                for k in range(3):
                    mod.append(torch.nn.Sequential(torch.nn.Linear(out_feat, 1024),torch.nn.ReLU(),torch.nn.Linear(1024, out_feat)))

        self.dropout = nn.Dropout(p=0.1) 


    def extract_features(self, sequence, rnn1, mask=None):
        feat = rnn1(sequence)
        #print(f"feat.shape {feat.shape}")
        return feat.mean(0) 

    def forward(self, inputs):
        
        #[ld,audio,video, text]
        inputs_ld,inputs_audio, inputs_video, input_text = inputs[0], inputs[1], inputs[2], inputs[3]
        proj_feat = []
        basic_feat = [] #[af,ldf,vf,tf] 

        if self.audio:
            if self.raw_data:
                audio_extr = self.audio_encoder(inputs_audio)
                af = self.dropout(audio_extr)
            else:
                inputs_audio = self.audio_pos_encoder(inputs_audio)
                audio_extr = self.extract_features(inputs_audio.permute(1,0,2),  self.arnn1)
                audio_extr = self.dropout(audio_extr)
                af = self.audio_linear(audio_extr)
                
            af_proj = self.proj_audio(af)
            
            if self.projection_features:
                af = af_proj.clone() 

            proj_feat.append(af_proj)
            basic_feat.append(af)

        if self.graph:
            ldf = self.stgcn(self.A_hat_v, inputs_ld)
            ldf = self.dropout(ldf)
            ldf_proj = self.proj_ld(ldf)
            if self.projection_features:
                ldf =  ldf_proj.clone()

            proj_feat.append(ldf_proj)
            basic_feat.append(ldf)

        if self.video:
            if self.raw_data:
                video_extr = self.video_encoder(inputs_video)

                vf = self.dropout(video_extr)
            else:
                video_extr = self.extract_features(inputs_video.permute(1,0,2),  self.vrnn1)
                vf = self.video_linear(video_extr)
            vf_proj = self.proj_video(vf)
            if self.projection_features:
                vf = vf_proj.clone()  
            
            proj_feat.append(vf_proj)
            basic_feat.append(vf)

        if self.text:
            if self.cmu_mosei:
                text_feat = self.text_pos_encoder(input_text)
                text_extr = self.extract_features(text_feat.permute(1,0,2),  self.trnn1)
                text_extr = self.dropout(text_extr)
                tf = self.text_linear(text_extr)
            else:
                text_feat = self.text_pos_encoder(input_text)
                text_extr = self.extract_features(text_feat.permute(1,0,2),  self.trnn1)
                text_extr = self.dropout(text_extr)
                tf = self.text_linear(text_extr)

            tf_proj = self.proj_text(tf)
            if self.projection_features:
                tf = tf_proj.clone()

            #tf_proj = torch.nn.functional.normalize(tf_proj, dim=-1)

            proj_feat.append(tf_proj)
            basic_feat.append(tf)
            
        if self.prediction_head:
            basic_feat = [af,ldf,vf,tf]
            converted_features = torch.Tensor([]).to(inputs_ld.device)
            for idx, modality in enumerate(self.intermodal_predictions):
                modality_conversion = torch.Tensor([]).to(inputs_ld.device)
                for internal_modality in modality:
                    feat = internal_modality(basic_feat[idx])
                    modality_conversion = torch.cat((modality_conversion, feat.unsqueeze(1)),1)
                converted_features = torch.cat((converted_features, modality_conversion.unsqueeze(1)), 1)

            if self.prediction_head:
                proj_feat.append(converted_features)
            

        
        
        return proj_feat, basic_feat
        

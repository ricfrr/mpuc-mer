from numpy.core.fromnumeric import resize
import torch 
import numpy as np
from torch._C import dtype 
from torch.utils.data import Dataset
import pandas as pd 
from tqdm import tqdm
from random import randint
from natsort import natsorted


import os
import cv2
import librosa
import statistics 
from statistics import mode 
import torchvision

class CMU_MOSEI(Dataset):
    def __init__(self, samples_path=None,n_frames=50,test=False,augment=False,keep_sentiment=False):
        super(CMU_MOSEI, self).__init__()
        
        self.samples_path = samples_path
        # best performance (especially text) when the full text sequence is considered 
        self.n_frames = n_frames
        self.test = test
        self.keep_sentiment = keep_sentiment
        self.data = []
        
        # number of raw frames used for the video pipeline
        # currently I am using 8 since I saw that is a good trade of between performance and training time 
        # it is possible to increase and reduce but it will change the overall performance
        self.n_raw_frames = 8
        self.augment = augment

        # reference to data index and split 
        self.samples = self.load_dataset(samples_path)
        
        # normalization from the pytorch website, is fixed since the model is pretrained on kinetics
        self.normalize = torchvision.transforms.Normalize(mean = [0.43216, 0.394666, 0.37645],std = [0.22803, 0.22145, 0.216989])
        self.resize_image = torchvision.transforms.Compose([torchvision.transforms.Resize((112,112))])
        # resize the image, resize_to_stack is useful to have the image that are the same dimension along the video
        self.resize_to_stack = torchvision.transforms.Compose([torchvision.transforms.Resize((180,180))])
        self.crop = torchvision.transforms.RandomCrop((112,112), pad_if_needed=True)
    
    def __len__(self):
        return len(self.samples["Labels"])

    def zero_ptp(self, a):
        return 1 if np.ptp(a) == 0 else np.ptp(a)
    
    def get_class_sample_count(self):
        
        count = np.zeros(7)
        labels = self.samples["Labels"]
        #labels = torch.from_numpy(np.array(self.samples["Labels"][index], dtype= np.int8))

        for l in labels:
            l =  np.array(l)
            count += l

        weight = 1. /count
        sample_weight = []
        for l in labels:
            l =  np.array(l)
            s_w =  l * weight 
            s_w =  s_w.sum()
            sample_weight.append(s_w) 

        return count, torch.Tensor(sample_weight)
    

    def load_dataset(self, path_file):
        # the prextracted text features are all with seq len, therefore we keep it also for the other features

        # read the pickle file where are stored labels, text features and landmarks
        sample_data = pd.read_pickle(path_file) 
        path = os.path.split(path_file)[0] # where the other file are stored 

        data = {"Mel_Spect": None, "OpenFace": None, "glove_vectors": None, "Labels" : None, "Video_path" : None}
        
        # remove not useful data from OpenFace_feat, and normalize [0,1] 
        # TENSORS ld selection 
        # start_x, end_x,start_y, end_y = 298-1,365,365,433
        # necessary because the open face feature are much more, like face gaze and so on 
        # but we are interested only on the facial landmark for now
        start_x, end_x,start_y, end_y = 299,367,367,435

        open_filter = []
        glove_text = []
        mel_spectograms = []
        video_frames = []
        labels = []  
        
        for idx,seq in tqdm(enumerate(sample_data)):
            
            seq_open_face = seq["OpenFace"]
            kx = []
            ky = []
            for frame in seq_open_face:
                ld_x = frame[start_x:end_x]
                ld_y = frame[start_y:end_y]
                kx.append(ld_x)
                ky.append(ld_y)
            kx = (kx - np.min(kx))/self.zero_ptp(kx)
            ky = (ky - np.min(ky))/self.zero_ptp(ky)

            try:

                spect_utter = pd.read_pickle(os.path.join(path,"Mel_Spect",seq["uttr_id"]+".pkl"))
                # power to db has show better performance than standard spectogram on RAVDESS 
                spect_utter = librosa.power_to_db(spect_utter, ref=np.max)

                path_frames = natsorted(os.listdir(os.path.join(path,"Frames_resize",seq["uttr_id"])))
                for idx, _ in enumerate(path_frames):
                    path_frames[idx] =  os.path.join(path,"Frames_resize",seq["uttr_id"],path_frames[idx])
                seven_labels = []
                # here the neutral emotion is added, in the unsupervised setting is not usef anymore
                # but with the supervised setting was useful especially in the contrastive 
                if max(seq["label"]) == 0:
                    seven_labels = [1] + list(seq["label"])
                else:
                    seven_labels = [0] + list(seq["label"])

                # filter out the sequence that are very short in terms of frames
                if len(path_frames) > 2:
                    video_frames.append(path_frames)                
                    open_filter.append(np.moveaxis(np.array([kx,ky]),0,2))
                    glove_text.append(seq["text_feat"])
                    mel_spectograms.append(spect_utter)
                    labels.append(seven_labels)
            except:
                continue

        print(len(sample_data))
        print(len(open_filter))

        # labels -> ["neutral","happy","sad","anger","surprise","disgust","fear"]
        data["OpenFace"] = open_filter
        data["glove_vectors"] = glove_text
        data["Mel_Spect"] = mel_spectograms
        data["Video_path"] = video_frames
        data['Labels'] =  labels

        return data

    def load_frames(self,video_path, idxs):
        # loading frames function
        # if we decide to augment self.augment
        # jittering and hflip and random crop are performed over the sequence
        # return the list of frames

        jitter = randint(0,1)
        if jitter and self.augment:
            r1 = 0 
            r2 = 0.4
            jit_params = (r1 - r2) * torch.rand(4) + r2
            color_jitter = torchvision.transforms.ColorJitter(brightness=jit_params[0].item(), contrast=jit_params[1].item(), saturation=jit_params[2].item(), hue=jit_params[3].item())    
        
        v_flip = randint(0,1)
        frames = torch.Tensor([])
        filter_idx = np.linspace(idxs[0], idxs[-1], num=self.n_raw_frames, dtype=int)

        for id in filter_idx:
            img = torchvision.io.read_image(video_path[id])
            img = img.float()
            img = img/255.0
            if not self.augment:
                img = self.resize_image(img)
            else:
                img = self.resize_to_stack(img)

            if jitter and self.augment:
                img = color_jitter(img)
            if v_flip and self.augment:
                img = torchvision.transforms.functional.hflip(img)
            img = self.normalize(img)
            frames = torch.cat((frames,img.unsqueeze(0)),0)

        if self.augment:
            frames = self.crop(frames)

        # (3 x T x H x W)
        if len(frames.shape) == 4: 
            frames = frames.permute(1,0,2,3)
        else:
            frames = torch.zeros(3,self.n_raw_frames,112,112)

        if frames.shape[1] < self.n_raw_frames:
            pad =   self.n_frames - frames.shape[1] + 1
            frames = torch.nn.functional.pad(frames, (0,0,0,0,0,pad), "constant")
        
        return frames

    
    def __getitem__(self, index: int):

        audio_feat = torch.from_numpy(self.samples["Mel_Spect"][index][:-1])
        landmarks = torch.from_numpy(self.samples["OpenFace"][index])
        text =  torch.from_numpy(self.samples["glove_vectors"][index])
        video_path = self.samples["Video_path"][index]

        labels = torch.from_numpy(np.array(self.samples["Labels"][index], dtype= np.int8))

        num_frames = len(video_path)
        
        if num_frames >= self.n_frames:
            start_frame =  randint(0, num_frames-self.n_frames)
        else:
            start_frame = 0
            pad = self.n_frames - num_frames + 1  
            video_path = video_path + [video_path[-1] for _ in range(pad)]  
        
        if self.test:
            start_frame = 0 
        
        if num_frames >  self.n_frames:
            start_frame_2 =  randint(0, num_frames-self.n_frames)
        else:
            start_frame_2 = 0

        start_frame_2 = 0 if self.test else start_frame_2
        end_frame = start_frame+self.n_frames
        end_frame_2 = start_frame_2+self.n_frames

        idx_seq_1 = [k for k in range(start_frame, end_frame)]
        idx_seq_2 = [k for k in range(start_frame_2, end_frame_2)]

        frames_1 = self.load_frames(video_path,idx_seq_1)
        frames_2 = self.load_frames(video_path,idx_seq_2)
        
        #[ld,audio,video, text]
        seq_1  = [landmarks[start_frame:end_frame,17:,: ],audio_feat[start_frame:end_frame, :],frames_1,text[start_frame:end_frame, :]]
        seq_2  = [landmarks[start_frame_2: end_frame_2, 17:,: ],audio_feat[start_frame_2:end_frame_2, :],frames_2,text[start_frame_2:end_frame_2,:]]
        return labels, seq_1, seq_2
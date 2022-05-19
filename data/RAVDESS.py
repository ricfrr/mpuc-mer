import torch
from torch.utils.data import Dataset
import os
import json
from random import randint
from torch import nn
from torch.nn import functional as F
import pandas as pd 
import numpy as np
import librosa
from tqdm import tqdm
from spec_augment import spec_augment_pytorch

def make_dataset(directory, class_to_idx):
    instances = []
    directory = os.path.expanduser(directory)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index
                instances.append(item)
    return instances

class RAVDESS(Dataset):
    def __init__(self, root, samples=None,min_frames=25,n_mels = 128,test=False, augment= False):
        super(RAVDESS, self).__init__()
        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        if samples is None:
            samples = make_dataset(self.root, class_to_idx)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            raise RuntimeError(msg)
        self.classes = classes
        self.augment = augment
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.min_frames = min_frames
        self.test = test
        self.n_mels = n_mels 
        # perform normalization and some preprocessing over the audio, 
        # keep the audio flag to false, if true skip the process and generate random data
        # useful for rapid code debug 
        # in RAVDESS we have only audio and landmark features, TODO implement video features
        self.preprocess_landmark(audio=False)
        self.preprocess_audio(audio=False)

    def __len__(self):
        return len(self.samples)


    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def preprocess_landmark(self,audio=False):
        new_samples = []
        print("preprocessing landmarks")
        # read and normalize the landmarks
        for idx in tqdm(range(len(self.samples))):
            path_ld = self.samples[idx][0]            
            if audio:
                kx = np.zeros((90,68))
                ky = np.zeros((90,68))
            else:        
                data = pd.read_csv(path_ld)
                kx = data.iloc[:,297:365].to_numpy()
                ky = data.iloc[:,365:433].to_numpy()
                kx = (kx - np.min(kx))/np.ptp(kx)
                ky = (ky - np.min(ky))/np.ptp(ky)   
            new_samples.append(([kx,ky],self.samples[idx][1],self.samples[idx][2])) 
                
        self.samples = new_samples


    def preprocess_audio(self,audio=False):
        new_samples = []
        print("preprocessing audio")
        for idx in tqdm(range(len(self.samples))):
            path_audio = self.samples[idx][2]
            with open(path_audio, 'rb') as f:
                mel_spect = np.load(f)
                len_seq = mel_spect.shape[0]
                mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
            
            if audio:
                new_samples.append(([np.zeros((len_seq,68)),np.zeros((len_seq,68))],self.samples[idx][1],torch.Tensor(mel_spect))) 
            else:
                new_samples.append((self.samples[idx][0],self.samples[idx][1],torch.Tensor(mel_spect))) 
        self.samples = new_samples
    
    def get_class_sample_count(self):
        count =np.zeros(len(self.classes), dtype=int)
        for s in self.samples:
            count[s[1]] +=1
        weight = 1. /count
        sample_weight = []
        for s in self.samples:
            sample_weight.append(weight[s[1]]) 
        return count, torch.Tensor(sample_weight)
    
    def rotate(self, out, origin=(0.5, 0.5), degrees=0):
        # not used anymore but useful to perform rotation over the facial landmark 
        out_rot = torch.Tensor([])
        for p in out:
            angle = np.deg2rad(degrees)
            R = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle),  np.cos(angle)]])
            o = np.atleast_2d(origin)
            p = np.atleast_2d(p)
            p = np.squeeze((R @ (p.T-o.T) + o.T).T)
            out_rot = torch.cat((out_rot,torch.Tensor(p).unsqueeze(0)))
        return out_rot

    def __getitem__(self, index: int):
        target = self.samples[index][1]
        kx,ky = self.samples[index][0][0], self.samples[index][0][1] 
        mel_spect = self.samples[index][2]
        
        if not self.test:
            try:
                mel_spect = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spect.T)
                mel_spect =  mel_spect.T
            except:
                pass

        ld = np.array([kx,ky]).T
        ld = torch.Tensor(np.rollaxis(ld, 1, 0))
        
        num_frames = ld.shape[0]
        
        if num_frames < self.min_frames:
            pad = self.min_frames - num_frames
            ld = torch.cat((ld,ld[:pad]), 0)
            mel_spect =torch.cat((mel_spect,mel_spect[:pad]), 0)
        
        if num_frames > self.min_frames:
            start_frame =  randint(0, num_frames-self.min_frames)
        else:
            start_frame = 0
        
        if self.test:
            start_frame = 0 
        
        # [ld,audio,video, text]
        # the [] is for code simplicy because MOSEI has also ld and text 
        if num_frames > self.min_frames:
            start_frame_2 =  randint(0, num_frames-self.min_frames)
        else:
            start_frame_2 = 0

        seq_1 = [ld[start_frame: start_frame+self.min_frames,17:,: ], mel_spect[start_frame: start_frame+self.min_frames,:], [], []]
        seq_2 = [ld[start_frame_2: start_frame_2+self.min_frames,17:,: ], mel_spect[start_frame_2: start_frame_2+self.min_frames,:], [], []]

        return target, seq_1, seq_2
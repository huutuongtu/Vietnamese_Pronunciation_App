# from cmath import acos
import torch
from torch.utils.data import Dataset
import pandas as pd
from char_embedding import tensor_to_text,text_to_tensor
import numpy as np

data = pd.read_csv("/home/tuht/Viet_MDD/embedding_nucleus_train.csv")
sample = data.shape[0]
cols = ['Path', 'Canonical','Transcript']

class MDD_Dataset(Dataset):

    def __init__(self):
        acoustic_canonical = data
        self.n_samples = sample
        A = acoustic_canonical['Path']
        C = acoustic_canonical['Canonical']
        B = acoustic_canonical['Transcript'] #output
        

        self.A_data = A 
        self.C_data = C
        self.y_data = B 

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        p = self.A_data[index]
  
        phonetic =  '/home/tuht/Viet_MDD/phonetic/' + p + ".npy"
        phonetic = np.load(phonetic)
        phonetic = torch.tensor(phonetic)
        acoustic = '/home/tuht/Viet_MDD/filterbank/' + p + ".npy"
        acoustic = np.load(acoustic)
        acoustic = acoustic.T
        acoustic = torch.tensor(acoustic)
        pitch = '/home/tuht/Viet_MDD/kaldi_pitch/' + p + ".npy"
        pitch = np.load(pitch)
        pitch = pitch.T
        pitch = torch.tensor(pitch)
        linguistic = text_to_tensor(self.C_data[index])
        linguistic = torch.tensor(linguistic)


        label = text_to_tensor(self.y_data[index])
        label = torch.tensor(label)
        return acoustic, phonetic, linguistic, pitch, label

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples



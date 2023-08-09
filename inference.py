from jiwer import cer,wer
import torch
import torch.nn as nn
from model import Acoustic_Phonetic_Linguistic
import numpy as np
from torch.utils.data import DataLoader      
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder
from model import Acoustic_Phonetic_Linguistic
from char_embedding import tensor_to_text,text_to_tensor
import pandas as pd
import glob
import time
import os
import kenlm

device = 'cuda'

# vocab_length = 46
data = pd.read_csv("/home/tuht/Viet_MDD/embedding_nucleus_test.csv")
# f = open("./cer.txt", 'a')
fw = open("./PER.txt", 'a')

net = Acoustic_Phonetic_Linguistic()

# labels = ["tS", "_5a", "h", "x", "m", "7", "n", "_5b", "o", "f", "kp", "a_X", "e", "r", "i", "t_h", "w", "_3", "_1", "l", "O", "v", "_2", "p", "M", "u", "_6a", "M7", "z", "S", "uo", "_6b", "7_X", "t", "ts_", "d", "_4", "a", "E_X", "b", "s", "j", "E", "k", "N", "J", "O_X", "G", "wp", "ie", "Nm", "dZ", " "]
# labels = ['_6a', 'M_2', 'e_4', 'O_X_6b', 'M_5b', 'n_1', 'm', 'N', 'Nm_6a', '7_X_2', 'E_X_6a', 'O_6a', 'a_X_3', '7_5a', 'o_6b', 'ie_6b', 'e_6b', 'a_1', 'M7_5a', 'i_5a', 't_h', 'w_6a', 'Nm_1', 'O_X_5a', 'wp_2', 'N_4', 'w_1', 'p_5b', 'm_2', 'M7_3', '7_X_3', 'J_5a', 'O_5a', '7_6b', 'u_6a', 'uo_3', 'm_5a', '_6b', 'u_1', 'o_2', 'j_6a', 'a_5a', 'uo_5a', 'o_1', 'E_6b', 'E_X_6b', 'O_3', 'kp_6b', '7_X_4', 'uo_4', 'M_3', 'b', 'm_1', 'n_6a', 'ie_6a', '_4', 'o_5b', 'wp_6b', 'a_X_6a', '7_6a', 'n_5a', 'uo_5b', '7_1', 'm_4', 'J_2', 's', 'ie_3', 'wp_5a', 'O_X_1', 'G', 'u_3', 'N_2', 'O_X_6a', 'i_2', 'ie_5a', 'N_1', 'a_2', 'w_2', 'O_X_4', 'O_1', 'p', 'Nm_5a', '7_X_6b', '_5a', 'M_6b', 'M7_4', 'e_3', 'a_4', 'M_1', 'Nm_2', 'wp_4', 'j_3', 'dZ', 'ie_4', 'v', 'E_5a', 'k_5b', 'uo_6a', 'o_6a', 'ie_5b', 'J', 'M7_2', 'i_4', '7_5b', 'x', 'm_3', 'O_6b', 'E_X_5a', 'J_4', '_2', 'u_5a', 'E_X_2', 'N_3', 'J_6a', 'N_5a', 'O_X_5b', 'wp_6a', 'M7_5b', 'd', 't', 'tS', 'u_5b', 'm_6a', 'E_X_4', 'n_4', 'a_X_1', 'j_2', 'E_5b', 'E_X_3', 'k', '7_X_5a', 't_5b', '7_X_5b', 'a_5b', 'O_X_2', 'E_6a', 'E_4', 'w_5a', 'N_6a', 'w_4', 'uo_1', 'ie_2', 'i_5b', 'E_3', 'i_6a', 'u_2', 'a_X_5b', 'ie_1', 'o_5a', '_1', '7_4', 'a_6b', 'M_6a', 'wp_3', '7_X_6a', 'p_6b', 'wp_5b', 'Nm_3', 'o_4', 'r', 'i_3', 'j_5a', 'k_6b', '7_3', 'n_3', 'j_4', 'M7_6a', 'O_2', 'J_3', 'o_3', 'u_4', 'z', 'j_1', '7_2', 'kp_5b', 'O_4', 'wp_1', 'J_1', 'a_X_4', 'ts_', 'a_3', 'M7_6b', 'O_5b', '_5b', 'l', 'uo_6b', 't_6b', 'a_X_5a', 'a_X_2', 'M_5a', 'u_6b', 'e_5b', '_3', 'E_1', 'Nm_4', 'h', 'a_6a', 'e_1', 'S', 'i_6b', 'w_3', 'E_X_1', 'a_X_6b', 'e_5a', 'e_6a', '7_X_1', 'E_2', 'n_2', 'f', 'E_X_5b', 'n', 'e_2', 'uo_2', 'O_X_3', 'M7_1', 'i_1', 'M_4', ' ']
# labels = ['a_X', '_1', 't_h', 'x', 'h', 'z', '_5b', 'e', 'M7', 'tS', 'J', 's', 'k', 'n', 'ie', '_2', '7_X', 'u', 'w', 'f', 'm', 'v', '_3', 'kp', 'o', 'i', 'd', 'N', 't', 'wp', '_6b', 'p', 'S', 'E_X', 'M', 'r', '_4', 'dZ', 'G', '_6a', 'O', 'uo', 'ts_', 'O_X', 'b', 'a', 'j', 'l', '_5a', 'Nm', 'E', '7', '|', ' ']

#embedding nucleus
labels = ['u_2', 'uo_3', 'E_X_6a', 'ts_', 'O_X_2', 'ie_2', 'ie_1', 'o_3', 'dZ', 'e_4', 'E_X_3', 'ie_6a', 'o_4', 'M7_5a', 't_h', 'uo_5a', '7_X_5a', 'O_X_5b', 'kp', 'a_X_3', 'M_3', 'O_1', 'h', 'M_5b', 'a_5b', 'i_5b', 'O_5a', 'O_X_6b', 'u_5a', '7_2', 'a_3', 'o_6a', 'a_X_2', 'i_5a', 'j', 'e_2', 'k', 'M_2', '7_X_4', '7_X_1', '7_6a', 'a_X_6b', 'O_X_5a', 'M_5a', 'd', 'b', 'O_4', 'E_X_4', 'z', 's', 'u_1', 'M7_3', 'e_6a', 'O_3', 'E_X_2', '7_X_6b', 'uo_2', 'u_3', 'a_X_6a', 'o_5a', 'a_X_1', 'o_1', 'a_5a', 'o_5b', 'E_4', 'o_2', 'a_6a', 'i_1', 'O_X_4', 'e_5b', '7_5b', 'E_5b', '7_X_2', 'uo_4', 'ie_5b', 'M7_2', '7_4', 'N', 'f', 'a_2', 'e_1', 't', 'e_5a', 'tS', 'M7_4', 'E_X_5a', 'u_5b', 'S', 'm', 'w', 'r', 'a_X_4', 'uo_6b', 'a_X_5a', '7_X_6a', 'ie_6b', 'E_6a', 'G', 'uo_5b', '7_3', 'e_6b', 'M7_6b', 'i_6b', 'O_X_6a', '7_1', 'v', 'M_6a', 'J', 'wp', 'M_4', 'ie_3', 'a_X_5b', 'M7_6a', '7_6b', 'uo_1', 'u_6b', 'ie_5a', '7_5a', 'E_1', 'E_X_1', 'o_6b', 'Nm', 'E_6b', 'u_6a', 'a_4', 'u_4', 'a_6b', 'E_2', '7_X_5b', 'a_1', 'p', 'M7_5b', 'i_2', 'O_X_1', 'O_2', 'M_6b', 'E_X_6b', 'x', 'E_3', 'E_X_5b', 'M7_1', '|', '7_X_3', 'O_X_3', 'M_1', 'ie_4', 'O_6a', 'l', 'O_6b', 'uo_6a', 'i_6a', 'E_5a', 'i_3', 'i_4', 'O_5b', 'e_3', 'n', ' ']

alpha = 0.7
beta = 3.0


decoder = build_ctcdecoder(
    labels = labels,
    kenlm_model_path = '/home/tuht/train_kenlm/embedding_nucleus.binary'
)

PATH = []
TRANSCRIPT = []
CANONICAL = []
PREDICT = []
ckp = ['/home/tuht/PAPL_MHA_KALDI/MDD_Checkpoint/MHA_KALDI_embedding_nucleus.pth']
for i in range(len(ckp)):
    print(ckp[i])
    # f.write(str(ckp[i]) + "\n")
    fw.write(str(ckp[i]) + "\n")
    net = torch.load(ckp[i])
    net.eval().to(device)
    charerrorrate = []
    worderrorrate = []
    for i in range(len(data)):
    # for i in range(1):
        print(i)
        path = data['Path'][i]
        can = data['Canonical'][i]
        transcript = data['Transcript'][i]
        p = path
        phonetic =  '/home/tuht/Viet_MDD/phonetic/' + p + ".npy"
        phonetic = np.load(phonetic)
        phonetic = torch.tensor(phonetic)
        acoustic = '/home/tuht/Viet_MDD/filterbank/' + p + ".npy"
        acoustic = np.load(acoustic)
        acoustic = acoustic.T
        acoustic = torch.tensor(acoustic)
        acoustic = acoustic.to(torch.float).to(device)
        pitch = '/home/tuht/Viet_MDD/kaldi_pitch/' + p + ".npy"
        pitch = np.load(pitch)
        pitch = pitch.T
        pitch = torch.tensor(pitch).to(device)
        pitch = pitch.to(torch.float).to(device)
        linguistic = text_to_tensor(can)
        linguistic = torch.tensor(linguistic).to(device)


        acoustic = acoustic.to(device)
        phonetic = phonetic.to(device)
        linguistic = linguistic.to(device)
        
        pitch = pitch.unsqueeze(0)
        acoustic = acoustic.unsqueeze(0)
        phonetic = phonetic.unsqueeze(0)
        linguistic = linguistic.unsqueeze(0)
        outputs = net(acoustic,phonetic,linguistic, pitch)
        outputs = outputs.to(device)
        x = torch.log_softmax(outputs, dim = 1)

        x = x.detach().cpu().numpy()
        # x = x.squeeze(0)
        ground_truth = (transcript)
        hypothesis = str(decoder.decode(x))
        print("__________")
        print(hypothesis)
        print(ground_truth)
        PATH.append(data['Path'][i])
        CANONICAL.append(data['Canonical'][i])
        TRANSCRIPT.append(data['Transcript'][i])
        PREDICT.append(hypothesis)
        # print(text_to_tensor(hypothesis))
        # print(text_to_tensor(ground_truth))


        # print(type(charerrorrate[0]))
        error = wer(ground_truth, hypothesis)
        worderrorrate.append(error)
    # print(sum(charerrorrate)/len(charerrorrate))
    

    
    fw.write(str(sum(worderrorrate)/len(worderrorrate))+"\n")
test = pd.DataFrame([PATH,CANONICAL,TRANSCRIPT, PREDICT]) #Each list would be added as a row
test = test.transpose() #To Transpose and make each rows as columns
test.columns=['Path','Canonical', 'Transcript', 'Predict'] #Rename the columns
test.to_csv("MHA_KALDI_embedding_nucleus_LM.csv")

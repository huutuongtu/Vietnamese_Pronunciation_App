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
from infer import phonetic_embedding, get_filterbank, get_pitch_kaldi, get_pitch_NCCF, text_to_phoneme, reconstruct_remove_final_embedding_nucleus, remove_final_embedding_nucleus
# from app.util.util import generate_mdd_for_app
from metric import Align, Correct_Rate, Accuracy
# from app.util import generate_mdd_for_app, text_to_phonemes
# import kenlm
import sys


net = Acoustic_Phonetic_Linguistic()

# labels = ["tS", "_5a", "h", "x", "m", "7", "n", "_5b", "o", "f", "kp", "a_X", "e", "r", "i", "t_h", "w", "_3", "_1", "l", "O", "v", "_2", "p", "M", "u", "_6a", "M7", "z", "S", "uo", "_6b", "7_X", "t", "ts_", "d", "_4", "a", "E_X", "b", "s", "j", "E", "k", "N", "J", "O_X", "G", "wp", "ie", "Nm", "dZ", " "]
# labels = ['_6a', 'M_2', 'e_4', 'O_X_6b', 'M_5b', 'n_1', 'm', 'N', 'Nm_6a', '7_X_2', 'E_X_6a', 'O_6a', 'a_X_3', '7_5a', 'o_6b', 'ie_6b', 'e_6b', 'a_1', 'M7_5a', 'i_5a', 't_h', 'w_6a', 'Nm_1', 'O_X_5a', 'wp_2', 'N_4', 'w_1', 'p_5b', 'm_2', 'M7_3', '7_X_3', 'J_5a', 'O_5a', '7_6b', 'u_6a', 'uo_3', 'm_5a', '_6b', 'u_1', 'o_2', 'j_6a', 'a_5a', 'uo_5a', 'o_1', 'E_6b', 'E_X_6b', 'O_3', 'kp_6b', '7_X_4', 'uo_4', 'M_3', 'b', 'm_1', 'n_6a', 'ie_6a', '_4', 'o_5b', 'wp_6b', 'a_X_6a', '7_6a', 'n_5a', 'uo_5b', '7_1', 'm_4', 'J_2', 's', 'ie_3', 'wp_5a', 'O_X_1', 'G', 'u_3', 'N_2', 'O_X_6a', 'i_2', 'ie_5a', 'N_1', 'a_2', 'w_2', 'O_X_4', 'O_1', 'p', 'Nm_5a', '7_X_6b', '_5a', 'M_6b', 'M7_4', 'e_3', 'a_4', 'M_1', 'Nm_2', 'wp_4', 'j_3', 'dZ', 'ie_4', 'v', 'E_5a', 'k_5b', 'uo_6a', 'o_6a', 'ie_5b', 'J', 'M7_2', 'i_4', '7_5b', 'x', 'm_3', 'O_6b', 'E_X_5a', 'J_4', '_2', 'u_5a', 'E_X_2', 'N_3', 'J_6a', 'N_5a', 'O_X_5b', 'wp_6a', 'M7_5b', 'd', 't', 'tS', 'u_5b', 'm_6a', 'E_X_4', 'n_4', 'a_X_1', 'j_2', 'E_5b', 'E_X_3', 'k', '7_X_5a', 't_5b', '7_X_5b', 'a_5b', 'O_X_2', 'E_6a', 'E_4', 'w_5a', 'N_6a', 'w_4', 'uo_1', 'ie_2', 'i_5b', 'E_3', 'i_6a', 'u_2', 'a_X_5b', 'ie_1', 'o_5a', '_1', '7_4', 'a_6b', 'M_6a', 'wp_3', '7_X_6a', 'p_6b', 'wp_5b', 'Nm_3', 'o_4', 'r', 'i_3', 'j_5a', 'k_6b', '7_3', 'n_3', 'j_4', 'M7_6a', 'O_2', 'J_3', 'o_3', 'u_4', 'z', 'j_1', '7_2', 'kp_5b', 'O_4', 'wp_1', 'J_1', 'a_X_4', 'ts_', 'a_3', 'M7_6b', 'O_5b', '_5b', 'l', 'uo_6b', 't_6b', 'a_X_5a', 'a_X_2', 'M_5a', 'u_6b', 'e_5b', '_3', 'E_1', 'Nm_4', 'h', 'a_6a', 'e_1', 'S', 'i_6b', 'w_3', 'E_X_1', 'a_X_6b', 'e_5a', 'e_6a', '7_X_1', 'E_2', 'n_2', 'f', 'E_X_5b', 'n', 'e_2', 'uo_2', 'O_X_3', 'M7_1', 'i_1', 'M_4', ' ']
# labels = ['a_X', '_1', 't_h', 'x', 'h', 'z', '_5b', 'e', 'M7', 'tS', 'J', 's', 'k', 'n', 'ie', '_2', '7_X', 'u', 'w', 'f', 'm', 'v', '_3', 'kp', 'o', 'i', 'd', 'N', 't', 'wp', '_6b', 'p', 'S', 'E_X', 'M', 'r', '_4', 'dZ', 'G', '_6a', 'O', 'uo', 'ts_', 'O_X', 'b', 'a', 'j', 'l', '_5a', 'Nm', 'E', '7', '|', ' ']
labels = ['u_2', 'uo_3', 'E_X_6a', 'ts_', 'O_X_2', 'ie_2', 'ie_1', 'o_3', 'dZ', 'e_4', 'E_X_3', 'ie_6a', 'o_4', 'M7_5a', 't_h', 'uo_5a', '7_X_5a', 'O_X_5b', 'kp', 'a_X_3', 'M_3', 'O_1', 'h', 'M_5b', 'a_5b', 'i_5b', 'O_5a', 'O_X_6b', 'u_5a', '7_2', 'a_3', 'o_6a', 'a_X_2', 'i_5a', 'j', 'e_2', 'k', 'M_2', '7_X_4', '7_X_1', '7_6a', 'a_X_6b', 'O_X_5a', 'M_5a', 'd', 'b', 'O_4', 'E_X_4', 'z', 's', 'u_1', 'M7_3', 'e_6a', 'O_3', 'E_X_2', '7_X_6b', 'uo_2', 'u_3', 'a_X_6a', 'o_5a', 'a_X_1', 'o_1', 'a_5a', 'o_5b', 'E_4', 'o_2', 'a_6a', 'i_1', 'O_X_4', 'e_5b', '7_5b', 'E_5b', '7_X_2', 'uo_4', 'ie_5b', 'M7_2', '7_4', 'N', 'f', 'a_2', 'e_1', 't', 'e_5a', 'tS', 'M7_4', 'E_X_5a', 'u_5b', 'S', 'm', 'w', 'r', 'a_X_4', 'uo_6b', 'a_X_5a', '7_X_6a', 'ie_6b', 'E_6a', 'G', 'uo_5b', '7_3', 'e_6b', 'M7_6b', 'i_6b', 'O_X_6a', '7_1', 'v', 'M_6a', 'J', 'wp', 'M_4', 'ie_3', 'a_X_5b', 'M7_6a', '7_6b', 'uo_1', 'u_6b', 'ie_5a', '7_5a', 'E_1', 'E_X_1', 'o_6b', 'Nm', 'E_6b', 'u_6a', 'a_4', 'u_4', 'a_6b', 'E_2', '7_X_5b', 'a_1', 'p', 'M7_5b', 'i_2', 'O_X_1', 'O_2', 'M_6b', 'E_X_6b', 'x', 'E_3', 'E_X_5b', 'M7_1', '|', '7_X_3', 'O_X_3', 'M_1', 'ie_4', 'O_6a', 'l', 'O_6b', 'uo_6a', 'i_6a', 'E_5a', 'i_3', 'i_4', 'O_5b', 'e_3', 'n', ' ']

alpha = 0.7
beta = 3.0

decoder = build_ctcdecoder(
    labels = labels,
    kenlm_model_path = 'embedding_nucleus.arpa'
 
)

ckp = 'MDD_Checkpoint/MHA_KALDI_embedding_nucleus.pth'
net = torch.load(ckp, map_location=torch.device('cpu')).eval()
device = 'cpu'
"""
def inference_for_app(path, canonical):
    # def inference(path, canonical):
    text = canonical
    phonetic = torch.tensor(phonetic_embedding(path)).to(torch.float).to(device)
    acoustic = torch.tensor(get_filterbank(path).T).to(torch.float).to(device)
    pitch = torch.tensor(get_pitch_NCCF(path).T).to(torch.float).to(device)
    canonical, raw_sequence = text_to_phoneme(canonical = canonical)
    y = canonical
    linguistic = torch.tensor(text_to_tensor(canonical)).to(torch.float).to(device)
    pitch = pitch.unsqueeze(0)
    acoustic = acoustic.unsqueeze(0)
    phonetic = phonetic
    linguistic = linguistic.unsqueeze(0)
    if pitch.shape[1]!=phonetic.shape[1]:
        pitch = pitch[:, :phonetic.shape[1], :]
    if acoustic.shape[1]!=phonetic.shape[1]:
        acoustic = acoustic[:, :phonetic.shape[1], :]
    outputs = net(acoustic,phonetic,linguistic, pitch)
    outputs = outputs.to(device)
    x = torch.log_softmax(outputs, dim = 1)
    x = x.detach().cpu().numpy()
    # ground_truth = (transcript)
    hypothesis = str(decoder.decode(x))
    print("__________")
    # print(hypothesis)
    # print(ground_truth)
    # error = wer(ground_truth, hypothesis)
    # canonical = remove_final_embedding_nucleus(canonical)
    hypothesis = reconstruct_remove_final_embedding_nucleus(hypothesis)
    # print(hypothesis)
    canonical = reconstruct_remove_final_embedding_nucleus(y)
    # print(y)
    # print(text)
    _, word_phoneme_in = text_to_phonemes(text)
    return x, canonical, word_phoneme_in
"""

def create_random_masking(tensor, percentage, fill_value=158):
  """Creates a random masking of the given tensor, with the specified percentage of values masked.

  Args:
    tensor: A torch.Tensor of any shape.
    percentage: The percentage of values to mask.
    fill_value: The value to fill in the masked values.

  Returns:
    A new torch.Tensor with the specified percentage of values masked.
  """

  mask = torch.rand(tensor.shape) < percentage / 100
#   new_tensor = torch.mul(tensor, mask)
  tensor[~mask] = fill_value

  return tensor



#Bản gốc train và inference không masking, tuy nhiên lên app lúc infer masking để mô hình bớt phụ thuộc canonical

def inference(path, canonical):
    text = canonical
    phonetic = torch.tensor(phonetic_embedding(path)).to(torch.float).to(device)
    acoustic = torch.tensor(get_filterbank(path).T).to(torch.float).to(device)
    pitch = torch.tensor(get_pitch_NCCF(path).T).to(torch.float).to(device)
    canonical, raw_sequence = text_to_phoneme(canonical = canonical)
    y = canonical
    linguistic = torch.tensor(text_to_tensor(canonical)).to(torch.float).to(device)
    linguistic = create_random_masking(linguistic, 80, 158)
    # tensor_masking_rand = torch.rand((3, 5)) < 0.7
    # linguistic[tensor_masking_rand] = 158
    print(linguistic)
    pitch = pitch.unsqueeze(0)
    acoustic = acoustic.unsqueeze(0)
    phonetic = phonetic
    linguistic = linguistic.unsqueeze(0)
    if pitch.shape[1]!=phonetic.shape[1]:
        pitch = pitch[:, :phonetic.shape[1], :]
    if acoustic.shape[1]!=phonetic.shape[1]:
        acoustic = acoustic[:, :phonetic.shape[1], :]
    outputs = net(acoustic,phonetic,linguistic, pitch)
    outputs = outputs.to(device)
    x = torch.log_softmax(outputs, dim = 1)
    x = x.detach().cpu().numpy()
    # ground_truth = (transcript)
    hypothesis = str(decoder.decode(x))
    print("__________")
    hypothesis = reconstruct_remove_final_embedding_nucleus(hypothesis)
    print(hypothesis)
    return hypothesis, y, text


inference('demo_audio/booibonghi_booibonghi.wav', "bà ơi bà nghỉ")
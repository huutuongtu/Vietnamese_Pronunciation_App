# import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time
import os
import librosa
import torch.nn as nn
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time
import os
import librosa
import torch.nn as nn
# from tkinter import N
import librosa
import pandas as pd
import numpy as np
import numpy as np
from python_speech_features import fbank
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import torchaudio
import torch
import torchaudio.functional as F
# import Align

tokenizer = Wav2Vec2Processor.from_pretrained("../PAPL_MHA_KALDI/pretrained_finetuned")
model = Wav2Vec2ForCTC.from_pretrained("../PAPL_MHA_KALDI/pretrained_finetuned").eval()
newmodel = torch.nn.Sequential(*(list(model.children())[:-2])).to('cpu').eval()


def phonetic_embedding(wav_dir):
    link = wav_dir
    y, sr = librosa.load(link, sr=16000)
    y_16k = librosa.resample(y, sr, 16000)
    audio_input = librosa.to_mono(y_16k)
    input_values = tokenizer(audio_input, return_tensors="pt", sampling_rate=16000).input_values
    return ((newmodel(input_values).last_hidden_state))


def get_pitch_kaldi(path):
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(path)
    t = librosa.get_duration(filename = path)
    pitch_feature = F.compute_kaldi_pitch(SPEECH_WAVEFORM, 16000, frame_length = 0.02, frame_shift = 20)
    pitch, nfcc = pitch_feature[..., 0], pitch_feature[..., 1]
    data = pitch.cpu().detach().numpy()
    return data


def get_pitch_NCCF(path):
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(path)
    pitch = F.detect_pitch_frequency(SPEECH_WAVEFORM, 16000, frame_time=0.02, win_length=3)
    data = pitch.cpu().detach().numpy()
    return data


def get_filterbank(path):
    (rate,sig) = wav.read(path)
    filter, energy = fbank(sig,rate, winlen=0.02, winstep = 0.02, nfilt=80)
    filter = filter.reshape(80, -1)
    energy = energy.reshape(1,-1)
    data = np.concatenate((filter,energy))
    return data


f = open("../PAPL_MHA_KALDI/vi_SG_lexicon.dict", "r", encoding='utf8')
data = f.readlines()
init = []
PHONEME = []
WORD = []
for i in range(len(data)):
    word = data[i].split("|")[0]
    WORD.append(word)
    phoneme = (data[i].split("|")[1].split("\n")[0])
    init.append(phoneme.split(" ")[0])
    PHONEME.append(phoneme)

def text_to_phoneme(canonical):
    canonical = canonical.lower()
    res = ''
    seq = canonical.split(" ")
    for text in seq:
        res = res + PHONEME[WORD.index(text)] + " "
    return res.strip(), canonical




NUCLEAR = ['a', 'E', 'e', 'i', 'O', 'o', '7', 'u', 'M', 'a_X', '7_X', 'E_X', 'O_X', 'ie', 'uo', 'M7']
tone = ['_1', '_2', '_3', '_4', '_5a', '_5b', '_6a', '_6b']
EMBEDDING_NUCLEAR = []
RAW = []

for nucl in NUCLEAR:
    for tonal in tone:
        RAW.append(nucl + " " + tonal)
        EMBEDDING_NUCLEAR.append(nucl + tonal)

def reconstruct_remove_final_add_nucleous(phoneme):
    res = ''
    phoneme = phoneme.split("|")
    for word in phoneme:
        t = ''
        word = word.split(" ")
        for char in word:
            if char in tone:
                t = char + " "
            else:
                res = res + char + " "
        res = res + t
    return res.strip().replace("  ", " ").replace("  ", " ").replace("  ", " ").replace("  ", " ")

def reconstruct_remove_final_embedding_nucleus(phoneme):
    remove_final_add_nucleous = ''
    phoneme = phoneme.split("|")
    for word in phoneme:
        word_split = word.split(" ")
        for char in word_split:
            if char in EMBEDDING_NUCLEAR:
                # print(char)
                id = EMBEDDING_NUCLEAR.index(char)
                word = word.replace(EMBEDDING_NUCLEAR[id], RAW[id])
        remove_final_add_nucleous = remove_final_add_nucleous + word + "|"
    return reconstruct_remove_final_add_nucleous(remove_final_add_nucleous)


def remove_final_embedding_nucleus(canonical):
    res = ''
    tone = ["_1", "_2", "_3", "_4", "_5a", "_5b", "_6a", "_6b"]
    phoneme = canonical.split(" ")
    arr = []
    j = 0
    cnt = 0
    for i in phoneme:
        if i in tone:
            arr.append("")
    for i in phoneme:
        if i not in tone:
            arr[j] = arr[j] + " " + i
        else:
            arr[j] = arr[j] + " " + i
            arr[j] = arr[j].strip()
            j = j + 1
    for word in arr:
        tmp = ''
        word = word.split(" ")
        for char in word:
            if char not in NUCLEAR and char not in tone:
                tmp = tmp + char + " "
            elif char not in tone:
                tmp = tmp + char + word[-1] + " "
        arr[cnt] = tmp + "|" + " "
        cnt = cnt + 1
    for word in arr:
        res = res + word
    return res.strip()

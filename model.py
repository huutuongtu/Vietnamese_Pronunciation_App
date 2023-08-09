import torch.nn as nn
import torch
from linguistic_encoder import Linguistic_encoder
from acoustic_encoder import Acoustic_encoder
from phonetic_encoder import Phonetic_encoder  
from pitch_encoder import Pitch_encoder     
from char_embedding import tensor_to_text,text_to_tensor


class Acoustic_Phonetic_Linguistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.Acoustic_encoder = Acoustic_encoder()
        self.Phonetic_encoder = Phonetic_encoder()
        self.Linguistic_encoder = Linguistic_encoder()
        self.Pitch_encoder = Pitch_encoder()
        self.text_to_tensor = text_to_tensor
        self.tensor_to_text = tensor_to_text
        self.fc1 = nn.Linear(4608,159, bias = True)     
        self.multihead_attn = nn.MultiheadAttention(2304, 16, batch_first=True)
        


    def forward(self, acoustic, phonetic, linguistic, pitch):
  
        phonetic = self.Phonetic_encoder(phonetic) #batch x time x 768
        acoustic = self.Acoustic_encoder(acoustic) #batch x time x 768
        pitch = self.Pitch_encoder(pitch)
        linguistic = self.Linguistic_encoder(linguistic) # shape [0]: 2304 x len(canon)
        Hv = linguistic[0] 
        Hk = linguistic[1] 
        acoustic = acoustic.squeeze(0).squeeze(0)
        acoustic = torch.t(acoustic)
        acoustic = acoustic.unsqueeze(0)
        pitch = pitch.squeeze(0).squeeze(0)
        pitch = torch.t(pitch)
        pitch = pitch.unsqueeze(0)
        Hq = (torch.cat((acoustic,phonetic, pitch),2))
        Hq = Hq
        Hk = Hk.unsqueeze(0)
        Hv = Hv.unsqueeze(0)
        attn_output, attn_output_weights = self.multihead_attn(Hq, Hk, Hv)
        c = attn_output
        before_Linear = torch.cat((c,Hq), 2)
        output = self.fc1(before_Linear)
        return output.squeeze(0)
        
        
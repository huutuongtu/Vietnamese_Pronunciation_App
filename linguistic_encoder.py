import torch.nn as nn
import torch        
from char_embedding import text_to_tensor

class Linguistic_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(256,64)
        self.fc1 = nn.Linear(128, 2304)
        self.bilstm = nn.LSTM(input_size= 64, hidden_size = 64,bidirectional = True)
        self.fc3 = nn.Linear(128,2304)

    def forward(self, x):
        x = torch.t(x)  #(len(canonical) x 1)
        x = torch.tensor(x, dtype=torch.int)
        x = self.embedding(x).squeeze(1)
        o, (h_n, c_n) = self.bilstm(x)
        y = self.fc1(o)
        x = self.fc3(o)
        return x,y
    

# linguistic_encoder = Linguistic_encoder()
# x, y = linguistic_encoder(torch.tensor(text_to_tensor("t 7 h O j k 7_X w 7")))
# print(x.shape)
# print(y.shape)



        
import torch.nn as nn
import torch



class CNN_Stack(nn.Module):
    def __init__(self):
        super().__init__()


        # self.fc = nn.Linear(1024,768)
        self.Conv2d = nn.Conv2d(1,1,3,1,1)
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(768)


    def forward(self, x):
        # x = self.fc(x)
        x = (self.Conv2d(x))
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(0)
        
        x = self.bn(x)
        x = self.reLU(x)
        x = self.drop_out(x)
        x = x.squeeze(0)
        x = torch.t(x)
        x = x.unsqueeze(0)
        return x

class RNN_Stack(nn.Module):
    def __init__(self):
        super().__init__() 
        self.reLU = nn.ReLU()
        self.drop_out = nn.Dropout(p=0.2)
        self.bn = nn.BatchNorm1d(768)
        self.bilstm = nn.LSTM(input_size= 768, hidden_size = 384,bidirectional = True)

    def forward(self, x):
        x = self.bilstm(x)
        x = self.bn(x[0].squeeze(0))
        x = self.drop_out(x)
        
        return x.unsqueeze(0)
        
        

class Phonetic_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = CNN_Stack()
        self.RNN = RNN_Stack()

        


    def forward(self, x):
        x = self.CNN(x)
        x = self.RNN(x)
        return x
        
        
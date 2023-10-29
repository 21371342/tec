from numpy.core.fromnumeric import reshape
import torch
import torch.nn as nn


class Lstm_model(nn.Module):
    def __init__(self,inputsize,outputsize):
        super(Lstm_model,self).__init__()
        self.inputsize = inputsize
        self.lstm1 = nn.LSTM(self.inputsize,32,num_layers = 1,bias = True
                            ,batch_first = True,dropout = 0,bidirectional = True)
        #self.relu1 = nn.ReLU()
        self.lstm2 = nn.LSTM(64,16,num_layers = 1,bias = True
                            ,batch_first = True,dropout = 0,bidirectional = True)
        #self.relu2 = nn.ReLU()
        self.lstm3 = nn.LSTM(32,16,num_layers = 1,bias = True
                             ,batch_first = True,dropout = 0,bidirectional = True)
        self.lstm4 = nn.LSTM(32,8,num_layers = 1,bias = True
                             ,batch_first = True,dropout = 0,bidirectional = True)
        self.linear = nn.Linear(16,outputsize)


    def forward(self,inputs):
        inputs = inputs.float()
        y1,_ = self.lstm1(inputs)
        #y1 = self.relu1(y1)
        y2,_ = self.lstm2(y1)
        y3,_ = self.lstm3(y2)
        y4,_ = self.lstm4(y3)
        y  = self.linear(y4)

        return y[:,0,:] 
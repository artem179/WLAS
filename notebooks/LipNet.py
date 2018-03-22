import torch
import torch.nn as nn
from torch.autograd import Variable

class Flatten(nn.Module):
    def forward(self, x):
        out_x = x.transpose(1, 2)
        out_x = out_x.contiguous()
        dims = out_x.size()
        out_x = out_x.view(dims[0], dims[1], dims[2]*dims[3]*dims[4])
        return out_x

    
class LipNet(nn.Module):
    def __init__(self, hidden_size=256, vocab_size=28, n_layers=1, in_channels=1):
        super(LipNet, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.conv1 = nn.Conv3d(in_channels=self.in_channels, out_channels=32, kernel_size=(3, 5, 5), 
                               stride=(1, 2, 2), padding=(1, 2, 2))
        self.pooling = nn.MaxPool3d((1, 2, 2))
        self.batchnorm1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 5, 5), 
                               stride=(1, 2, 2), padding=(1, 2, 2))
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=96, kernel_size=(3, 3, 3), 
                               stride=(1, 2, 2), padding=(1, 1, 1))
        self.batchnorm3 = nn.BatchNorm3d(96)
        self.flat = Flatten()
        self.gru1 = nn.GRU(input_size=96, hidden_size=hidden_size, num_layers=self.n_layers, 
                           bidirectional=True, batch_first=True)
        self.dense1 = nn.Linear(512, 28)
        self.softmax = nn.Softmax(dim=2)
        
        
    def forward(self, input, hidden):
        output = self.conv1(input)
        output = self.pooling(output)
        output = self.conv2(output)
        output = self.pooling(output)
        output = self.conv3(output)

        output = self.pooling(output)
        output = self.flat(output)
        output, hidden = self.gru1(output, hidden)
        output = self.dense1(output)
        #print(output.size())
        output = self.softmax(output)
        return output, hidden
    
    
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.hidden_size))
    
if __name__ == "__main__":
    ln = LipNet()
    hidden = ln.init_hidden(1)
    a = torch.Tensor(1, 1, 75, 50, 100)
    test = Variable(a)
    print(ln(test_fuck, hidden))
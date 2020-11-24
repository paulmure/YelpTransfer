import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        
        self.rnn = nn.LSTM(emb_dim, hidden_dim, 
                           num_layers=num_layers, bidirectional=True)
    
    def forward(self, x):
        _, (h_n, c_n) = self.rnn(x)
        return h_n, c_n


if __name__ == "__main__":
    x = torch.randn(10, 2, 100)
    encoder = Encoder(100, 50, 3)
    h_n, c_n = encoder(x)
    print("h_n shape:", h_n.shape)
    print("c_n shape:", c_n.shape)

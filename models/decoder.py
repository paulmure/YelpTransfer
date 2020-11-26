import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim):
        super(Decoder, self).__init__()
        
        self.rnn = nn.LSTM(emb_dim, hidden_dim, 
                           num_layers=1)
        self.fc = nn.Linear(hidden_dim, emb_dim)
    
    def forward(self, x, hidden, cell):
        output, (h_n, c_n) = self.rnn(x, (hidden, cell))
        output = self.fc(output)
        output = F.relu(output)
        return output, (h_n, c_n)


if __name__ == "__main__":
    x = torch.randn(1, 2, 200)
    h_n = torch.randn(1, 2, 150)
    c_n = torch.randn(1, 2, 150)
    decoder = Decoder(200, 150)
    output, (h_n, c_n) = decoder(x, h_n, c_n)
    print("output_shape:", output.shape)
    print("h_n shape:", h_n.shape)
    print("c_n shape:", c_n.shape)

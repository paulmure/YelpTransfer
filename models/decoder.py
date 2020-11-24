import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, word_embed_dim):
        super(Decoder, self).__init__()
        
        self.rnn = nn.LSTM(word_embed_dim, hidden_dim, 
                           num_layers=1)
        self.fc = nn.Linear(hidden_dim, word_embed_dim)
    
    def forward(self, x):
        output, (h_n, c_n) = self.rnn(x)
        output = self.fc(output)
        output = F.relu(output)
        return output, (h_n, c_n)


if __name__ == "__main__":
    x = torch.randn(1, 2, 100)
    decoder = Decoder(200, 150, 100)
    output, (h_n, c_n) = decoder(x)
    print("output_shape:", output.shape)
    print("h_n shape:", h_n.shape)
    print("c_n shape:", c_n.shape)

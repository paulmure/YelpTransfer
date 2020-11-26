import torch
import torch.nn as nn
import random

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VQEmbedding


class Model(nn.Module):
    def __init__(self,
                 word_emb_dim,
                 encoder_hidden_dim,
                 encoder_num_layers,
                 quantizer_num_embeddings,
                 decoder_hidden_dim,
                 seq_len,
                 teacher_force):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.teacher_force = teacher_force

        self.encoder = Encoder(word_emb_dim, encoder_hidden_dim, encoder_num_layers)
        self.pre_quantizer_fc = nn.Linear(encoder_hidden_dim * encoder_num_layers * 2, decoder_hidden_dim)
        self.h_n_quantizer = VQEmbedding(quantizer_num_embeddings, decoder_hidden_dim)
        self.c_n_quantizer = VQEmbedding(quantizer_num_embeddings, decoder_hidden_dim)
        self.decoder = Decoder(word_emb_dim, decoder_hidden_dim)
    
    def forward(self, x):
        z_e_h_n, z_e_c_n = self.encoder(x)
        z_e_h_n = z_e_h_n.permute(1, 0, 2)
        z_e_h_n = z_e_h_n.reshape(z_e_h_n.shape[0], -1)

        z_e_c_n = z_e_c_n.permute(1, 0, 2)
        z_e_c_n = z_e_c_n.reshape(z_e_c_n.shape[0], -1)

        z_e_h_n = self.pre_quantizer_fc(z_e_h_n)
        z_e_c_n = self.pre_quantizer_fc(z_e_c_n)

        hidden, z_q_h_n = self.h_n_quantizer.straight_through(z_e_h_n)
        cell, z_q_c_n = self.c_n_quantizer.straight_through(z_e_c_n)

        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)

        outputs = torch.zeros_like(x)
        dec_input = x[0, :, :]
        outputs[0, :, :] = dec_input  # the first token will always either be a <pad> or <sos>

        for t in range(1, self.seq_len):
            output, (hidden, cell) = self.decoder(dec_input.unsqueeze(0), hidden, cell)
            output = output.squeeze(0)
            outputs[t, :, :] = output
            teacher_force = random.random() < self.teacher_force
            dec_input = x[t, :, :] if teacher_force else output
        
        return outputs, z_e_h_n, z_e_c_n, z_q_h_n, z_q_c_n


if __name__ == "__main__":
    x = torch.randn(10, 2, 200)
    model = Model(200, 150, 3, 150, 180, 10, 0.2)

    outputs, z_e_h_n, z_e_c_n, z_q_h_n, z_q_c_n = model(x)
    print("outputs shape:", outputs.shape)
    print("z_e_h_n shape:", z_e_h_n.shape)
    print("z_e_c_n shape:", z_e_c_n.shape)
    print("z_q_h_n shape:", z_q_h_n.shape)
    print("z_q_c_n shape:", z_q_c_n.shape)

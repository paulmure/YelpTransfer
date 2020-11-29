import torch
import torch.optim as optim
import torch.nn.functional as F

import os
import time
import pickle
import numpy as np
from tqdm import tqdm
from math import floor

from models.model import Model
from dataset import YelpDataset

hyper_params = {
    'word_emb_dim': 300,
    'encoder_hidden_dim': 200,
    'encoder_num_layers': 3,
    'quantizer_num_embeddings': 10,
    'decoder_hidden_dim': 220,
    'seq_len': 32,
    'teacher_force': 0.3,
    'commitment_cost': 0.25,
    'epochs': 10,
    'batch_size': 16,
    'learning_rate': 0.001,
    'device': 'cuda',
    'data_path': os.path.join('data', 'yelp_review_data_small')
}


class Trainer():
    def __init__(self,
                 word_emb_dim,
                 encoder_hidden_dim,
                 encoder_num_layers,
                 quantizer_num_embeddings,
                 decoder_hidden_dim,
                 seq_len,
                 teacher_force,
                 commitment_cost,
                 epochs,
                 batch_size,
                 learning_rate,
                 device,
                 data_path):

        self.epochs = epochs
        self.device = torch.device(device)
        self.commitment_cost = commitment_cost

        self.model = Model(word_emb_dim, encoder_hidden_dim, encoder_num_layers, quantizer_num_embeddings,
                           decoder_hidden_dim, seq_len, teacher_force)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, amsgrad=True)
        
        self.model.train()
        self.model.to(device)

        dataset = YelpDataset(data_path)
        dataset = torch.utils.data.Subset(dataset, range(100))
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        self.logs = {f'epoch_{i}': {'loss_recons': [], 'loss_vq': [], 'loss_commit': [], 'total_loss': []} \
                for i in range(1, epochs+1)}

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f'Epoch {epoch}:')
            self.time_epoch_start()

            self.optimizer.zero_grad()
            for i, (stars, word_vecs) in enumerate(tqdm(self.dataloader)):
                word_vecs = word_vecs.to(self.device)
                word_vecs = word_vecs.permute(1, 0, 2)
                outputs, z_e_h_n, z_e_c_n, z_q_h_n, z_q_c_n = self.model(word_vecs)

                loss_recons = F.mse_loss(word_vecs, outputs)

                loss_vq_hn = F.mse_loss(z_q_h_n, z_e_h_n.detach())
                loss_commit_hn = F.mse_loss(z_e_h_n, z_q_h_n.detach())
                loss_vq_cn = F.mse_loss(z_q_c_n, z_e_c_n.detach())
                loss_commit_cn = F.mse_loss(z_e_c_n, z_q_c_n.detach())

                total_loss = loss_recons + loss_vq_hn + loss_vq_cn + \
                    self.commitment_cost * (loss_commit_hn + loss_commit_cn)

                total_loss.backward()

                if i % 10 == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                self.log_training_step(loss_recons, loss_vq_hn + loss_vq_cn, loss_commit_hn + loss_commit_cn, total_loss, epoch)

            self.optimizer.step()

            self.time_epoch_end()
            self.log_epoch(epoch)

        torch.save(self.model.state_dict(), 'test_model')
        self.save_training_log()

    def time_epoch_start(self):
        self.timer = time.time()

    def time_epoch_end(self):
        runtime = time.time() - self.timer
        print(f'Runtime: {(floor(runtime) // 60):02}:{(floor(runtime) % 60):02}')

    def log_training_step(self, loss_recons, loss_vq, loss_commit, total_loss, epoch):
        logger = self.logs[f'epoch_{epoch}']
        logger['loss_recons'].append(loss_recons.detach().cpu())
        logger['loss_vq'].append(loss_vq.detach().cpu())
        logger['loss_commit'].append(loss_commit.detach().cpu())
        logger['total_loss'].append(total_loss.detach().cpu())
    
    def log_epoch(self, epoch):
        logger = self.logs[f'epoch_{epoch}']
        epoch_recon_error = np.mean(logger['loss_recons'])
        epoch_total_loss = np.mean(logger['total_loss'])
        print(f'Recon Error:{epoch_recon_error}, Total Loss: {epoch_total_loss}')

    def save_training_log(self):
        with open('training_log', 'wb') as f:
            pickle.dump(self.logs, f)


if __name__ == "__main__":
    trainer = Trainer(**hyper_params)
    trainer.train()

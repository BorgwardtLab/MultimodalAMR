import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
from torch.autograd import Variable
from modules import ResBlock



class ResBetaVAE(nn.Module):
    def __init__(self, config): #input_size=512, embedding_dim=32, beta=1.0, gamma=1000):
        super(ResBetaVAE, self).__init__()
        self.config = config
        self.embedding_dim = config["embedding_dim"]
        self.input_size = config["fingerprint_size"]
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.num_iter = 0
        self.C_max = Variable(torch.FloatTensor([config["C_max"]]))
        self.C_stop_iter = config["C_stop_iter"]

        p_dropout = config["p_dropout"]
        encoder_blocks = []
        encoder_blocks.append(nn.Linear(self.input_size, config["hidden_dim"]))
        for _ in range(config["n_encoder_hidden_layers"]):
            encoder_blocks.append(ResBlock(config["hidden_dim"], p_dropout))
        encoder_blocks.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_blocks)

        self.fc_mu = nn.Linear(config["hidden_dim"], self.embedding_dim)
        self.fc_logvar = nn.Linear(config["hidden_dim"], self.embedding_dim)

        # decoder
        decoder_blocks = []
        decoder_blocks.append(nn.Linear(self.embedding_dim, config["hidden_dim"]))
        for _ in range(config["n_decoder_hidden_layers"]):
            decoder_blocks.append(ResBlock(config["hidden_dim"], p_dropout))
        decoder_blocks.append(nn.Linear(config["hidden_dim"], self.input_size))
        # decoder_blocks.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_blocks)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        # eps = torch.normal(torch.zeros_like(std), 0.1)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch binary_cross_entropy_with_logits
        # recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean')
        recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='mean', pos_weight=(1-x).sum()/(3*x.sum()))

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_divergence = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return (recon_loss + self.beta * kl_divergence), recon_loss, kl_divergence  # divide total loss by batch size


    def loss_c(self, recon_x, x, mu, logvar, **kwargs):
        self.num_iter += 1
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recon_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        self.C_max = self.C_max.to(x.device)
        C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
        loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        return loss, recons_loss, kld_loss

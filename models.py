from cgitb import enable
import os
import re
import plotly
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from Bio import SeqIO
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from scipy.stats import spearmanr


class deepSequenceSimple(nn.Module):
    def __init__(
        self,
        encoder_arch=[263 * 23, 1500, 1500],
        decoder_arch=[100, 500, 263 * 23],
        n_latent=2,
        activation_function="ReLU",
        enable_bn=True,
    ):

        super(deepSequenceSimple, self).__init__()

        # VGG without Bn as AutoEncoder is hard to train
        self.encoder = Encoder(
            arch=encoder_arch,
            n_latent=n_latent,
            enable_bn=enable_bn,
            activation_function=activation_function,
        )
        self.decoder = Decoder(
            arch=decoder_arch,
            n_latent=n_latent,
            enable_bn=enable_bn,
            activation_function=activation_function,
        )

    def reparameterize(self, z_mu, z_logsigma):
        z_std = torch.exp(z_logsigma)
        eps_dist = torch.distributions.normal.Normal(0, 1)
        return z_mu + z_std * eps_dist.sample(z_mu.shape).cuda()

    def forward(self, inputs):
        x = inputs[0].cuda()
        orig_shape = x.shape
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        z_mu, z_logsigma = self.encoder(x.cuda())
        z = self.reparameterize(z_mu, z_logsigma)

        x_hat = self.decoder(z)
        x_hat = x_hat.view(orig_shape)
        x_hat = F.log_softmax(x_hat, 2)

        x = x.view(orig_shape)

        logpx_z = (x * x_hat).view(batch_size, -1).sum(1)
        return {
            "x_hat": x_hat,
            "logpx_z": logpx_z,
            "z_mu": z_mu,
            "z_logsigma": z_logsigma,
        }


class Encoder(nn.Module):
    def __init__(
        self,
        arch=[263 * 23, 1500, 1500],
        n_latent=2,
        enable_bn=True,
        activation_function="relu",
    ):

        super(Encoder, self).__init__()

        exec(f"self.af = nn.{activation_function}")

        layers = []
        for i in range(len(arch[:-1])):
            _in, _out = arch[i], arch[i + 1]
            layers += [nn.Linear(in_features=_in, out_features=_out)]
            if enable_bn:
                layers += [nn.BatchNorm1d(_out)]
            layers += [self.af()]

        self.net = Sequential(*layers)

        self.l_mu = nn.Linear(in_features=arch[-1], out_features=n_latent)
        self.l_logsigma = nn.Linear(in_features=arch[-1], out_features=n_latent)
        print("Initialized Encoder: %s" % self)

    def forward(self, x):
        x = self.net(x)
        z_mu = self.l_mu(x)
        z_logsigma = self.l_logsigma(x)
        return z_mu, z_logsigma


class Decoder(nn.Module):
    def __init__(
        self,
        arch=[100, 500, 263 * 23],
        n_latent=2,
        enable_bn=True,
        activation_function="ReLU",
    ):

        super(Decoder, self).__init__()

        exec(f"self.af = nn.{activation_function}")
        arch = [n_latent] + arch
        self.layers = []
        for i in range(len(arch[:-1])):
            _in, _out = arch[i], arch[i + 1]
            if i != 0:
                self.layers += [self.af()]
            if enable_bn:
                self.layers += [nn.BatchNorm1d(_in)]
            self.layers += [nn.Linear(in_features=_in, out_features=_out)]

        self.net = Sequential(*self.layers)
        self.mu_l = nn.Parameter(torch.Tensor(1))
        nn.init.ones_(self.mu_l)
        self.logsigma_l = nn.Parameter(torch.Tensor(1))
        nn.init.constant_(self.logsigma_l, -5)

        print("Initialized Decoder: %s" % self.net)

    def forward(self, x):
        x = self.net(x)

        return x

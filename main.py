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

from models import deepSequenceSimple
from utils import (
    get_data,
    plot_latent,
    plot_losses,
    train,
    validate,
    vae_loss,
    LossAccumulator,
    aa1_to_index,
    read_experimental_data,
    approximate_log_ratios,
    plot_correlations,
    ds_vae_loss,
)

import torch

torch.cuda.empty_cache()

# FASTA parser requires Biopython

BATCH_SIZE = 100
MAX_NUM_EPOCHS = 100
EARLY_STOPPING = 15
DEBUG_MODE = False
PRINT_FREQ = 1500
LR = 1e-3
PLOT_CORRELATIONS = True
PLOT_LATENT = False


train_dataset, valid_dataset, weights_train, weights_valid, phyla_map = get_data(
    "data/BLAT_ECOLX_1_b0.5_labeled.fasta", calc_weights=True
)
train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
    weights_train, len(weights_train)
)
valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(
    weights_valid, len(weights_valid)
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=valid_sampler
)


model = deepSequenceSimple(enable_bn=False, activation_function="ReLU")

model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  #
loss_function = ds_vae_loss
loss_acc = LossAccumulator(keys=["mse", "kl"])

experimental_data = read_experimental_data("data/BLAT_ECOLX_Ranganathan2015.csv")

losses = []
correlations = []
best_val_loss = np.inf
noImprovementSince = 0
for epoch in range(MAX_NUM_EPOCHS):
    train_losses = train(
        train_loader, model, optimizer, loss_function, loss_acc, epoch, DEBUG_MODE
    )

    valid_losses = validate(
        valid_loader, model, optimizer, loss_function, loss_acc, epoch
    )

    epoch_losses = {"train_" + k: v for k, v in train_losses.items()}
    epoch_losses.update({"valid_" + k: v for k, v in valid_losses.items()})
    epoch_losses.update({"epoch": epoch})
    losses += [epoch_losses]

    if PLOT_LATENT:
        if epoch % 20 == 0:
            plot_latent(model, train_loader, valid_loader, False, epoch, phyla_map)
            print("plotted")

    plot_losses(losses)

    if PLOT_CORRELATIONS:
        if epoch % 5 == 0:
            e, a, c, p = approximate_log_ratios(
                experimental_data, model, loss_function, "cuda"
            )
            correlations += [[epoch, c]]
            plot_correlations(correlations)

    if valid_losses["total"] < best_val_loss:
        noImprovementSince = 0
        best_val_loss = valid_losses["total"]
    else:
        print(f"No Improvement on validation set in {noImprovementSince} epochs")
        noImprovementSince += 1

    if noImprovementSince >= EARLY_STOPPING:
        print(f"No Improvement on validation set in {EARLY_STOPPING} epochs, quitting")

        if PLOT_LATENT:
            plot_latent(model, train_loader, valid_loader, False, epoch)
        break

    if DEBUG_MODE:
        break

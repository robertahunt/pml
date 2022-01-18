from cgitb import enable
import os
import re
import plotly
import itertools
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from Bio import SeqIO
from copy import deepcopy
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
    load_checkpoint,
    plot_latent,
    save_losses,
    train,
    validate,
    vae_loss,
    LossAccumulator,
    aa1_to_index,
    read_experimental_data,
    approximate_log_ratios,
    save_correlations,
    ds_vae_loss,
    load_checkpoint,
    save_checkpoint,
)

import torch

torch.cuda.empty_cache()

# FASTA parser requires Biopython


class Config(object):
    def __init__(
        self,
        name="base",
        batch_size=100,
        max_num_epochs=100,
        early_stopping=15,
        debug_mode=False,
        print_freq=1500,
        lr=5 * 1e-4,
        plot_correlations=True,
        plot_latent=True,
        encoder_arch=[263 * 23, 1500, 1500],
        decoder_arch=[100, 2000, 263 * 23],
        n_latent=2,
        use_weighted_sampler=False,
        batch_norm=False,
        activation_function="ReLU",
    ):
        # set it up so each input to the class is saved as self.{input}
        # ie self.latent

        locs = list(locals().keys())
        self.dict = {}
        for k in locs:
            if k in ["locs", "self", "k"]:
                continue
            exec(f"self.{k}=locals()['{k}']")
            self.dict[k] = locals()[k]


base_config = Config()
bnorm_config = Config(name="bnorm", batch_norm=True)
ls10_config = Config(name="ls10", n_latent=10)
ls50_config = Config(name="ls50", n_latent=50)
ls100_config = Config(name="ls100", n_latent=100)
p = 225
deepnar_config = Config(
    name="deepnar",
    encoder_arch=[263 * 23, p * 8, p * 2, p],
    decoder_arch=[p, p * 2, p * 8, 263 * 23],
)

configs = [
    base_config,
    bnorm_config,
    ls10_config,
    ls50_config,
    ls100_config,
    deepnar_config,
]
n_experiments = 3

experiments = list(itertools.product(configs, list(range(n_experiments))))
exp_no = 0
results = {}
for config, exp_id in experiments:
    print("\n\nStarting exp_no {exp_no}/%s" % len(experiments))
    print("\t with config %s" % config.dict)

    folder = os.path.join("experiments", str(config.name) + "_" + str(exp_no))
    plots_folder = os.path.join(folder, "plots")
    checkpoints_folder = os.path.join(folder, "checkpoints")
    if os.path.exists(folder):
        print("Experiment {folder} already exists, moving to next")
        continue
    else:
        print("Creating Folders...")
        os.mkdir(folder)
        os.mkdir(plots_folder)
        os.mkdir(checkpoints_folder)

    print("Starting Experiment")

    train_dataset, valid_dataset, weights_train, weights_valid, phyla_map = get_data(
        "data/BLAT_ECOLX_1_b0.5_labeled.fasta", calc_weights=True
    )
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights_train, len(weights_train)
    )
    valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights_valid, len(weights_valid)
    )

    if config.use_weighted_sampler:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=train_sampler,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            sampler=valid_sampler,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=config.batch_size
        )

    model = deepSequenceSimple(
        encoder_arch=config.encoder_arch,
        decoder_arch=config.decoder_arch,
        n_latent=config.n_latent,
        enable_bn=config.batch_norm,
        activation_function=config.activation_function,
    )
    model.count_parameters()
    model.save_parameters(folder)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  #
    loss_function = ds_vae_loss
    loss_acc = LossAccumulator(keys=["logpx_z", "kl"])

    experimental_data = read_experimental_data("data/BLAT_ECOLX_Ranganathan2015.csv")

    losses = []
    correlations = []
    best_val_loss = np.inf
    noImprovementSince = 0
    for epoch in range(config.max_num_epochs):
        train_losses = train(
            train_loader,
            model,
            optimizer,
            loss_function,
            loss_acc,
            epoch,
            config.debug_mode,
        )

        valid_losses = validate(
            valid_loader, model, optimizer, loss_function, loss_acc, epoch
        )

        epoch_losses = {"train_" + k: v for k, v in train_losses.items()}
        epoch_losses.update({"valid_" + k: v for k, v in valid_losses.items()})
        epoch_losses.update({"epoch": epoch})
        losses += [epoch_losses]

        if config.plot_latent:
            if epoch % 20 == 0:
                plot_latent(model, train_loader, valid_loader, False, epoch, phyla_map)
                print("plotted")

        save_losses(losses, folder)

        if config.plot_correlations:
            if epoch % 5 == 0:
                e, a, c, p = approximate_log_ratios(
                    experimental_data, model, loss_function, "cuda"
                )
                correlations += [[epoch, c]]
                save_correlations(correlations, folder)

        if valid_losses["total"] < best_val_loss:
            noImprovementSince = 0
            best_val_loss = valid_losses["total"]
            results["losses"] = losses
            results["correlations"] = correlations
            save_checkpoint(
                checkpoints_folder, "best.pth.tar", model, optimizer, results
            )
        else:
            print(f"No Improvement on validation set in {noImprovementSince} epochs")
            noImprovementSince += 1

        if noImprovementSince >= config.early_stopping:
            print(
                f"No Improvement on validation set in {config.early_stopping} epochs, quitting"
            )
            model, optimizer, best_results = load_checkpoint(
                checkpoints_folder, "best.pth.tar", model, optimizer
            )
            if config.plot_latent:
                plot_latent(model, train_loader, valid_loader, True, epoch, phyla_map)
            break

        if config.debug_mode:
            break

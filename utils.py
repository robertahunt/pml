import os
import re
import plotly
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from Bio import SeqIO
from tqdm import tqdm
from cgitb import enable
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential

from scipy.stats import spearmanr

# Mapping from amino acids to integers
aa1_to_index = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
    "Z": 21,
    "-": 22,
}
aa1 = "ACDEFGHIKLMNPQRSTVWYXZ-"

phyla = [
    "Acidobacteria",
    "Actinobacteria",
    "Bacteroidetes",
    "Chloroflexi",
    "Cyanobacteria",
    "Deinococcus-Thermus",
    "Firmicutes",
    "Fusobacteria",
    "Proteobacteria",
    "Other",
]


def plot_latent(
    model, train_loader, valid_loader, best, epoch, phyla_map, plots_folder
):
    zs = []
    for loader in [train_loader, valid_loader]:
        for batch_num, inputs in enumerate(loader):
            x, y = inputs
            outputs = model(inputs)
            z_mu, z_logvar = outputs["z_mu"], outputs["z_logsigma"]
            z = model.reparameterize(z_mu, z_logvar).detach().cpu().numpy()

            if not len(zs):
                zs = z
                ys = y.detach().cpu().numpy()
            else:
                zs = np.append(zs, z, axis=0)
                ys = np.append(ys, y.detach().cpu().numpy(), axis=0)

    if zs.shape[1] != 2:
        tsne = TSNE(n_components=2, perplexity=20)
        zs = tsne.fit_transform(zs)
        print("latent dim not 2, using tsne")

    df = pd.DataFrame(zs, columns=["x0", "x1"])
    df["class"] = ys
    df["class"] = df["class"].map(phyla_map)

    df = df.sort_values("class")
    fig_tsne = px.scatter(
        df,
        x="x0",
        y="x1",
        color="class",
        opacity=0.5,
        hover_data=["class"],
    )
    suffix = "_best" if best == True else ""
    tsne_fp = os.path.join(plots_folder, f"plotly_{epoch}{suffix}.html")

    plotly.offline.plot(fig_tsne, filename=tsne_fp)


def get_data(data_filename, calc_weights=False, weights_similarity_threshold=0.8):
    """Create dataset from FASTA filename"""
    ids = []
    labels = []
    seqs = []
    label_re = re.compile(r"\[([^\]]*)\]")
    for record in SeqIO.parse(data_filename, "fasta"):
        ids.append(record.id)
        seqs.append(
            np.array(
                [aa1_to_index[aa] for aa in str(record.seq).upper().replace(".", "-")]
            )
        )

        label = label_re.search(record.description).group(1)
        # Only use most common classes
        if label not in phyla:
            label = "Other"
        labels.append(label)
    seqs = torch.from_numpy(np.vstack(seqs).astype(np.float32))

    labels = np.array(labels)

    phyla_lookup_table, phyla_idx = np.unique(labels, return_inverse=True)

    weights = None
    if calc_weights is not False:

        # Experiencing memory issues on colab for this code because pytorch doesn't
        # allow one_hot directly to bool. Splitting in two and then merging.
        # one_hot = F.one_hot(seqs.long()).to('cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = F.one_hot(seqs.long()).bool()
        one_hot[seqs > 19] = 0
        flat_one_hot = one_hot.flatten(1)

        weights = []
        weight_batch_size = 1000
        flat_one_hot = flat_one_hot.float()
        for i in range(seqs.size(0) // weight_batch_size + 1):
            x = flat_one_hot[i * weight_batch_size : (i + 1) * weight_batch_size]
            similarities = torch.mm(x, flat_one_hot.T)
            lengths = (
                (seqs[i * weight_batch_size : (i + 1) * weight_batch_size] <= 19)
                .sum(1)
                .unsqueeze(-1)
                .to("cuda" if torch.cuda.is_available() else "cpu")
            )
            w = (
                1.0
                / (similarities.cuda() / lengths)
                .gt(weights_similarity_threshold)
                .sum(1)
                .float()
            )
            weights.append(w)

        weights = torch.cat(weights)
        neff = weights.sum()

    seqs = F.one_hot(seqs.long()).float()

    (
        train_seqs,
        valid_seqs,
        train_phyla,
        valid_phyla,
        weights_train,
        weights_valid,
    ) = train_test_split(
        seqs, torch.from_numpy(phyla_idx), weights, train_size=0.7, stratify=phyla_idx
    )
    phyla_map = {k: v for k, v in zip(phyla_idx, labels)}
    train_dataset = torch.utils.data.TensorDataset(*[train_seqs, train_phyla])
    valid_dataset = torch.utils.data.TensorDataset(*[valid_seqs, valid_phyla])

    return train_dataset, valid_dataset, weights_train, weights_valid, phyla_map


class LossAccumulator(object):
    def __init__(self, keys):
        self.keys = keys
        self.losses = {k: 0 for k in keys}
        self.n = 0
        self.results = []
        pass

    def update(self, losses, n):
        for name, loss in losses:
            self.losses[name] += loss.item() * n
        self.n += n

    def avg(self):
        return {k: v / self.n for k, v in self.losses.items()}

    def avg_all(self):
        return sum([v for k, v in self.avg().items()])

    def reset(self):
        self.losses = {k: 0 for k in self.keys}
        self.n = 0

    def get_result(self):
        loss_results = self.avg()
        loss_results["total"] = self.avg_all()
        loss_results = {k: np.round(v, 3) for k, v in loss_results.items()}
        return loss_results

    def print(self, batch_num, no_batches, epoch, beta=0):
        print(
            f"Epoch {epoch}, [{batch_num}/{no_batches}] %s, beta %s"
            % (self.get_result(), np.round(beta, 3))
        )


def train(loader, model, optimizer, loss_function, loss_acc, epoch, DEBUG_MODE, beta):
    model.train()
    loss_acc.reset()
    for batch_num, inputs in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = loss_function(inputs, outputs, beta=beta)
        loss = sum([x[1] for x in losses])

        batch_size = inputs[0].shape[0]
        loss_acc.update(losses, batch_size)

        if not DEBUG_MODE:
            loss.backward()
            optimizer.step()
        if beta < 1:
            beta += 0.001
    loss_acc.print(batch_num, len(loader), epoch, beta)

    return loss_acc.get_result(), beta


def validate(loader, model, optimizer, loss_function, loss_acc, epoch, beta):
    model.eval()
    loss_acc.reset()
    for batch_num, inputs in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = loss_function(inputs, outputs, beta=beta)

        batch_size = inputs[0].shape[0]
        loss_acc.update(losses, batch_size)

    return loss_acc.get_result()


def vae_loss(inputs, outputs, **kwargs):
    x = inputs[0].cuda().float()
    x_hat, z_mu, z_logsigma, logpx_z = (
        outputs["x_hat"],
        outputs["z_mu"],
        outputs["z_logsigma"],
        outputs["logpx_z"],
    )

    # mse = F.mse_loss(x, x_hat, reduction="sum")
    # logpxz = torch.sum(x * x_hat)
    kl = -0.5 * torch.sum(
        1.0 + 2.0 * z_logsigma - z_mu.pow(2) - torch.exp(2.0 * z_logsigma)
    )
    return [("logpx_z", -logpx_z.mean()), ("kl", kl)]


def ds_vae_loss(inputs, outputs, **kwargs):
    x = inputs[0].cuda().float()
    x_hat, z_mu, z_logsigma, logpx_z = (
        outputs["x_hat"],
        outputs["z_mu"],
        outputs["z_logsigma"],
        outputs["logpx_z"],
    )

    # mse = F.mse_loss(x, x_hat, reduction="sum")
    # logpxz = torch.sum(x * x_hat)
    # kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl = -0.5 * torch.sum(
        1.0 + 2.0 * z_logsigma - z_mu.pow(2) - torch.exp(2.0 * z_logsigma), dim=1
    )
    return [("logpx_z", -logpx_z.mean()), ("kl", kl.mean())]


def iwae_loss(inputs, outputs, beta=1):
    log_pxGz, log_pz, log_qzGx = (
        outputs["log_pxGz"],
        outputs["log_pz"],
        outputs["log_qzGx"],
    )
    w = log_pxGz + (log_pz - log_qzGx) * beta
    loss = -torch.mean(torch.logsumexp(w, 0))

    return [("iwae", loss)]


def save_losses(losses, folder):
    losses = pd.DataFrame(losses)
    losses = losses.set_index("epoch")
    losses.plot()
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    loss_plot_fp = os.path.join(folder, "plots", "losses.png")
    loss_csv_fp = os.path.join(folder, "losses.csv")
    plt.savefig(loss_plot_fp, bbox_inches="tight")
    plt.close()
    losses.to_csv(loss_csv_fp)


def save_correlations(cs, folder):
    cs = pd.DataFrame(cs, columns=["epoch", "correlation"])
    cs = cs.set_index("epoch")
    cs.plot()
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    cs_plot_fp = os.path.join(folder, "plots", "correlations.png")
    cs_csv_fp = os.path.join(folder, "correlations.csv")
    plt.savefig(cs_plot_fp, bbox_inches="tight")
    plt.close()
    cs.to_csv(cs_csv_fp)


def read_experimental_data(filename, measurement_col_name="2500", sequence_offset=0):
    """Read experimental data from csv file, and check that amino acid match those
    in the first sequence of the alignment.

    measurement_col_name specifies which column in the csv file contains the experimental
    observation. In our case, this is the one called 2500.

    sequence_offset is used in case there is an overall offset between the
    indices in the two files.
    """

    measurement_df = pd.read_csv(
        filename, delimiter=",", usecols=["mutant", measurement_col_name]
    )

    zero_index = None

    experimental_data = {}
    for idx, entry in measurement_df.iterrows():
        mutant_from, position, mutant_to = (
            entry["mutant"][:1],
            int(entry["mutant"][1:-1]),
            entry["mutant"][-1:],
        )

        # Use index of first entry as offset (keep track of this in case
        # there are index gaps in experimental data)
        if zero_index is None:
            zero_index = position

        # Corresponding position in our alignment
        seq_position = position - zero_index + sequence_offset

        # Make sure that two two inputs agree on the indices: the
        # amino acids in the first entry of the alignment should be
        # identical to those in the experimental file.
        # assert mutant_from == aa1[wt_sequence[seq_position]]

        if seq_position not in experimental_data:
            experimental_data[seq_position] = {}

        # Check that there is only a single experimental value for mutant
        assert mutant_to not in experimental_data[seq_position]

        experimental_data[seq_position]["pos"] = seq_position
        experimental_data[seq_position]["WT"] = mutant_from
        experimental_data[seq_position][mutant_to] = entry[measurement_col_name]

    experimental_data = (
        pd.DataFrame(experimental_data).transpose().set_index(["pos", "WT"])
    )
    return experimental_data


def approximate_log_ratios(
    experimental_data,
    model,
    loss_function,
    device,
    num_samples=10,
    model_type="vae",
    N_pred_iterations=30,
):

    experimental_values = np.empty((263, 20))
    approximate_vae_values = np.empty((263, 20))

    # compute x_WT
    x_WT = torch.empty(263)
    for (position, mutant_from), _ in experimental_data.iterrows():
        x_WT[position] = aa1_to_index[mutant_from]

    x_WT_ = x_WT.clone()

    # get output for x_WT
    x_WT = x_WT[None, :]
    if model_type == "iwae":
        dim_0, dim_1 = x_WT.size()
        x_WT = x_WT.expand(num_samples, dim_0, dim_1)
    x_WT = x_WT.to(device)
    if model_type == "vae":
        x_WT = F.one_hot(x_WT.long())

        # Add 0s for 3 classes that aren't included in experimental data
        x_WT = F.pad(x_WT, pad=(0, 3)).float()
        WT_elbos = []
        for h in range(N_pred_iterations):
            outputs = model([x_WT])
            losses = loss_function([x_WT], outputs)
            WT_elbos += [sum([x[1] for x in losses]).detach().cpu().numpy()]
        elbo_WT = -np.mean(WT_elbos)
        # elbo_WT, _, _ = model.calculate_loss(x_WT, device=device)
    else:
        elbo_WT = model.calculate_loss(x_WT, beta=1.0, device=device)

    # approximate log ratios
    for (position, mutant_from), row in tqdm(
        experimental_data.iterrows(), total=len(experimental_data)
    ):
        i = 0
        for mutant_to, exp_value in row.iteritems():
            if not np.isnan(exp_value):
                # compute x_MT
                x_MT = x_WT_.clone()

                x_MT[position] = aa1_to_index[mutant_to]
                # get output for x_MT
                x_MT = x_MT[None, :]
                x_MT = F.one_hot(x_MT.long())

                # Add 0s for 3 classes that aren't included in experimental data
                x_MT = F.pad(x_MT, pad=(0, 3)).float()
                if model_type == "iwae":
                    dim_0, dim_1 = x_MT.size()
                    x_MT = x_MT.expand(num_samples, dim_0, dim_1)
                x_MT = x_MT.to(device)
                x_MT = x_MT.repeat(N_pred_iterations, 1, 1)
                if model_type == "vae":
                    outputs = model([x_MT])
                    losses = loss_function([x_MT], outputs)
                    elbo_MT = -sum([x[1] for x in losses]).detach().cpu().numpy()

                else:
                    wtf
                    elbo_MT = model.calculate_loss(x_MT, beta=1.0, device=device)
                # compute the approximate log-ratio
                approx_log_ratio = elbo_MT - elbo_WT

                # store values in numpy arrays
                approximate_vae_values[position, i] = approx_log_ratio
                experimental_values[position, i] = exp_value

            else:
                approximate_vae_values[position, i] = 0.0
                experimental_values[position, i] = np.nan
            i += 1

    # compute the Spearman R statistics
    correlation, pvalue = spearmanr(
        experimental_values.flatten(),
        approximate_vae_values.flatten(),
        nan_policy="omit",
    )
    return experimental_values, approximate_vae_values, correlation, pvalue


def save_checkpoint(folder, fn, model, optimizer, results):
    fp = os.path.join(folder, fn)

    state = {
        "state_dict": model.state_dict(),
        "optimizer_dict": optimizer.state_dict(),
        "results_dict": results,
    }

    torch.save(state, fp)


def load_checkpoint(folder, fn, model, optimizer):
    fp = os.path.join(folder, fn)
    state = torch.load(fp)
    model.load_state_dict(state["state_dict"])
    try:
        optimizer.load_state_dict(state["optimizer_dict"])
    except:
        print("Could not load optimizer state dictionary. ")

    results = state["results_dict"]
    return model, optimizer, results

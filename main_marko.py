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

# FASTA parser requires Biopython

BATCH_SIZE = 16
MAX_NUM_EPOCHS = 200
EARLY_STOPPING = 20
DEBUG_MODE = False
PRINT_FREQ = 1500

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
    # seqs = (seqs - torch.min(seqs, dim=0).values) / (torch.max(seqs, dim=0).values - torch.min(seqs, dim=0).values)
    labels = np.array(labels)

    phyla_lookup_table, phyla_idx = np.unique(labels, return_inverse=True)

    dataset = torch.utils.data.TensorDataset(*[seqs, torch.from_numpy(phyla_idx)])

    weights = None
    if calc_weights is not False:

        # Experiencing memory issues on colab for this code because pytorch doesn't
        # allow one_hot directly to bool. Splitting in two and then merging.
        # one_hot = F.one_hot(seqs.long()).to('cuda' if torch.cuda.is_available() else 'cpu')
        one_hot1 = F.one_hot(seqs[: len(seqs) // 2].long()).bool()
        one_hot2 = F.one_hot(seqs[len(seqs) // 2 :].long()).bool()
        one_hot = torch.cat([one_hot1, one_hot2]).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        assert len(seqs) == len(one_hot)
        del one_hot1
        del one_hot2
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
                / (similarities / lengths)
                .gt(weights_similarity_threshold)
                .sum(1)
                .float()
            )
            weights.append(w)

        weights = torch.cat(weights)
        neff = weights.sum()

    return dataset, weights


dataset, weights = get_data("BLAT_ECOLX_1_b0.5_labeled.fasta", calc_weights=True)
sampler = torch.utils.data.WeightedRandomSampler(
    weights=weights, num_samples=weights.size(dim=0), replacement=True
)
dataloader_sampler = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, sampler=sampler
)
dataloader_no_sampler = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

# device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class deepSequenceSimple(nn.Module):
    def __init__(
        self,
        encoder_arch=[263 * 23, 1500, 1500],
        decoder_arch=[100, 500, 263 * 23],
        n_latent=2,
        activation_function="ReLU",
        gate_function="Sigmoid",
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
            gate_function=gate_function,
        )

    def reparameterize(self, z_mu, z_logvar):
        z_std = torch.exp(0.5 * z_logvar)
        eps_dist = torch.distributions.normal.Normal(0, 1)
        return z_mu + z_std * eps_dist.sample(z_mu.shape).cuda()

    def forward(self, inputs):
        x = inputs[0].cuda()
        orig_shape = x.shape
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)

        z_mu, z_logvar = self.encoder(x.cuda())
        z = self.reparameterize(z_mu, z_logvar)

        x_hat = self.decoder(z)
        x_hat = x_hat.view(orig_shape)
        return x_hat, z_mu, z_logvar


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
        gate_function="Sigmoid",
    ):

        super(Decoder, self).__init__()

        exec(f"self.af = nn.{activation_function}")
        exec(f"self.gate_function = nn.{gate_function}")
        arch = [n_latent] + arch
        layers = []
        for i in range(len(arch[:-1])):
            _in, _out = arch[i], arch[i + 1]
            if i != 0:
                layers += [self.af()]
            if enable_bn:
                layers += [nn.BatchNorm1d(_in)]
            layers += [nn.Linear(in_features=_in, out_features=_out)]

        layers += [self.gate_function()]

        self.net = Sequential(*layers)
        print("Initialized Decoder: %s" % self.net)

    def forward(self, x):
        x = self.net(x)

        return x


# define the model
class VAE(nn.Module):
    def __init__(self, latent_dim=30, alpha=1e-3):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_features=263, out_features=1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1500),
            nn.ReLU(),
        )

        self.hidden_mu = nn.Linear(in_features=1500, out_features=latent_dim)
        self.hidden_log_var = nn.Linear(in_features=1500, out_features=latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=263 * 23),
            nn.Sigmoid(),
        )

        self.alpha = alpha

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden_mu(hidden)
        log_var = self.hidden_log_var(hidden)
        return mu, log_var

    def reparameterize(self, mu, std):
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, zGx):
        xGz = self.decoder(zGx)
        xGz = xGz.view(-1, 263, 23)
        return xGz

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = log_var.mul(0.5).exp_()
        zGx = self.reparameterize(mu, std)
        xGz = self.decode(zGx)
        return xGz

    def calculate_loss(self, x, device):
        # encode inputs
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        # reparameterize
        zGx = self.reparameterize(mu, std)
        xGz = self.decode(zGx)
        # calculate losses
        reconstruction_loss = F.mse_loss(
            xGz, F.one_hot(x.long(), num_classes=23).float(), reduction="sum"
        )
        kl_divergence = (
            -0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)
        ).mean(dim=0)
        vae_loss = self.alpha * reconstruction_loss + kl_divergence
        return vae_loss, self.alpha * reconstruction_loss, kl_divergence


# define the model
class IWAE(nn.Module):
    def __init__(self, latent_dim=30):
        super(IWAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_features=263, out_features=1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1500),
            nn.ReLU(),
        )

        self.hidden_mu = nn.Linear(in_features=1500, out_features=latent_dim)
        self.hidden_log_var = nn.Linear(in_features=1500, out_features=latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=100),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=263 * 23),
            nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mu = self.hidden_mu(hidden)
        log_var = self.hidden_log_var(hidden)
        return mu, log_var

    def reparameterize(self, mu, std):
        qzGx = td.Normal(loc=mu, scale=std)
        zGx = qzGx.rsample()
        return zGx, qzGx

    def decode(self, zGx):
        xGz = self.decoder(zGx)
        dim_0, dim_1, _ = xGz.size()
        xGz = xGz.view(dim_0, dim_1, 263, 23)
        return xGz

    def forward(self, x):
        mu, log_var = self.encode(x)
        std = log_var.mul(0.5).exp_()
        zGx, _ = self.reparameterize(mu, std)
        xGz = self.decode(zGx)
        return xGz

    def calculate_loss(self, x, beta, device):
        # encode inputs
        mu, log_var = self.encode(x)
        std = torch.exp(0.5 * log_var)
        # reparameterize
        zGx, qzGx = self.reparameterize(mu, std)
        # decode
        xGz = self.decode(zGx)
        # calculate q(z|x)
        log_qzGx = qzGx.log_prob(zGx)
        log_qzGx = torch.sum(log_qzGx, dim=-1)
        # calculate p(z)
        mu_prior = torch.zeros(self.latent_dim).to(device)
        std_prior = torch.ones(self.latent_dim).to(device)
        pz = td.Normal(loc=mu_prior, scale=std_prior)
        log_pz = torch.sum(pz.log_prob(zGx), dim=-1)
        # calculate p(x|z)
        pxGz = td.categorical.Categorical(logits=xGz).log_prob(x)
        log_pxGz = torch.sum(pxGz, dim=-1)
        # calculate loss
        w = log_pxGz + (log_pz - log_qzGx) * beta
        loss = -torch.mean(torch.logsumexp(w, 0))
        return loss


# train the model
def train_model(
    model, optimizer, epochs, train_loader, device, num_samples=10, model_type="vae"
):
    print("Training the model!\n")
    beta = 0.0
    for epoch in range(epochs):
        # TRAINING
        print("\nEpoch: {}/{}".format(epoch + 1, epochs))
        train_loss = 0.0
        if model_type == "vae":
            rec_loss, kl_loss = 0.0, 0.0
        for train_data in train_loader:
            train_Xs, _ = train_data
            if model_type == "iwae":
                dim_0, dim_1 = train_Xs.size()
                train_Xs = train_Xs.expand(num_samples, dim_0, dim_1)
            train_Xs = train_Xs.to(device)
            # forward-propagation
            if model_type == "vae":
                loss, rec_loss_, kl_loss_ = model.calculate_loss(
                    train_Xs, device=device
                )
            else:
                loss = model.calculate_loss(train_Xs, beta=beta, device=device)
            # back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if model_type == "vae":
                rec_loss += rec_loss_
                kl_loss += kl_loss_
            if beta < 1.0:
                beta += 0.001
        train_loss /= len(train_loader)
        print("Training Loss: {}".format(train_loss))
        if model_type == "vae":
            rec_loss /= len(train_loader)
            print("Reconstruction Loss: {}".format(rec_loss))
            kl_loss /= len(train_loader)
            print("KL Loss: {}".format(kl_loss))

    print("\nFinished training the model!")


class LossAccumulator(object):
    def __init__(self, keys):
        self.keys = keys
        self.losses = {k: 0 for k in keys}
        self.n = 0
        self.results = []
        pass

    def update(self, losses):
        for name, loss in losses:
            self.losses[name] += loss
        self.n += 1

    def avg(self):
        return {k: v.item() / self.n for k, v in self.losses.items()}

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

    def print(self, batch_num, no_batches, epoch):
        print(f"Epoch {epoch}, [{batch_num}/{no_batches}] %s" % (self.get_result()))


def train(loader, model, optimizer, loss_function, loss_acc, epoch):
    loss_acc.reset()
    for batch_num, inputs in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = loss_function(inputs, outputs)
        loss = sum([x[1] for x in losses])

        loss_acc.update(losses)

        if not DEBUG_MODE:
            loss.backward()
            optimizer.step()
    loss_acc.print(batch_num, len(loader), epoch)

    return loss_acc.get_result()


def validate(loader, model, optimizer, loss_function, loss_acc, epoch):
    loss_acc.reset()
    for batch_num, inputs in enumerate(loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = loss_function(inputs, outputs)

        loss_acc.update(losses)

    return loss_acc.get_result()


def vae_loss(inputs, outputs):
    x = inputs[0].cuda().float()
    x_hat, z_mu, z_logvar = outputs

    mse = F.mse_loss(x, x_hat, reduction="sum")
    kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    return [("mse", mse), ("kl", kl)]


def plot_losses(losses):
    losses = pd.DataFrame(losses)
    losses = losses.set_index("epoch")
    losses.plot()
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    loss_plot_fp = "losses.png"
    plt.savefig(loss_plot_fp, bbox_inches="tight")
    plt.close()


train_dataset, valid_dataset, weights, phyla_map = get_data(
    "data/BLAT_ECOLX_1_b0.5_labeled.fasta", calc_weights=False
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=True
)

model = deepSequenceSimple(enable_bn=False, activation_function="ReLU")
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=[0.5, 0.999])  #
loss_function = vae_loss
loss_acc = LossAccumulator(keys=["mse", "kl"])
losses = []


def plot_latent(model, train_loader, valid_loader, best, epoch):
    zs = []
    for loader in [train_loader, valid_loader]:
        for batch_num, inputs in enumerate(loader):
            x, y = inputs
            x_hat, z_mu, z_logvar = model(inputs)
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
    tsne_fp = os.path.join("plots", f"plotly_{epoch}{suffix}.html")

    plotly.offline.plot(fig_tsne, filename=tsne_fp)


best_val_loss = np.inf
noImprovementSince = 0
for epoch in range(MAX_NUM_EPOCHS):
    train_losses = train(train_loader, model, optimizer, loss_function, loss_acc, epoch)
    valid_losses = validate(
        valid_loader, model, optimizer, loss_function, loss_acc, epoch
    )

    epoch_losses = {"train_" + k: v for k, v in train_losses.items()}
    epoch_losses.update({"valid_" + k: v for k, v in valid_losses.items()})
    epoch_losses.update({"epoch": epoch})
    losses += [epoch_losses]

    if epoch % 20 == 0:
        plot_latent(model, train_loader, valid_loader, False, epoch)
        print("plotted")

    plot_losses(losses)

    if valid_losses["total"] < best_val_loss:
        noImprovementSince = 0
        best_val_loss = valid_losses["total"]
    else:
        print(f"No Improvement on validation set in {noImprovementSince} epochs")
        noImprovementSince += 1

    if noImprovementSince >= EARLY_STOPPING:
        print(f"No Improvement on validation set in {EARLY_STOPPING} epochs, quitting")
        plot_latent(model, train_loader, valid_loader, False, epoch)
        break

    if DEBUG_MODE:
        break

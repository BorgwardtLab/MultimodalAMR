import pandas as pd
import numpy as np
from argparse import ArgumentParser
import os
from os.path import join, exists
import sys
sys.path.insert(0,'../')


from models.res_vae import ResBetaVAE
from models.data_loaders import ChEMBL_Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.uniform_(m.weight, -0.002, 0.02)
        m.bias.data.fill_(0.001)


class BetaVAE_Experiment(pl.LightningModule):
    def __init__(self, config=None, target_sets_h5_filepath=None):
        super().__init__()
        self.config = config

        self.interval = config.get("interval", 100)
        self.warmup_steps = config.get("warmup_steps", 100)
        self.iter = 0
        self.max_beta = config["beta"]
        self.vae = ResBetaVAE(config)
        self.vae.apply(init_weights)

        self.loss_type = config.get("loss_type", "H")

        self.kld_weight = config["embedding_dim"] / config["fingerprint_size"]



    def forward(self, batch):
        return self.vae(batch)

    def _update_beta(self):
        cycle_iter = max(0, (self.iter - self.warmup_steps)) #% (2 * self.interval)
        self.vae.beta = max(0, min(cycle_iter * self.max_beta/self.interval, self.max_beta))


    def training_step(self, batch, idx_batch):
        self.iter += 1
        self._update_beta()
        fingerprints, ids = batch
        rx, mu, logvar = self(fingerprints)
        if self.loss_type=="H":
            loss, recon_loss, kl_divergence = self.vae.loss(rx, fingerprints, mu, logvar)
        elif self.loss_type=="B":
            loss, recon_loss, kl_divergence = self.vae.loss_c(rx, fingerprints, mu, logvar, M_N=self.kld_weight)
        else:
            raise ValueError("Invalid loss choice")
        self.log("beta", self.vae.beta, on_step=True,
                 logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.config["batch_size"])
        self.log("train_reconstruction_loss", recon_loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.config["batch_size"])
        self.log("train_kl_divergence", kl_divergence, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=self.config["batch_size"])
        self.log("train_weighted_kl_divergence", kl_divergence*self.vae.beta, on_step=True, on_epoch=True,
                 prog_bar=False, logger=True, batch_size=self.config["batch_size"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.vae.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        # scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=self.learning_rate, mode="triangular2", step_size_up=10, cycle_momentum=False)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer


def main(args):
    config = vars(args)

    output_folder = join("outputs", "ChEMBL_ResVAE", args.experiment_name)
    if not exists(output_folder):
        os.makedirs(output_folder)

    with open(join(output_folder, "config.yml"), "w") as f:
        yaml.dump(config, f)

    print("Loading fingerprints")
    chembl_fingerprint_df = pd.read_csv("processed_data/chembl_molecular_fingerprints.csv")
    chembl_fingerprint_df = chembl_fingerprint_df[["molecule_chembl_id", config["fingerprint"]]]

    train_ds = ChEMBL_Dataset(chembl_fingerprint_df)
    train_loader = DataLoader(train_ds, shuffle=True, batch_size=config["batch_size"], num_workers=config["num_workers"])
    experiment = BetaVAE_Experiment(config)

    with open(join(output_folder, "architecture.txt"), "w") as f:
        f.write(experiment.vae.__repr__())

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(output_folder, "checkpoints"),
                                          monitor="train_loss", filename="gst-{epoch:02d}-{train_loss:.3f}")
    early_stopping_callback = EarlyStopping(
        monitor="train_loss", mode="min", patience=args.patience
    )
    callbacks = [checkpoint_callback, early_stopping_callback]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=join(output_folder, "logs/"))

    lim_batches = 80000/len(train_ds)
    print(f"Using {lim_batches} training batches")
    print("Training..")
    trainer = pl.Trainer(devices="auto", accelerator="auto", default_root_dir=output_folder, max_epochs=args.n_epochs, callbacks=callbacks,
                         logger=tb_logger, log_every_n_steps=3,
                         limit_train_batches=lim_batches
                         )
    trainer.fit(experiment, train_dataloaders=train_loader)
    best_exp = BetaVAE_Experiment.load_from_checkpoint(checkpoint_callback.best_model_path, config=config)
    torch.save(best_exp.vae.state_dict(), os.path.join(output_folder, "vae_weights.pt"))
    print("Training complete!")

    print("Embedding ChEMBL dataset")
    embed_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=config["num_workers"])
    best_exp.vae.eval()
    embeddings = []
    chem_ids = train_ds.molecule_chembl_id
    for idx, batch in tqdm(enumerate(embed_loader)):
        fingerprints, ids = batch
        mu, logvar = best_exp.vae.encode(fingerprints)
        embeddings.append(mu.detach().numpy())
    embeddings = np.vstack(embeddings)
    chembl_embeddings = pd.DataFrame(data=embeddings, index=chem_ids)
    chembl_embeddings.to_csv(os.path.join(output_folder, "chembl_embeddings.csv"))
    pca_embeddings = PCA(n_components=2).fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.scatter(pca_embeddings[:,0], pca_embeddings[:,1])
    fig.savefig(os.path.join(output_folder, "chembl_embeddings.png"), facecolor="white")
    print("Embedding complete!")




if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="res_vae_logits")
    parser.add_argument("--loss_type", type=str, default="H")


    parser.add_argument("--embedding_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_encoder_hidden_layers", type=int, default=3)
    parser.add_argument("--n_decoder_hidden_layers", type=int, default=2)

    parser.add_argument("--fingerprint", type=str, default="morgan_512_fps")
    parser.add_argument("--fingerprint_size", type=int, default=512)

    parser.add_argument("--beta", type=float, default=1.3)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--interval", type=int, default=100)

    parser.add_argument("--gamma", type=float, default=1000)
    parser.add_argument("--C_max", type=float, default=25)
    parser.add_argument("--C_stop_iter", type=float, default=1e5)

    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--p_dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=3)

    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()


    main(args)
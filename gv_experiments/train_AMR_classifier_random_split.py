import sys
sys.path.insert(0, "..")
# sys.path.insert(0, "../data_split")

import numpy as np
import os
from os.path import join, exists
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from argparse import ArgumentParser
import json
from experiments.pl_experiment import AMR_Classifier_Experiment
from data_split.data_utils import DataSplitter
from models.data_loaders import DrugResistanceDataset, SampleEmbDataset




if __name__=="__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="rs2")
    parser.add_argument("--experiment_group", type=str, default="random_split")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--driams_dataset", type=str, choices=['A', 'B', 'C', 'D'], default="B")
    parser.add_argument("--driams_long_table", type=str,
                        default="../processed_data/DRIAMS_combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str,
                        default="../data/DRIAMS-B/spectra_binned_6000_2018.npy")
    parser.add_argument("--drugs_embeddings", type=str,
                        default="../processed_data/vae_drugs_embeddings.csv")

    parser.add_argument("--species_embedding_dim", type=int, default=16)
    parser.add_argument("--drug_embedding_dim", type=int, default=16)
    parser.add_argument(
        "--sample_hidden_layers", nargs="+", type=int, default=[256, 64]
    )
    parser.add_argument("--sample_embedding_dim", type=int, default=16)

    parser.add_argument("--conv_out_size", type=int, default=64)

    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=3)

    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    config = vars(args)

    output_folder = join("outputs", args.experiment_group, args.experiment_name)
    if not exists(output_folder):
        os.makedirs(output_folder)

    # if exists(join(output_folder, "embeddings", "drugs_embeddings.csv")):
    #     sys.exit(0)


    driams_long_table = pd.read_csv(args.driams_long_table)
    spectra_matrix = np.load(args.spectra_matrix)
    drugs_embeddings = pd.read_csv(args.drugs_embeddings, index_col=0)
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)

    species_list = sorted(dsplit.long_table["sample_id"].unique())
    train_df, val_df, test_df = dsplit.random_train_val_test_split(val_size=0.1, test_size=0.2, random_state=args.seed)

    train_dset = DrugResistanceDataset(train_df, spectra_matrix, drugs_embeddings, species_list)
    print(train_dset[0])
    val_dset = DrugResistanceDataset(val_df, spectra_matrix, drugs_embeddings, species_list)
    test_dset = DrugResistanceDataset(test_df, spectra_matrix, drugs_embeddings, species_list)


    sorted_species = sorted(dsplit.long_table["species"].unique())
    idx2species = {i: s for i, s in enumerate(sorted_species)}
    species2idx = {s: i for i, s in idx2species.items()}

    config["n_unique_species"] = len(idx2species)


    # Save configuration
    with open(join(output_folder, "config.json"), "w") as f:
        json.dump(config, f)


    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    experiment = AMR_Classifier_Experiment(config)

    with open(join(output_folder, "architecture.txt"), "w") as f:
        f.write(experiment.model.__repr__())



    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(output_folder, "checkpoints"),
                                          monitor="val_loss", filename="gst-{epoch:02d}-{val_loss:.4f}")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience
    )
    callbacks = [checkpoint_callback, early_stopping_callback]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=join(output_folder, "logs/"))

    print("Training..")
    trainer = pl.Trainer(devices="auto", accelerator="auto", default_root_dir=output_folder, max_epochs=args.n_epochs, callbacks=callbacks,
                         logger=tb_logger, log_every_n_steps=3,
                        #  limit_train_batches=20, limit_val_batches=10, #limit_test_batches=5
                         )
    trainer.fit(experiment, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    print("Training complete!")


    print("Testing..")
    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)
    # test_classification = {
    #     k: float(v) for k, v in test_results[0]["test_classification"].items()}
    with open(join(output_folder, "test_metrics.json"), "w") as f:
        json.dump(test_results[0], f, indent=2)

    print("Saving embeddings..")
    experiment.model.eval()
    embeddings_folder = join(output_folder, "embeddings")
    if not exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    # Species learned embeddings
    species_list = [idx2species[i] for i in range(len(idx2species))]
    species_lrn_embs = pd.DataFrame(experiment.model.species_emd.weight.detach().numpy(), 
                    index=species_list, columns=["z{}".format(i) for i in range(config["species_embedding_dim"])])
    species_lrn_embs.index.name = "species"
    species_lrn_embs.to_csv(join(embeddings_folder, "species_learned_embeddings.csv"))
    
    print("Testing complete")
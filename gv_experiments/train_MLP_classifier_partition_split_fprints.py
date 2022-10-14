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
from experiments.pl_experiment import Classifier_Experiment
from data_split.data_utils import DataSplitter
from models.data_loaders import DrugResistanceDataset_Fingerprints, SampleEmbDataset
from models.classifier import MLP_Classifier
import sys


def main(args):
    config = vars(args)

    output_folder = join("outputs", args.experiment_group, args.experiment_name, str(args.seed))
    if not exists(output_folder):
        os.makedirs(output_folder)

    metrics_folder = join("outputs", args.experiment_group, args.experiment_name, "metrics")
    if not exists(metrics_folder) and args.seed==0:
        os.makedirs(metrics_folder)

    experiment_folder = join("outputs", args.experiment_group, args.experiment_name)
    if exists(join(metrics_folder, "test_metrics_{}.json".format(args.seed))):
        sys.exit(0)

    driams_long_table = pd.read_csv(args.driams_long_table)
    spectra_matrix = np.load(args.spectra_matrix)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    driams_long_table = driams_long_table[driams_long_table["drug"].isin(drugs_df.index)]
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)

    species_list = sorted(dsplit.long_table["sample_id"].unique())
    trainval_df, test_df = dsplit.combination_train_test_split(dsplit.long_table, test_size=0.2, random_state=args.seed)
    # train_df, val_df = dsplit.combination_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)
    train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)
    test_df.to_csv(join(output_folder, "test_set.csv"), index=False)
    # sys.exit(0)

    train_dset = DrugResistanceDataset_Fingerprints(train_df, spectra_matrix, drugs_df, species_list, fingerprint_class=config["fingerprint_class"])
    print(train_dset[0])
    val_dset = DrugResistanceDataset_Fingerprints(val_df, spectra_matrix, drugs_df, species_list, fingerprint_class=config["fingerprint_class"])
    test_dset = DrugResistanceDataset_Fingerprints(test_df, spectra_matrix, drugs_df, species_list, fingerprint_class=config["fingerprint_class"])


    sorted_species = sorted(dsplit.long_table["species"].unique())
    idx2species = {i: s for i, s in enumerate(sorted_species)}
    species2idx = {s: i for i, s in idx2species.items()}

    config["n_unique_species"] = len(idx2species)


    # Save configuration
    if not exists(join(experiment_folder, "config.json")):
        del config["seed"]
        with open(join(experiment_folder, "config.json"), "w") as f:
            json.dump(config, f)



    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MLP_Classifier(config)
    experiment = Classifier_Experiment(config, model)

    if not exists(join(experiment_folder, "architecture.txt")):
        with open(join(experiment_folder, "architecture.txt"), "w") as f:
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
    with open(join(metrics_folder, "test_metrics_{}.json".format(args.seed)), "w") as f:
        json.dump(test_results[0], f, indent=2)


    print("Testing complete")



if __name__=="__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="MLP_class")
    parser.add_argument("--experiment_group", type=str, default="MLP_random_split_fprint")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--driams_dataset", type=str, choices=['A', 'B', 'C', 'D'], default="B")
    parser.add_argument("--driams_long_table", type=str,
                        default="../processed_data/DRIAMS_combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str,
                        default="../data/DRIAMS-B/spectra_binned_6000_2018.npy")
    parser.add_argument("--drugs_df", type=str,
                        default="../processed_data/drug_fingerprints.csv")


    parser.add_argument("--fingerprint_class", type=str, default="all", choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem"])
    # parser.add_argument("--fingerprint_size", type=int, default=167)

    parser.add_argument("--n_hidden_layers", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--input_size", type=int, default=8089)


    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    args.num_workers = os.cpu_count()
    main(args)

import sys
sys.path.insert(0, "../")
import numpy as np
import os
from os.path import join, exists
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from tqdm import tqdm
from argparse import ArgumentParser

from experiments.pl_experiment import AMR_Classifier
from data_split.data_utils import DataSplitter
from models.data_loaders import DrugResistanceDataset




if __name__=="__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="test111")
    parser.add_argument("--experiment_group", type=str, default="gv")

    parser.add_argument("--driams_dataset", type=str, choices=['A', 'B', 'C', 'D'], default="B")
    parser.add_argument("--driams_long_table", type=str,
                        default="processed_data/DRIAMS_combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str,
                        default="data/DRIAMS-B/spectra_binned_6000_2018.npy")
    parser.add_argument("--drugs_embeddings", type=str,
                        default="processed_data/vae_drugs_embeddings.csv")

    parser.add_argument("--species_embedding_dim", type=int, default=16)
    parser.add_argument("--drug_embedding_dim", type=int, default=16)
    parser.add_argument(
        "--sample_hidden_layers", nargs="+", type=int, default=[256, 64]
    )
    parser.add_argument("--sample_embedding_dim", type=int, default=16)

    parser.add_argument("--conv_out_size", type=int, default=64)

    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
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
    drugs_embeddings = pd.read_csv(args.drugs_embeddings)
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)

    train_val_df, test_df = dsplit

    train_dset = DrugResistanceDataset(driams_long_table, spectra_matrix, drugs_embeddings)
    val_dset = DrugResistanceDataset(driams_long_table, spectra_matrix, drugs_embeddings)
    test_dset = DrugResistanceDataset(driams_long_table, spectra_matrix, drugs_embeddings)


    species2idx = train_dset.species2idx
    idx2species = train_dset.idx2species
    config["n_unique_species"] = len(idx2species)

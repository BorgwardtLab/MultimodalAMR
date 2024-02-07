import sys

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
from multimodal_amr.experiments.pl_experiment import Classifier_Experiment
from multimodal_amr.data_split.data_utils import DataSplitter
from multimodal_amr.models.data_loaders import DrugResistanceDataset_Fingerprints, SampleEmbDataset
from multimodal_amr.models.classifier import Residual_AMR_Classifier
import sys
import time
from copy import deepcopy

COMBINATIONS = [
("Staphylococcus aureus", "Ciprofloxacin"),
("Klebsiella pneumoniae", "Meropenem"),
("Klebsiella pneumoniae", "Ciprofloxacin"),
("Escherichia coli", "Tobramycin"),
("Klebsiella pneumoniae", "Tobramycin"),
("Escherichia coli", "Cefepime"),
("Escherichia coli", "Ceftriaxone"),
("Escherichia coli", "Ciprofloxacin"),
("Klebsiella pneumoniae", "Cefepime"),
("Klebsiella pneumoniae", "Ceftriaxone"),
("Staphylococcus aureus", "Fusidic acid"),
("Staphylococcus aureus", "Oxacillin")
]


def main(args):
    config = vars(args)
    seed = args.seed
    output_folder = join("outputs", args.experiment_group, args.experiment_name, str(args.seed))
    if not exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    metrics_folder = join("outputs", args.experiment_group, args.experiment_name, "metrics")
    if not exists(metrics_folder):
        os.makedirs(metrics_folder, exist_ok=True)

    experiment_folder = join("outputs", args.experiment_group, args.experiment_name)
    

    driams_long_table = pd.read_csv(args.driams_long_table)

    comb_idx = args.seed//5
    split_idx = args.seed % 5
    


    target_species, target_drug = COMBINATIONS[comb_idx]
    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    target_sets = splits["{}_{}".format(target_species, target_drug)]
    test_split = target_sets[split_idx]


    predictions_folder = join(experiment_folder, "predictions", f"{target_species}_{target_drug}" )
    if not exists(predictions_folder):
        os.makedirs(predictions_folder, exist_ok=True)
        
    if exists(join(predictions_folder, f"split_{split_idx}.csv")):
        print("\n\nExperiment already performed!\n\n")
        sys.exit(0)


    spectra_matrix = np.load(args.spectra_matrix)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    driams_long_table = driams_long_table[driams_long_table["drug"].isin(drugs_df.index)]
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)

    samples_list = sorted(dsplit.long_table["sample_id"].unique())

    ix = (dsplit.long_table["sample_id"].isin(test_split))&(dsplit.long_table["drug"]==target_drug)
    test_df = dsplit.long_table[ix]
    trainval_df = dsplit.long_table[~ix]
    train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)

    finetune_ix = (~dsplit.long_table["sample_id"].isin(test_split))&(dsplit.long_table["drug"]==target_drug)&(dsplit.long_table["species"]==target_species)
    finetune_df = dsplit.long_table[finetune_ix]

    ft_train_df, ft_val_df = dsplit.baseline_train_test_split(finetune_df, test_size=0.25, random_state=args.seed)

    train_dset = DrugResistanceDataset_Fingerprints(train_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    val_dset = DrugResistanceDataset_Fingerprints(val_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    test_dset = DrugResistanceDataset_Fingerprints(test_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])

    ft_train_dset = DrugResistanceDataset_Fingerprints(ft_train_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    ft_val_dset = DrugResistanceDataset_Fingerprints(ft_val_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])


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

    model = Residual_AMR_Classifier(config)
    experiment = Classifier_Experiment(config, model)

    if not exists(join(experiment_folder, "architecture.txt")):
        with open(join(experiment_folder, "architecture.txt"), "w") as f:
            f.write(experiment.model.__repr__())


    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience
    )
    callbacks = [early_stopping_callback]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=join(output_folder, "logs/"))

    print("Training..")
    trainer = pl.Trainer(devices="auto", accelerator="auto", default_root_dir=output_folder, max_epochs=args.n_epochs, callbacks=callbacks,
                         logger=tb_logger, log_every_n_steps=3
                         )
    trainer.fit(experiment, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    print("Pretraining complete!")

    ft_train_loader = DataLoader(
        ft_train_dset, batch_size=16, shuffle=True, drop_last=True, num_workers=args.num_workers)
    ft_val_loader = DataLoader(
        ft_val_dset, batch_size=16, shuffle=False, num_workers=args.num_workers)

    config2 = deepcopy(config)
    config2["learning_rate"]/=3
    experiment2 = Classifier_Experiment(config2, deepcopy(experiment.model))
    trainer2 = pl.Trainer(devices="auto", accelerator="auto", default_root_dir=output_folder, max_epochs=args.n_finetune_epochs, callbacks=callbacks,
                         logger=tb_logger, log_every_n_steps=3
                         )
    trainer2.fit(experiment2, train_dataloaders=ft_train_loader,
            val_dataloaders=ft_val_loader)


    print("Testing..")
    test_results = trainer2.test(ckpt_path="best", dataloaders=test_loader)
    with open(join(metrics_folder, "test_metrics_{}.json".format(seed)), "w") as f:
        json.dump(test_results[0], f, indent=2)

    test_df["predicted_proba"] = experiment2.test_predictions
    test_df.to_csv(join(predictions_folder, f"split_{split_idx}.csv"), index=False)

    print("Testing complete")



if __name__=="__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="ResMLP_comparison")
    parser.add_argument("--experiment_group", type=str, default="PaperComparison")
    parser.add_argument("--seed", type=int, default=14)

    parser.add_argument("--driams_dataset", type=str, choices=['A', 'B', 'C', 'D'], default="A")
    parser.add_argument("--driams_long_table", type=str,
                        default="../processed_data/DRIAMS_combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str,
                        default="../data/DRIAMS-A/spectra_binned_6000_all.npy")
    parser.add_argument("--drugs_df", type=str,
                        default="../processed_data/drug_fingerprints.csv")
    parser.add_argument("--splits_file", type=str,
                        default="../data/AMR_baseline_splits.json")

    parser.add_argument("--drug_emb_type", type=str, default="fingerprint", choices=["fingerprint", "vae_embedding"])
    parser.add_argument("--fingerprint_class", type=str, default="morgan_1024", choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem"])
    parser.add_argument("--fingerprint_size", type=int, default=1024)


    parser.add_argument("--n_hidden_layers", type=int, default=5)
    parser.add_argument("--conv_out_size", type=int, default=512)
    parser.add_argument("--sample_embedding_dim", type=int, default=512)
    parser.add_argument("--drug_embedding_dim", type=int, default=512)


    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--n_finetune_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    
    args = parser.parse_args()
    args.num_workers = os.cpu_count()
    args.species_embedding_dim = 0


    main(args)

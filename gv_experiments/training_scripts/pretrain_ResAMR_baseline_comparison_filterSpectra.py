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
from copy import deepcopy
import wandb

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

    if config.get("training_setup", None) is not None:
        comb_idx = args.training_setup//5
        split_idx = args.training_setup % 5
        target_species, target_drug = COMBINATIONS[comb_idx]
    else:
        target_species = config.get("target_species", None)
        target_drug = config.get("target_drug", None)
        if target_drug is None or target_species is None:
            comb_idx = config.get("combination_idx", None)
            if comb_idx is None:
                raise ValueError(
                    "Please specify either the target species and target drug, or an index to select a combination")
            target_species, target_drug = COMBINATIONS[comb_idx]
        split_idx = config["split_idx"]

    output_folder = join("outputs", args.experiment_group, args.experiment_name, "{}_{}".format(
        target_species, target_drug), "pretraining", f"split_{split_idx}")

    root_folder = config.get("root_folder", None)
    if root_folder is not None:
        output_folder = join(root_folder, output_folder)
    experiment_folder = output_folder

    checkpoints_folder = os.path.join(output_folder, "checkpoints")
    if os.path.exists(checkpoints_folder):
        fnames = os.listdir(checkpoints_folder)
        if len(fnames) > 0:
            if "val_loss=0.0000" not in fnames[0]:
                print(fnames)
                print("Combination already trained, exiting")
                sys.exit(0)

    os.makedirs(output_folder, exist_ok=True)
    predictions_folder = join(output_folder, "predictions_check")
    os.makedirs(predictions_folder, exist_ok=True)
    data_folder = join(output_folder, "pretraining_data_splits")
    os.makedirs(data_folder, exist_ok=True)

    driams_long_table = pd.read_csv(args.driams_long_table)

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    target_sets = splits["{}_{}".format(target_species, target_drug)]
    test_samples = set(target_sets[split_idx])

    spectra_matrix = np.load(args.spectra_matrix)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    driams_long_table = driams_long_table[driams_long_table["drug"].isin(
        drugs_df.index)]
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)

    samples_list = sorted(dsplit.long_table["sample_id"].unique())
    assert len(samples_list) == len(spectra_matrix)

    ix = (dsplit.long_table["species"] == target_species) & (
        dsplit.long_table["drug"] == target_drug)
    target_df = dsplit.long_table[ix]
    target_df.to_csv(join(data_folder, "target_df.csv"), index=False)
    test_df = target_df[target_df["sample_id"].isin(test_samples)]
    test_df.to_csv(join(data_folder, "test_df.csv"), index=False)

    trainval_df = dsplit.long_table[~ix]

    n_tv_samples_prefiltering = len(trainval_df)
    trainval_df = trainval_df[~trainval_df["sample_id"].isin(test_samples)]
    print("Removed {} samples".format(
        n_tv_samples_prefiltering-len(trainval_df)))

    train_df, val_df = dsplit.baseline_train_test_split(
        trainval_df, test_size=0.2, random_state=args.seed)
    train_df.to_csv(join(data_folder, "train_df.csv"), index=False)
    val_df.to_csv(join(data_folder, "val_df.csv"), index=False)

    train_dset = DrugResistanceDataset_Fingerprints(
        train_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    val_dset = DrugResistanceDataset_Fingerprints(
        val_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    test_dset = DrugResistanceDataset_Fingerprints(
        test_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])

    sorted_species = sorted(dsplit.long_table["species"].unique())
    idx2species = {i: s for i, s in enumerate(sorted_species)}
    species2idx = {s: i for i, s in idx2species.items()}

    config["n_unique_species"] = len(idx2species)

    # Save configuration
    if not exists(join(experiment_folder, "config.json")):
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

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_folder,
                                          monitor="val_loss", filename="gst-{epoch:02d}-{val_loss:.4f}", save_top_k=1)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience
    )
    callbacks = [early_stopping_callback, checkpoint_callback]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=join(output_folder, "logs/"))
    # wandb_logger = pl_loggers.WandbLogger(project=args.experiment_name)

    print("Training..")
    trainer = pl.Trainer(devices="auto", accelerator="auto", default_root_dir=output_folder, max_epochs=args.n_epochs, callbacks=callbacks,
                         logger=tb_logger, log_every_n_steps=3, num_sanity_val_steps=0
                         )
    trainer.fit(experiment, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    print("Pretraining complete!")
    print("Best checkpoint ", checkpoint_callback.best_model_path)
    if "val_loss=0.000" in checkpoint_callback.best_model_path:
        os.rmdir(os.path.join(output_folder, "checkpoints"))
        raise RuntimeError("An issue occurred during the training of the model. A likely cause is the Nvidia driver.")

    print("Testing..")
    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)

    test_df["predicted_proba"] = experiment.test_predictions
    test_df.to_csv(join(predictions_folder,
                   f"test_pretrained_{target_species.replace(' ', '_')}_{target_drug.replace(' ', '_')}_split{split_idx}.csv"), index=False)

    print("Testing complete")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str,
                        default="PretrainingResMLP_filterSpectra")
    parser.add_argument("--experiment_group", type=str,
                        default="SingleSpSingleDrugResMLP")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--training_setup", type=int)
    parser.add_argument("--combination_idx", type=int)
    parser.add_argument("--target_drug", type=str)
    parser.add_argument("--target_species", type=str)
    parser.add_argument("--split_idx", type=int)

    parser.add_argument("--driams_dataset", type=str,
                        choices=['A', 'B', 'C', 'D'], default="A")
    parser.add_argument("--driams_long_table", type=str)
    parser.add_argument("--spectra_matrix", type=str)
    parser.add_argument("--drugs_df", type=str)
    parser.add_argument("--splits_file", type=str)
    parser.add_argument("--root_folder", type=str)

    parser.add_argument("--drug_emb_type", type=str,
                        default="fingerprint", choices=["fingerprint", "vae_embedding"])
    parser.add_argument("--fingerprint_class", type=str, default="morgan_1024",
                        choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem"])
    parser.add_argument("--fingerprint_size", type=int, default=1024)

    parser.add_argument("--n_hidden_layers", type=int, default=5)
    parser.add_argument("--conv_out_size", type=int, default=512)
    parser.add_argument("--sample_embedding_dim", type=int, default=512)
    parser.add_argument("--drug_embedding_dim", type=int, default=512)

    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    args.num_workers = os.cpu_count()
    args.species_embedding_dim = 0

    main(args)

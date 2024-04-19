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
import argparse
from argparse import ArgumentParser
import json
from multimodal_amr.experiments.pl_experiment import Classifier_Experiment
from multimodal_amr.data_split.data_utils import DataSplitter
from multimodal_amr.models.data_loaders import DrugResistanceDataset_Fingerprints, SampleEmbDataset
from multimodal_amr.models.classifier import Residual_AMR_Classifier
import wandb
from sklearn.metrics import roc_curve, auc, precision_recall_curve

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

    print(
        f"Split {split_idx} - Target species {target_species} - Target drug {target_drug}")

    pretrained_checkpoints_folder = config["pretrained_checkpoints_folder"]

    output_folder = join("outputs", args.experiment_group, args.experiment_name,
                         "finetuning", "{}_{}".format(target_species, target_drug), f"split_{split_idx}")
    experiment_folder = join("outputs", args.experiment_group, args.experiment_name,
                             "finetuning", "{}_{}".format(target_species, target_drug))
    root_folder = config.get("root_folder", None)
    if root_folder is not None:
        output_folder = join(root_folder, output_folder)
        experiment_folder = join(root_folder, experiment_folder)
    data_folder = join(output_folder, "finetuning_data_splits")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)

    model_checkpoint_folder = join(pretrained_checkpoints_folder, "{}_{}".format(
        target_species.replace(" ", "_"), target_drug.replace(" ", "_")))
    if not os.path.exists(model_checkpoint_folder):
        raise FileNotFoundError(
            f"Pretrained model folder {model_checkpoint_folder} not found")
    print("Checkpoint folder ", model_checkpoint_folder)
    checkpoint_fname = os.listdir(model_checkpoint_folder)[0]
    print("Checkpoints file ", checkpoint_fname)
    checkpoint_path = join(model_checkpoint_folder, checkpoint_fname)
    print("Checkpoints path ", checkpoint_path)

    driams_long_table = pd.read_csv(args.driams_long_table)

    with open(args.splits_file, "r") as f:
        splits = json.load(f)
    target_sets = splits["{}_{}".format(target_species, target_drug)]
    test_split = target_sets[split_idx]

    predictions_folder = join(experiment_folder, "predictions")
    if not exists(predictions_folder):
        os.makedirs(predictions_folder, exist_ok=True)

    if exists(join(predictions_folder, f"split_{split_idx}.csv")):
        print("\n\nExperiment already performed!\n\n")
        sys.exit(0)

    spectra_matrix = np.load(args.spectra_matrix)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    driams_long_table = driams_long_table[driams_long_table["drug"].isin(
        drugs_df.index)]
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)

    samples_list = sorted(dsplit.long_table["sample_id"].unique())
    assert len(samples_list) == len(spectra_matrix)

    ix = (dsplit.long_table["species"] == target_species) & (
        dsplit.long_table["drug"] == target_drug)
    target_combination_df = dsplit.long_table[ix]
    test_df = target_combination_df[target_combination_df["sample_id"].isin(
        test_split)]
    test_df.to_csv(join(data_folder, "test_df.csv"), index=False)

    trainval_df = target_combination_df[~target_combination_df["sample_id"].isin(
        test_split)]

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
    if not exists(join(output_folder, "config.json")):
        # del config["seed"]
        with open(join(output_folder, "config.json"), "w") as f:
            json.dump(config, f)

    train_loader = DataLoader(
        train_dset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_loader = DataLoader(
        val_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(
        test_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = Residual_AMR_Classifier(config)

    # Load pretrained model
    checkpoint = torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage)
    experiment = Classifier_Experiment(config, model)
    experiment.load_state_dict(checkpoint["state_dict"])
    experiment.test_predictions = []

    if not exists(join(output_folder, "architecture.txt")):
        with open(join(output_folder, "architecture.txt"), "w") as f:
            f.write(experiment.model.__repr__())

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(output_folder, "checkpoints"),
                                          monitor="val_loss", filename="gst-{epoch:02d}-{val_loss:.4f}", save_top_k=1)
    early_stopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=args.patience
    )
    callbacks = [early_stopping_callback, checkpoint_callback]

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=join(output_folder, "logs/"))
    # wandb_logger = pl_loggers.WandbLogger(project=args.experiment_name+"_finetuning")

    print("Training..")
    trainer = pl.Trainer(devices="auto", accelerator="auto", default_root_dir=output_folder, max_epochs=args.n_epochs, callbacks=callbacks,
                         logger=tb_logger, log_every_n_steps=3, num_sanity_val_steps=0
                         )
    trainer.fit(experiment, train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    print("Finetuning complete!")

    print("Best checkpoint ", checkpoint_callback.best_model_path)
    if "val_loss=0.000" in checkpoint_callback.best_model_path:
        os.rmdir(os.path.join(output_folder, "checkpoints"))
        raise RuntimeError("An issue occurred during the training of the model. A likely cause is the Nvidia driver.")

    print("Testing..")
    test_results = trainer.test(ckpt_path="best", dataloaders=test_loader)

    test_df["predicted_proba"] = experiment.test_predictions
    test_df.to_csv(
        join(predictions_folder, f"split_{split_idx}.csv"), index=False)

    fpr, tpr, thresh = roc_curve(
        test_df["response"], test_df["predicted_proba"])
    roc_score = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(
        test_df["response"], test_df["predicted_proba"])
    auprc = auc(recall, precision)
    test_metrics = {"AUROC": roc_score, "AUPRC": auprc}
    print("Test metrics: ", test_metrics)
    with open(join(output_folder, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("Testing complete")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str,
                        default="FinetuningResMLP")
    parser.add_argument("--experiment_group", type=str,
                        default="SingleSpSingleDrugResMLP")
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--pretrained_checkpoints_folder",
                        type=str, default="")

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

    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    args = parser.parse_args()
    args.num_workers = os.cpu_count()
    args.species_embedding_dim = 0

    main(args)

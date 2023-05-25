import sys
sys.path.insert(0, "..")
# sys.path.insert(0, "../data_split")

import numpy as np
import os
from os.path import join, exists
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
import json
from data_split.data_utils import DataSplitter
from models.data_loaders import DrugResistanceDataset_Fingerprints
from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score, f1_score, average_precision_score
from sklearn.metrics import precision_score, recall_score
import json
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import itertools
from sklearn.preprocessing import StandardScaler


TRAINING_SETUPS = list(itertools.product(['A', 'B', 'C', 'D'], ["random", "drug_species_zero_shot"], np.arange(10))) + \
                  list(itertools.product(['A', 'B', 'C', 'D'], ["drugs_zero_shot"], np.arange(60)))


def main(args):
    config = vars(args)

    # Set up output folders
    output_folder = join("outputs", args.experiment_group, args.experiment_name+"_"+str(args.seed))
    if not exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    metrics_folder = join("outputs", args.experiment_group, args.experiment_name+"_metrics")
    if not exists(metrics_folder):
        os.makedirs(metrics_folder, exist_ok=True)

    # Load data
    driams_long_table = pd.read_csv(args.driams_long_table)
    spectra_matrix = np.load(args.spectra_matrix)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)
    driams_long_table = driams_long_table[driams_long_table["drug"].isin(drugs_df.index)]

    # Split sets
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)
    samples_list = sorted(dsplit.long_table["sample_id"].unique())
    if args.split_type=="random":
        train_df, val_df, test_df = dsplit.random_train_val_test_split(val_size=0.1, test_size=0.2, random_state=args.seed)
    elif args.split_type=="drug_species_zero_shot":
        trainval_df, test_df = dsplit.combination_train_test_split(dsplit.long_table, test_size=0.2, random_state=args.seed)
        train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)
    elif args.split_type =="drugs_zero_shot":
        drugs_list = sorted(dsplit.long_table["drug"].unique())
        if args.seed>=len(drugs_list):
            print("Drug index out of bound, exiting..\n\n")
            sys.exit(0)
        target_drug = drugs_list[args.seed]
        test_df, trainval_df = dsplit.drug_zero_shot_split(drug=target_drug)
        train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=args.seed)

    train_dset = DrugResistanceDataset_Fingerprints(train_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    val_dset = DrugResistanceDataset_Fingerprints(val_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])
    test_dset = DrugResistanceDataset_Fingerprints(test_df, spectra_matrix, drugs_df, samples_list, fingerprint_class=config["fingerprint_class"])

    sorted_species = sorted(dsplit.long_table["species"].unique())
    idx2species = {i: s for i, s in enumerate(sorted_species)}
    species2idx = {s: i for i, s in idx2species.items()}

    config["n_unique_species"] = len(idx2species)

    # Save configuration
    with open(join(output_folder, "config.json"), "w") as f:
        json.dump(config, f)

    # Process data
    X_train_spectra = []
    X_train_fingerprints = []
    y_train = []
    for i in tqdm(range(len(train_dset))):
        sample = train_dset[i]
        x_spectrum = sample[1].numpy()
        fprint = sample[2].numpy()
        X_train_spectra.append(x_spectrum)
        X_train_fingerprints.append(fprint)
        # X_train.append(np.concatenate([x_spectrum, fprint]))
        y_train.append(sample[3].item())

    X_train_spectra = np.vstack(X_train_spectra)
    X_train_fingerprints = np.vstack(X_train_fingerprints)
    y_train = np.array(y_train)

    X_val_spectra = []
    X_val_fingerprints = []
    y_val = []
    for i in tqdm(range(len(val_dset))):
        sample = val_dset[i]
        x_spectrum = sample[1].numpy()
        fprint = sample[2].numpy()
        X_val_spectra.append(x_spectrum)
        X_val_fingerprints.append(fprint)
        y_val.append(sample[3].item())

    X_val_spectra = np.vstack(X_val_spectra)
    X_val_fingerprints = np.vstack(X_val_fingerprints)
    y_val = np.array(y_val)


    X_test_spectra = []
    X_test_fingerprints = []
    y_test = []
    for i in tqdm(range(len(test_dset))):
        sample = test_dset[i]
        x_spectrum = sample[1].numpy()
        fprint = sample[2].numpy()
        X_test_spectra.append(x_spectrum)
        X_test_fingerprints.append(fprint)
        y_test.append(sample[3].item())

    X_test_spectra = np.vstack(X_test_spectra)
    X_test_fingerprints = np.vstack(X_test_fingerprints)
    y_test = np.array(y_test)
    print("Data Processed")

    # Project data
    print("Calculating PCA..")
    pca_spectra = PCA(n_components=args.perc_variance)
    pca_fprints = PCA(n_components=args.perc_variance)


    pca_spectra.fit(X_train_spectra)
    pca_fprints.fit(X_train_fingerprints)


    

    train_pca_embeddings = np.hstack([pca_spectra.transform(X_train_spectra), pca_fprints.transform(X_train_fingerprints)])
    val_pca_embeddings = np.hstack([pca_spectra.transform(X_val_spectra), pca_fprints.transform(X_val_fingerprints)])
    test_pca_embeddings = np.hstack([pca_spectra.transform(X_test_spectra), pca_fprints.transform(X_test_fingerprints)])


    scaler = StandardScaler()
    train_pca_embeddings = scaler.fit_transform(train_pca_embeddings)
    val_pca_embeddings = scaler.transform(val_pca_embeddings)
    test_pca_embeddings = scaler.transform(test_pca_embeddings)

    print("Training model")
    model = LogisticRegression()
    model.fit(train_pca_embeddings, y_train)

    # Test model
    test_predictions = model.predict(test_pca_embeddings)
    test_proba = model.predict_proba(test_pca_embeddings)
    ix = np.where(model.classes_>0.5)[0]
    test_metrics = {
            "mcc": matthews_corrcoef(y_test, test_predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test, test_predictions),
            "f1": f1_score(y_test, test_predictions, zero_division=0),
            "AUPRC": average_precision_score(y_test, test_proba[:, ix]),
            "precision": precision_score(y_test, test_predictions, zero_division=0),
            "recall": recall_score(y_test, test_predictions, zero_division=0),
            "n_pca_components_spectra": int(pca_spectra.n_components_),
            "n_pca_components_fprints": int(pca_fprints.n_components_),
        }

    print(test_metrics)
    if args.split_type == "drugs_zero_shot":
        with open(join(metrics_folder, "test_metrics_{}.json".format(target_drug)), "w") as f:
            json.dump(test_metrics, f, indent=2)
    else:
        with open(join(metrics_folder, "test_metrics_{}.json".format(args.seed)), "w") as f:
            json.dump(test_metrics, f, indent=2)

    print("Analysis complete!")



if __name__=="__main__":

    parser = ArgumentParser()

    parser.add_argument("--training_setup", type=int)

    parser.add_argument("--experiment_name", type=str, default="concat_95_perc_variance")
    parser.add_argument("--experiment_group", type=str, default="PCA_Split")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--driams_dataset", type=str, choices=['A', 'B', 'C', 'D'], default="B")
    parser.add_argument("--driams_long_table", type=str,
                        default="../processed_data/DRIAMS_combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str,
                        default="../data/DRIAMS-B/spectra_binned_6000_2018.npy")
    parser.add_argument("--drugs_df", type=str,
                        default="../processed_data/drug_fingerprints.csv")
    parser.add_argument("--split_type", type=str, default="random", choices=["random", "drug_species_zero_shot", "drugs_zero_shot"])

    parser.add_argument("--fingerprint_class", type=str, default="MACCS", choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem"])
    parser.add_argument("--perc_variance", type=float, default=0.95)

    args = parser.parse_args()
    if args.training_setup is not None:
        dataset, split_type, seed = TRAINING_SETUPS[args.training_setup]

        args.driams_dataset = dataset
        args.split_type = split_type
        args.seed = int(seed)

    args.experiment_name = args.experiment_name + f"_DRIAMS-{args.driams_dataset}_{args.split_type}"

    main(args)

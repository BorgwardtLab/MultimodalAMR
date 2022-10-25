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

from sklearn.preprocessing import StandardScaler

def main(args):
    config = vars(args)

    # Set up output folders
    output_folder = join("outputs", args.experiment_group, args.experiment_name+"_"+str(args.seed))
    if not exists(output_folder):
        os.makedirs(output_folder)

    metrics_folder = join("outputs", args.experiment_group, args.experiment_name+"_metrics")
    if not exists(metrics_folder):
        os.makedirs(metrics_folder)

    # Load data
    driams_long_table = pd.read_csv(args.driams_long_table)
    spectra_matrix = np.load(args.spectra_matrix)
    drugs_df = pd.read_csv(args.drugs_df, index_col=0)

    # Split sets
    dsplit = DataSplitter(driams_long_table, dataset=args.driams_dataset)
    samples_list = sorted(dsplit.long_table["sample_id"].unique())
    if args.split_type=="random":
        train_df, val_df, test_df = dsplit.random_train_val_test_split(val_size=0.1, test_size=0.2, random_state=args.seed)
    elif args.split_type=="partitioned":
        trainval_df, test_df = dsplit.combination_train_test_split(dsplit.long_table, test_size=0.2, random_state=args.seed)
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
    X_train = []
    y_train = []
    for i in tqdm(range(len(train_dset))):
        sample = train_dset[i]
        x_spectrum = sample[1].numpy()
        fprint = sample[2].numpy()
        X_train.append(np.concatenate([x_spectrum, fprint]))
        y_train.append(sample[3].item())

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    X_val = []
    y_val = []
    for i in tqdm(range(len(val_dset))):
        sample = val_dset[i]
        x_spectrum = sample[1].numpy()
        fprint = sample[2].numpy()
        X_val.append(np.concatenate([x_spectrum, fprint]))
        y_val.append(sample[3].item())

    X_val = np.vstack(X_val)
    y_val = np.array(y_val)

    X_test = []
    y_test = []
    for i in tqdm(range(len(test_dset))):
        sample = test_dset[i]
        x_spectrum = sample[1].numpy()
        fprint = sample[2].numpy()
        X_test.append(np.concatenate([x_spectrum, fprint]))
        y_test.append(sample[3].item())

    X_test = np.vstack(X_test)
    y_test = np.array(y_test)
    print("Data Processed")

    # Project data
    print("Calculating PCA..")
    pca = PCA(n_components=args.n_components)
    pca.fit(X_train)
    train_pca_embeddings = pca.transform(X_train)
    val_pca_embeddings = pca.transform(X_val)
    test_pca_embeddings = pca.transform(X_test)


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
            "recall": recall_score(y_test, test_predictions, zero_division=0)
        }

    print(test_metrics)
    with open(join(metrics_folder, "test_metrics_{}.json".format(args.seed)), "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("Analysis complete!")



if __name__=="__main__":

    parser = ArgumentParser()

    parser.add_argument("--experiment_name", type=str, default="comp128")
    parser.add_argument("--experiment_group", type=str, default="PCA_partition_split")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--driams_dataset", type=str, choices=['A', 'B', 'C', 'D'], default="B")
    parser.add_argument("--driams_long_table", type=str,
                        default="../processed_data/DRIAMS_combined_long_table.csv")
    parser.add_argument("--spectra_matrix", type=str,
                        default="../data/DRIAMS-B/spectra_binned_6000_2018.npy")
    parser.add_argument("--drugs_df", type=str,
                        default="../processed_data/drug_fingerprints.csv")


    parser.add_argument("--fingerprint_class", type=str, default="all", choices=["all", "MACCS", "morgan_512", "morgan_1024", "pubchem"])
    parser.add_argument("--n_components", type=int, default=128)

    args = parser.parse_args()
    for i in range(10):
        args.seed = i
        main(args)
        print(f"\n\nProcessing seed {i}\n\n")
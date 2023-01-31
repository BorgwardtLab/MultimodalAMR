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
import json
import itertools
from argparse import ArgumentParser



TRAINING_SETUPS = list(itertools.product(['A', 'B', 'C', 'D'], ["random", "partitioned"], np.arange(10))) + \
                list(itertools.product(['A', 'B', 'C', 'D'], ["drugs_zero_shot"], np.arange(60)))
if __name__=="__main__":

    parser = ArgumentParser()
    parser.add_argument("--training_setup", type=int, default=0)
    args = parser.parse_args()

    output_folder = "outputs/Imbalance"
    if not exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    drugs_df = pd.read_csv("processed_data/drug_fingerprints.csv", index_col=0)
    driams_long_table = pd.read_csv("processed_data/DRIAMS_combined_long_table.csv")
    driams_long_table = driams_long_table[driams_long_table["drug"].isin(drugs_df.index)]

    # imbalance_df = []


    # for dset, split_type, seed in tqdm(TRAINING_SETUPS):
    dset, split_type, seed = TRAINING_SETUPS[args.training_setup]
    dsplit = DataSplitter(driams_long_table, dataset=dset)
    target_drug = ""
    
    if split_type=="random":
        train_df, val_df, test_df = dsplit.random_train_val_test_split(val_size=0.1, test_size=0.2, random_state=seed)
    elif split_type=="partitioned":
        trainval_df, test_df = dsplit.combination_train_test_split(dsplit.long_table, test_size=0.2, random_state=seed)
        # train_df, val_df = dsplit.baseline_train_test_split(trainval_df, test_size=0.2, random_state=seed)
    elif split_type =="drugs_zero_shot":
        drugs_list = sorted(dsplit.long_table["drug"].unique())
        if seed>=len(drugs_list):
            print("Drug index out of bound, exiting..\n\n")
            sys.exit(0)
        target_drug = drugs_list[seed]
        test_df, trainval_df = dsplit.drug_zero_shot_split(drug=target_drug)
    # imbalance_df.append({"dataset": dset, "seed": seed, "Frac. Resistant": test_df["response"].mean(), "drug": target_drug})
    results = {"dataset": dset, "seed": int(seed), "split_type": split_type, "Frac. Resistant": test_df["response"].mean(), "drug": target_drug}

    with open(join(output_folder, f"imbalance_metrics_{split_type}_{seed}_{dset}.json"), "w") as f:
        json.dump(results, f)                           
    # imbalance_df = pd.DataFrame(imbalance_df)
    # imbalance_df.to_csv("outputs/Splits_Imbalance.csv", index=False)

    print("Analysis complete\n\n")
                    

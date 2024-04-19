import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def sum_solver(arr, target_sum):
    # Find the elements in an array that most closely approximate the desired sum
    possible_sums = {}
    records = []
    for i in range(len(arr)):
        possible_sums = {**possible_sums, **{arr[i] + s: i for s in possible_sums}}
        possible_sums[arr[i]] = i
        records.append(possible_sums)

    best_sum = max(s for s in possible_sums if s <= target_sum)
    record_idx = len(arr) - 1
    res = []
    while best_sum:
        last_idx = records[record_idx][best_sum]
        res.append(last_idx)
        best_sum -= arr[last_idx]
        record_idx = last_idx - 1
    return res


class DataSplitter:
    """
    This class gathers all methods to split the data for each DRIAMS dataset according to the
    specific task to perform.
    The first set of functions, labelled BASELINE COMPARISON, is used for comparison with the
    previous paper.
    The second set of functions is used for additional tests such as zero-shot prediction.
    """

    def __init__(self, long_table=None, dataset="B"):
        self.long_table = long_table
        if dataset is not None:
            self.long_table = long_table[long_table["dataset"] == dataset].reset_index(
                drop=True
            )
        self.dataset = dataset

    def __len__(self):
        return len(self.long_table)

    def __getitem__(self, idx):
        return self.long_table.iloc[idx]

    ###############################################################
    # BASELINE COMPARISON FUNCTIONS
    # The following functions are used to select data for one specific species-drug combination.
    # - "baseline_selection" filters the table and returns two DataFrames: one with the target data
    #   and one with the remaining data
    # - "baseline_train_test_split" takes as input the target DataFrame returned by the previous
    #   function and splits it into training and test sets stratifying by response
    # - "baseline_kfold_cv" handles the kfold split for the same target DataFrame, stratifying
    #   by response.

    def baseline_selection(self, drug=None, species=None):
        ix = (self.long_table["species"] == species) & (self.long_table["drug"] == drug)
        assert (
            sum(ix) > 0
        ), f"No split available for combination drug {drug} - species {species}"
        return self.long_table[ix].reset_index(drop=True), self.long_table[
            ~ix
        ].reset_index(drop=True)

    def baseline_train_test_split(self, df, test_size=0.2, random_state=42):
        train_X, test_X = train_test_split(
            df, stratify=df["response"], random_state=random_state
        )
        return train_X, test_X

    def baseline_kfold_cv(self, df, cv=5, random_state=None):
        skf = StratifiedKFold(n_splits=cv, random_state=random_state)
        folds = []
        for train_index, test_index in skf.split(df, df["response"]):
            folds.append((df.iloc[train_index], df.iloc[test_index]))
        return folds

    # END - BASELINE COMPARISON FUNCTIONS
    ###############################################################

    ###############################################################
    # AGGREGATE SPLITTING FUNCTIONS

    def balanced_train_test_split(self, df, test_size=0.2):
        drug_counts_mapping = dict(df["drug"].value_counts())
        drugs = sorted(list(df["drug"].unique()), key=lambda x: drug_counts_mapping[x])
        species_counts_mapping = dict(df["species"].value_counts())
        species = sorted(
            list(df["species"].unique()), key=lambda x: species_counts_mapping[x]
        )

        df2 = pd.DataFrame(df[["drug", "species"]].value_counts()).reset_index()
        df2.columns = ["drug", "species", "combination_counts"]
        df2["drug_counts"] = df2["drug"].map(drug_counts_mapping)
        df2["species_counts"] = df2["species"].map(species_counts_mapping)
        df2["sum_counts"] = df2["drug_counts"] + df2["species_counts"]

        train_pairings, test_pairings = None, None
        # Ensure that all species are represented in the training set
        for sp in species:
            species_pairings = df2[df2["species"] == sp]
            sel_samples = species_pairings.sample(
                weights=(1 / species_pairings["species_counts"])
            )
            if train_pairings is None:
                train_pairings = sel_samples
            else:
                train_pairings = pd.concat((train_pairings, sel_samples))

        df3 = df2[
            ~df2[["drug", "species"]]
            .apply(tuple, 1)
            .isin(train_pairings[["drug", "species"]].apply(tuple, 1))
        ]

        # Ensure that all drugs represented in the training set
        for dr in drugs:
            if dr in train_pairings["drug"]:
                continue
            drug_pairings = df3[df3["drug"] == dr]
            sel_samples = drug_pairings.sample(
                weights=(1 / drug_pairings["drug_counts"])
            )
            train_pairings = pd.concat((train_pairings, sel_samples))

        # Fill in test set
        df3 = df2[
            ~df2[["drug", "species"]]
            .apply(tuple, 1)
            .isin(train_pairings[["drug", "species"]].apply(tuple, 1))
        ].reset_index(drop=True)
        n_test = int(
            test_size * len(df)
        )  # - train_pairings['combination_counts'].sum()
        test_combination_ixs = sum_solver(df3["combination_counts"].values, n_test)
        test_pairings = df3.iloc[test_combination_ixs]

        train_data = df[
            ~df[["drug", "species"]]
            .apply(tuple, 1)
            .isin(test_pairings[["drug", "species"]].apply(tuple, 1))
        ].reset_index(drop=True)
        test_data = df[
            df[["drug", "species"]]
            .apply(tuple, 1)
            .isin(test_pairings[["drug", "species"]].apply(tuple, 1))
        ].reset_index(drop=True)
        return train_data, test_data
        # return train_data, test_data

    def drug_zero_shot_split(self, drug=None):
        ix = self.long_table["drug"] == drug
        assert sum(ix) > 0, f"No split available for drug {drug}"
        return self.long_table[ix].reset_index(drop=True), self.long_table[
            ~ix
        ].reset_index(drop=True)

    # END - AGGREGATE SPLITTING FUNCTIONS
    ###############################################################


def get_metrics(cm):
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    return TPR, TNR, ACC

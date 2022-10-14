import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import (
    make_scorer,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
    auc,
    precision_recall_curve,
    roc_curve,
)
from scipy.stats import uniform
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import sys

sys.path.insert(0, "../")


from data_utils import DataSplitter, get_metrics
import os
import pickle
from os.path import join, exists
import json
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE


def main(args):

    print("Loading data")
    data = pd.read_csv("../processed_data/DRIAMS_combined_long_table.csv")
    dsplit = DataSplitter(data, dataset=args.dataset)
    counts = dsplit.long_table[["species", "drug"]].value_counts()
    experiments = sorted(list(counts[counts > args.threshold].index))

    print("Executing {} experiments...".format(len(experiments)))

    print("Loading spectra")
    spectra_matrix = pd.read_csv(args.spectra_matrix_path, index_col=0)

    scorers = {
        "precision_score": make_scorer(precision_score),
        "recall_score": make_scorer(recall_score),
        "accuracy_score": make_scorer(accuracy_score),
        "balanced_accuracy_score": make_scorer(balanced_accuracy_score),
        "f1_score": make_scorer(f1_score),
        "mcc": make_scorer(matthews_corrcoef),
    }

    if args.model == "GBM":
        model_template = LGBMClassifier()
        distributions = {
            "lgbmclassifier__max_depth": [2, 3, 5, 7, None],
            "lgbmclassifier__min_data_in_leaf": [2, 5, 10, 50, 100],
            "lgbmclassifier__num_leaves": [2**2, 2**3, 2**5, 2**7],
        }

    elif args.model == "MLP":
        model_template = MLPClassifier(solver="lbfgs")
        distributions = {
            "mlpclassifier__activation": ["relu"],
            "mlpclassifier__alpha": np.logspace(-4, -1, 11),
            "mlpclassifier__hidden_layer_sizes": [(2048), (1024, 64), (2048, 512, 32)],
        }
    elif args.model == "LogisticRegression":
        model_template = LogisticRegression()
        distributions = {
            "logisticregression__class_weight": ["balanced", None],
            "logisticregression__C": uniform(loc=0.1, scale=9.9),
        }

    out_folder = join(args.output_folder, f"DRIAMS_{args.dataset}", args.model)

    if not exists(out_folder):
        os.makedirs(out_folder)

    with open(join(out_folder, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Training {args.model} baseline")

    combination_pairs = {
        "Escherichia coli": [
            "Ciprofloxacin",
            "Ceftriaxone",
            "Cefepime",
            "Tobramycin",
            "Piperacillin",
        ],
        "Klebsiella pneumoniae": [
            "Ciprofloxacin",
            "Ceftriaxone",
            "Cefepime",
            "Meropenem",
            "Tobramycin",
        ],
        "Staphylococcus aureus": ["Ciprofloxacin", "Fusidic acid", "Oxacillin"],
    }

    for seed, (sp, dr) in tqdm(enumerate(experiments)):

        if sp in combination_pairs.keys() and dr in combination_pairs[sp]:

            baseline_results = []

            target_df, _ = dsplit.baseline_selection(drug=dr, species=sp)

            # Align samples in targets and data
            target_df = pd.merge(
                target_df.set_index("sample_id")["response"],
                spectra_matrix,
                left_index=True,
                right_index=True,
            )

            # Check that both classes are present and have more than min_samples_per_class
            group_size = target_df.groupby("response").size()

            if len(group_size) != 2 or np.any(group_size < args.min_samples_per_class):
                continue

            print("Computing baseline for {} and {}".format(sp, dr))

            train_test_folds = dsplit.baseline_kfold_cv(target_df, cv=5)

            # Save splits to disk
            with open(
                os.path.join(join(out_folder, f"{sp}_{dr}_splits.pkl")), "wb"
            ) as handle:
                pickle.dump(train_test_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(
                os.path.join(join(out_folder, f"{sp}_{dr}_target.pkl")), "wb"
            ) as handle:
                pickle.dump(target_df, handle, protocol=pickle.HIGHEST_PROTOCOL)

            for n_fold, (train_data, test_data) in tqdm(enumerate(train_test_folds)):

                X_train = train_data.drop("response", axis=1)
                y_train = train_data["response"]
                X_test = test_data.drop("response", axis=1)
                y_test = test_data["response"]

                model = clone(model_template)
                model = make_pipeline(SMOTE(), StandardScaler(), model)

                skf = StratifiedKFold(n_splits=5)

                if args.n_random_iter > 1:
                    grid_search = RandomizedSearchCV(
                        model,
                        distributions,
                        scoring=scorers,
                        refit="f1_score",
                        cv=skf,
                        return_train_score=False,
                        n_jobs=3,
                        n_iter=args.n_random_iter,
                        verbose=0,
                        random_state=n_fold,
                    )

                    grid_search.fit(X_train.values, y_train.values)
                    config = grid_search.best_params_

                model = clone(model_template)
                model = make_pipeline(SMOTE(), StandardScaler(), model)

                if args.n_random_iter > 1:
                    model.set_params(**config)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                cm = confusion_matrix(y_test.values, y_pred)
                tpr, tnr, acc, precision = get_metrics(cm)
                f1 = f1_score(y_test.values, y_pred)
                mcc = matthews_corrcoef(y_test.values, y_pred)
                balanced_acc = balanced_accuracy_score(y_test.values, y_pred)

                roc_auc = 0.5
                auprc = 0

                if hasattr(model, "predict_proba"):
                    idx = np.where(model.classes_ == 1)[0]
                    y_proba = model.predict_proba(X_test)[:, idx]
                    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba, pos_label=1)
                    precisions, recall, pr_thresholds = precision_recall_curve(
                        y_test, y_proba
                    )
                    auprc = auc(recall, precisions)
                    roc_auc = roc_auc_score(y_test.values, y_proba)

                    # Produce ROC and PR curves to output
                    with open(
                        os.path.join(join(out_folder, f"{sp}_{dr}_pr.pkl")), "wb"
                    ) as handle:
                        pickle.dump(
                            [precisions, recall, pr_thresholds],
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    with open(
                        os.path.join(join(out_folder, f"{sp}_{dr}_roc.pkl")), "wb"
                    ) as handle:
                        pickle.dump(
                            [fpr, tpr, roc_thresholds],
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )

                split_result = {
                    "species": sp,
                    "drug": dr,
                    "fold": n_fold,
                    "precision": precision,
                    "recall": tpr,
                    "specificity": tnr,
                    "accuracy": acc,
                    "balanced_accuracy": balanced_acc,
                    "f1": f1,
                    "mcc": mcc,
                    "roc_auc": roc_auc,
                    "auprc": auprc,
                    "cmatrix": list(cm.flatten()),
                    "best_params": config,  # "n_test": len(y_pred),
                }
                baseline_results.append(split_result)

            baseline_results = pd.DataFrame(baseline_results)
            sp = sp.replace(" ", "_")
            baseline_results.to_csv(join(out_folder, f"{sp}_{dr}_metrics.csv"))

    print("Analysis complete")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="B", choices=["A", "B", "C", "D"]
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LogisticRegression",
        choices=["GBM", "MLP", "LogisticRegression"],
    )
    parser.add_argument(
        "--spectra_matrix_path",
        help="Specifies where to find the current spectral matrix",
        type=str,
        default="../data/DRIAMS-B/spectra_binned_6000_2018.csv",
    )
    parser.add_argument(
        "--output_folder",
        help="Specifies where to save the results",
        type=str,
        default="outputs/baselines",
    )
    parser.add_argument(
        "--threshold",
        help="Minimum number of samples per drug x species combination to include an experiment",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--n_random_iter",
        help="Numer of random search iterations for baseline hyperparameter tuning",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--min_samples_per_class",
        help="Minimum number of samples per class to use when training baseline models",
        type=int,
        default=10,
    )

    args = parser.parse_args()
    main(args)

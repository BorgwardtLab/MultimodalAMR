import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, balanced_accuracy_score
from sklearn.metrics import f1_score, fbeta_score, matthews_corrcoef, roc_auc_score, auc, precision_recall_curve
from scipy.stats import uniform
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
import sys
sys.path.insert(0,'../')


from models.data_utils import DataSplitter, get_metrics
import os
from os.path import join, exists
import json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace


def main(args):


    print("Loading data")
    data = pd.read_csv("../processed_data/DRIAMS_combined_long_table.csv")
    dsplit = DataSplitter(data, dataset=args.dataset)
    sorted_samples = sorted(dsplit.long_table["sample_id"].unique())
    idx2sample = {i: smp for i, smp in enumerate(sorted_samples)}
    sample2idx = {smp: i for i, smp in idx2sample.items()}
    counts = dsplit.long_table[["species", "drug"]].value_counts()
    experiments = sorted(list(counts[counts>args.threshold].index))
    print(len(experiments))
    sp, dr = experiments[args.n_experiment]
    sp = "Escherichia coli"
    dr = "Gentamicin"

    print("Loading spectra")
    spectra_matrix = np.load(args.spectra_matrix_path)
    spectra_matrix = pd.DataFrame(spectra_matrix, index=sorted_samples)

    scorers = {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'accuracy_score': make_scorer(accuracy_score),
        "balanced_accuracy_score": make_scorer(balanced_accuracy_score),
        'f1_score': make_scorer(f1_score),
        'mcc': make_scorer(matthews_corrcoef)
    }

    if args.model=="RF":
        model_template = RandomForestClassifier()
        distributions = {"n_estimators": [10, 50, 100, 200, 500, 1000], "max_depth": [2, 3, 5, 7, None], "max_features": ['auto', 'sqrt'], "min_samples_split": [2, 5, 10],
                        "criterion": ["gini", "entropy"]}

    elif args.model=="MLP":
        model_template = MLPClassifier()
        distributions = {"activation": ["tanh", "relu"], "alpha": np.logspace(-4, -1, 11), "hidden_layer_sizes": [(2048), (1024, 64), (2048, 512, 32)]}
    elif args.model=="LogisticRegression":
        model_template = LogisticRegression()
        distributions = {"class_weight": ["balanced", None], "C": uniform(loc=0.1, scale=9.9)}

    out_folder = join(args.output_folder, f"DRIAMS_{args.dataset}", args.model)
    if not exists(out_folder):
        os.makedirs(out_folder, exist_ok=True)

    with open(join(out_folder, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    

    print(f"Training {args.model} baseline")
    baseline_results = []
    # for seed, (sp, dr) in tqdm(enumerate(experiments)):
    target_df, _ = dsplit.baseline_selection(drug=dr, species=sp)
    target_df = pd.merge(target_df.set_index("sample_id")["response"], spectra_matrix, left_index=True, right_index=True)
    if np.sum(target_df["response"])<10:
        print("Too few positive samples")
        sys.exit(0)
    train_test_folds = dsplit.baseline_kfold_cv(target_df, cv=5)
    
    for n_fold, (train_data, test_data) in tqdm(enumerate(train_test_folds)):
        X_train = train_data.drop("response", axis=1)
        y_train = train_data["response"]
        X_test = test_data.drop("response", axis=1)
        y_test = test_data["response"]        
        # if np.sum(y_test)<3:
        #     continue
        
        model = clone(model_template)
        skf = StratifiedKFold(n_splits=5)
        # try:
        grid_search = RandomizedSearchCV(model, distributions, scoring=scorers, refit="f1_score",
                                            cv=skf, return_train_score=False, n_jobs=3, n_iter=args.n_random_iter, 
                                            verbose=0, random_state=n_fold)
        grid_search.fit(X_train.values, y_train.values)
        # except:
        #     continue
        # Evaluation of best configuration
        config = grid_search.best_params_
        model = clone(model_template)
        model.set_params(**config)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test.values, y_pred)
        cmat = pd.DataFrame(cm, index=["Label Negative", "Label Positive"], columns=["Pred. Negative", "Pred. Positive"])
        
        tpr, tnr, acc = get_metrics(cm)
        f1 = f1_score(y_test.values, y_pred)
        mcc = matthews_corrcoef(y_test.values, y_pred)
        balanced_acc = balanced_accuracy_score(y_test.values, y_pred)
        
        roc_auc = 0.5
        auprc = 0
        if hasattr(model, "predict_proba"):
            idx = np.where(model.classes_ == 1)[0]
            y_proba = model.predict_proba(X_test)[:, idx]
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
            auprc = auc(recall, precision)
            roc_auc = roc_auc_score(y_test.values, y_proba)

        split_result = {"species": sp, "drug": dr, "fold": n_fold, 
                        "recall": tpr, "specificity": tnr, "accuracy": acc, "balanced_accuracy": balanced_acc,
                    "f1": f1, "mcc": mcc, "roc_auc": roc_auc, "auprc": auprc, "cmatrix": list(cm.flatten()),
                    "best_params": config #"n_test": len(y_pred),
                    }
        baseline_results.append(split_result)

    baseline_results = pd.DataFrame(baseline_results)
    sp = sp.replace(" ", "_")
    baseline_results.to_csv(join(out_folder, f"{sp}_{dr}_metrics.csv"))
    print("Analysis complete")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="B", choices=["A", "B", "C", "D"])
    parser.add_argument("--model", type=str, default="LogisticRegression", choices=["RF", "MLP", "LogisticRegression"])
    parser.add_argument("--spectra_matrix_path", type=str, default="../data/DRIAMS-B/spectra_binned_6000_2018.npy")
    parser.add_argument("--output_folder", type=str, default="outputs/baselines")
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--n_random_iter", type=int, default=10)
    parser.add_argument("--n_experiment", type=int, default=0)

    args = parser.parse_args()
    main(args)
# Author Channi



# Currently under Construction - Use with Caution :P


# utilities
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterator, List, Dict, Tuple, Union
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from tqdm import tqdm
import pickle
import time 
from termcolor import colored 
import concurrent.futures
import argparse


# machine learning
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_validate, KFold, train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression, LogisticRegression


def _cross_validate(X_trans_, y, train_indices, test_indices, fold_id):
    global model

    model_init = model()
    X_trans_train = X_trans_.iloc[train_indices]
    X_trans_test = X_trans_.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    model_init.fit(X_trans_train, y_train)
    return model_init, train_indices, test_indices, X_trans_, y, fold_id


def _callback(future_job):
    global predictions_df
    global feature_importance
    global scores
    global train_scores
    global test_scores
    global target_name 

    model, train_indices, test_indices, X_trans_, y, fold_id = future_job.result()
    X_trans_train = X_trans_.iloc[train_indices]
    X_trans_test = X_trans_.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    y_pred = pd.DataFrame(model.predict(X_trans_test), columns=["Predicted Value"])
    temp = pd.concat([y_test.to_frame(name=target_name)\
                                .reset_index(drop=True),
                            y_pred], axis=1)
    predictions_df = pd.concat([predictions_df, temp],
                                    axis=0).reset_index(drop=True)
    
    if type(model).__name__.startswith("Logistic"):
        importances = model.coef_[0] 
    else:
        importances = model.feature_importances_
    
    coef_up = pd.DataFrame(importances, 
                        index=X_trans_.columns).reset_index()
    coef_up.columns = ["features", "importance"]
    coef_up.loc[:, "fold_id"] = fold_id

    feature_importance = pd.concat([feature_importance, coef_up],
                                        axis=0).reset_index(drop=True)

    train_scores.append(model.score(X_trans_train, y_train))
    test_scores.append(model.score(X_trans_test, y_test))

    scores["fold_id"].append(fold_id)
    scores["train_score"].append(train_scores[-1])
    scores["test_score"].append(test_scores[-1])

feature_importance = pd.DataFrame([], columns=["features", "importance", "fold_id"])
scores = defaultdict(list)
predictions_df = pd.DataFrame([])
train_scores = []
test_scores = []

def cross_validate(model, df: pd.DataFrame, target_name: str, n_splits: int = 10, shuffle: bool = True, max_workers: int = 8, delimiter: str = ","):
    
    global scores 
    global feature_importance
    global predictions_df
    global train_scores 
    global test_scores

    assert target_name in df, f"Invalid target '{target_name}'."
    assert isinstance(max_workers, int) and max_workers > 0

    cv = KFold(n_splits=n_splits, shuffle=shuffle)


    # predictions_df.columns = ["Predicted Value", target_name]

    X_trans_ = df.drop(columns=[target_name])
    y = df[target_name]

    print(colored("Initializing cross validation...", "blue"))
  
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        jobs = []
        for fold_id, (train_indices, test_indices) in tqdm(list(enumerate(cv.split(X_trans_, y)))):
            jobs.append(executor.submit(_cross_validate, 
                                        X_trans_=X_trans_, 
                                        y=y, 
                                        train_indices=train_indices, 
                                        test_indices=test_indices,
                                        fold_id=fold_id
                                        )
                            )
            jobs[-1].add_done_callback(_callback)
            
    train_scores = np.array(train_scores)
    test_scores = np.array(test_scores)
    print(train_scores.mean(), test_scores.mean())

    scores = pd.DataFrame(scores)
    return scores, predictions_df, feature_importance


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--target_name", type=str)
    parser.add_argument("--model", type=str, default="LogisticRegression")
    parser.add_argument("--task", choices=["classification", "regression"], type=str, default="classification")
    parser.add_argument("--shuffle", type=int, choices=[0, 1], default=1)
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--delimiter", type=str, choices=["\t", ","], default=",")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="results")
    

    args = parser.parse_args()
    data_path = args.data_path 
    target_name = args.target_name 
    task = args.task 
    shuffle = bool(args.shuffle) 
    n_splits = args.n_splits
    model = args.model 
    delimiter = args.delimiter 
    max_workers = args.max_workers 
    output = Path(args.output_dir).resolve()
    output.mkdir(exist_ok=True)
    
    print(colored("Parameters provided: ", "green"))
    for key, value in vars(args).items():
        print(colored(f"{key}: {value}", "green"))

    
    model_factory = {
            "classification": {
                    "RandomForest": RandomForestClassifier,
                    "LogisticRegression": LogisticRegression
                     },
            "regression": {
                "LinearRegression": LinearRegression,
                "RandomForest": RandomForestRegressor,
                }
            }

    model = model_factory[task][model]
    

    try:
        assert Path(data_path).is_file(), f"Invalid data path '{data_path}'."
        df = pd.read_csv(data_path, delimiter=delimiter)
    except (TypeError, AssertionError):
        print(colored("No path provided. Initializing test mode.", "red"))
        df = sns.load_dataset("titanic")[["survived", "age", "pclass"]].dropna().reset_index(drop=True)
        target_name = "survived"
    finally:
        print(colored(f"Dataset has been loaded succesfully.\nTarget={target_name}.", "blue"))
    
    then = time.perf_counter()
    scores, predictions_df, feature_importance = cross_validate(
                                                                model,
                                                                df, 
                                                                target_name, 
                                                                n_splits=n_splits, 
                                                                shuffle=shuffle, 
                                                                max_workers=max_workers, 
                                                                delimiter=delimiter
                                                                )
    now = time.perf_counter()
    print(colored(f"Process finished within {now-then:.2f} second(s).", "green"))
    print(colored("Saving outputs...", "green"))

    # save results 
    scores.to_csv(output.joinpath(f"model_{type(model).__name__}__cv_{n_splits}__scores.csv"), mode="w", index=False)
    predictions_df.to_csv(output.joinpath(f"model_{type(model).__name__}__cv_{n_splits}__predictions.csv"), index=False, mode="w"), 
    feature_importance.to_csv(output.joinpath(f"model_{type(model).__name__}__cv_{n_splits}__feature_importance.csv"), index=False, mode="w")
    
    print(colored("Process has been completed.", "green"))

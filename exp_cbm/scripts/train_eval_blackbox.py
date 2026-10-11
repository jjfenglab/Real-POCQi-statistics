"""
Black-box upper-bound baseline for the CBM pipeline.

Embeds the full `sentence` field (question + both answers, formatted the
same way the CBM pipeline consumes it) with a SentenceTransformer model
and fits a cross-validated L2 logistic regression on the train
partition. Evaluates on the test partition and writes a results CSV
whose columns match `evaluate_bayesian.py` so the two methods can be
concatenated downstream.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--in-dataset-file", required=True,
                   help="pairs.csv with columns [sentence, y, ...]")
    p.add_argument("--indices-csv", required=True,
                   help="indices.csv with columns [idx, partition]")
    p.add_argument("--embedding-model-name", default="all-MiniLM-L6-v2")
    p.add_argument("--method-name", default="blackbox_embedding")
    p.add_argument("--n-bootstraps", type=int, default=1000)
    p.add_argument("--out-model", required=True,
                   help="Pickle of the fitted LR for reuse.")
    p.add_argument("--result-csv-file", required=True)
    p.add_argument("--log-file", required=True)
    return p.parse_args()


def load_partition(dataset_file: str, indices_file: str, partition: str) -> pd.DataFrame:
    dset = pd.read_csv(dataset_file)
    idx_df = pd.read_csv(indices_file)
    idxs = idx_df.loc[idx_df.partition == partition, "idx"].to_numpy()
    return dset.iloc[idxs].reset_index(drop=True)


def bootstrap_ci(y_true, y_pred, eval_func, n_bootstraps: int, rng: np.random.Generator):
    """Mirrors evaluate_bayesian.get_bootstrap_ci so result columns are comparable."""
    point = eval_func(y_true, y_pred)
    scores = []
    n = len(y_pred)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, n)
        if np.unique(y_true[idx]).size < 2:
            continue
        try:
            scores.append(eval_func(y_true[idx], y_pred[idx]))
        except Exception:
            continue
    if not scores:
        return point, (np.nan, np.nan)
    return point, tuple(np.quantile(scores, [0.025, 0.975]))


def fmt(res) -> str:
    point, (lo, hi) = res
    return f"{point:.3f} ({lo:.3f}, {hi:.3f})"


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.searchsorted(bins[1:-1], y_prob)
    bin_sums = np.bincount(binids, weights=y_prob, minlength=n_bins)
    bin_true = np.bincount(binids, weights=y_true.astype(float), minlength=n_bins)
    bin_total = np.bincount(binids, minlength=n_bins)
    nonzero = bin_total > 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    return float(
        np.sum(np.abs(prob_true - prob_pred) * bin_total[nonzero] / bin_total.sum())
    )


def main():
    args = parse_args()
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file)],
    )
    logging.info("args: %s", args)

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_df = load_partition(args.in_dataset_file, args.indices_csv, "train")
    test_df = load_partition(args.in_dataset_file, args.indices_csv, "test")
    logging.info("train=%d test=%d", len(train_df), len(test_df))

    y_train = train_df["y"].to_numpy().astype(int)
    y_test = test_df["y"].to_numpy().astype(int)

    model = SentenceTransformer(args.embedding_model_name, local_files_only=True)
    logging.info("encoding %d train sentences", len(train_df))
    train_emb = model.encode(train_df["sentence"].tolist(), show_progress_bar=True)
    logging.info("encoding %d test sentences", len(test_df))
    test_emb = model.encode(test_df["sentence"].tolist(), show_progress_bar=True)
    logging.info("train_emb shape %s", train_emb.shape)

    lr = LogisticRegressionCV(
        cv=5,
        penalty="l2",
        max_iter=10000,
        Cs=20,
        n_jobs=-1,
        random_state=args.seed,
    )
    lr.fit(train_emb, y_train)

    train_prob = lr.predict_proba(train_emb)[:, 1]
    test_prob = lr.predict_proba(test_emb)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    train_auc = roc_auc_score(y_train, train_prob)
    logging.info("train AUC (in-sample): %.3f", train_auc)

    acc = bootstrap_ci(y_test, test_prob, lambda y, p: np.mean((p >= 0.5).astype(int) == y),
                       args.n_bootstraps, rng)
    auc = bootstrap_ci(y_test, test_prob, roc_auc_score, args.n_bootstraps, rng)
    brier = bootstrap_ci(y_test, test_prob, lambda y, p: np.mean((p - y) ** 2),
                         args.n_bootstraps, rng)
    ece = bootstrap_ci(y_test, test_prob, expected_calibration_error,
                       args.n_bootstraps, rng)
    eps = 1e-7
    log_lik = bootstrap_ci(
        y_test, test_prob,
        lambda y, p: np.mean(y * np.log(np.clip(p, eps, 1 - eps))
                             + (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))),
        args.n_bootstraps, rng,
    )
    f1 = bootstrap_ci(y_test, test_prob,
                      lambda y, p: f1_score(y, (p >= 0.5).astype(int), average="macro"),
                      args.n_bootstraps, rng)

    for name, res in [("acc", acc), ("auc", auc), ("brier", brier),
                      ("ece", ece), ("log_lik", log_lik), ("f1", f1)]:
        logging.info("test %s: %s", name, fmt(res))

    res_df = pd.DataFrame({
        "acc": [acc[0]], "acc_str": [fmt(acc)],
        "auc": [auc[0]], "auc_str": [fmt(auc)],
        "brier": [brier[0]], "brier_str": [fmt(brier)],
        "ece": [ece[0]], "ece_str": [fmt(ece)],
        "log_lik": [log_lik[0]], "log_lik_str": [fmt(log_lik)],
        "f1": [f1[0]], "f1_str": [fmt(f1)],
        "method": [args.method_name],
        "seed": [args.seed],
        "embedding_model": [args.embedding_model_name],
        "n_train": [len(train_df)],
        "n_test": [len(test_df)],
    })
    Path(args.result_csv_file).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(args.result_csv_file, index=False)

    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_model, "wb") as f:
        pickle.dump({"lr": lr, "embedding_model_name": args.embedding_model_name}, f)


if __name__ == "__main__":
    main()

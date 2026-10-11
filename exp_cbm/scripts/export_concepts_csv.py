"""
Export the concepts learned by BC-LLM along with their posterior frequency and
fitted linear-model coefficients. Rows are one (iteration, concept) pair over
the last `--num-posterior-iters` MH iterations of the training history.
"""
import argparse
import sys
import os
import logging

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.getcwd(), "bc-llm-head-to-head"))
from src.training_history import TrainingHistory


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training-history-file", type=str, required=True)
    p.add_argument("--num-posterior-iters", type=int, default=1,
                   help="Number of trailing iterations to summarize.")
    p.add_argument("--out-csv", type=str, required=True)
    p.add_argument("--log-file", type=str, default="_output/log_export_concepts.txt")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.INFO)

    history = TrainingHistory.load(args.training_history_file)
    # The first entry in _concepts is the initial proposal that has no
    # fitted coefficients yet — training adds coefs/intercepts only from
    # the first MH iteration onward, so _concepts is one longer than _coefs.
    num_fitted = len(history._coefs)
    concept_offset = len(history._concepts) - num_fitted
    logging.info("Loaded history: %d concept sets, %d fitted iters, offset %d",
                 len(history._concepts), num_fitted, concept_offset)

    start_iter = max(0, num_fitted - args.num_posterior_iters)

    rows = []
    for i in range(start_iter, num_fitted):
        concept_dicts = history._concepts[i + concept_offset]
        coefs = np.asarray(history._coefs[i]).reshape(-1)
        intercept = history._intercepts[i]
        if np.ndim(intercept) > 0:
            intercept = np.asarray(intercept).reshape(-1)[0]

        for j, cd in enumerate(concept_dicts):
            coef = float(coefs[j]) if j < len(coefs) else np.nan
            rows.append({
                "iter": i,
                "concept_index": j,
                "concept": cd.get("concept"),
                "coef": coef,
                "intercept": float(intercept),
            })

    long_df = pd.DataFrame(rows)

    # Posterior summary: average coefficient per unique concept across the
    # last iters, plus how often each concept was sampled.
    denom = max(1, num_fitted - start_iter)
    summary = (
        long_df.groupby("concept")
        .agg(mean_coef=("coef", "mean"),
             std_coef=("coef", "std"),
             n_appearances=("coef", "size"))
        .reset_index()
    )
    summary["posterior_freq"] = summary["n_appearances"] / denom
    summary = summary.sort_values("mean_coef", key=np.abs, ascending=False)

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    summary.to_csv(args.out_csv, index=False)
    long_df.to_csv(args.out_csv.replace(".csv", "_per_iter.csv"), index=False)

    logging.info("Wrote %d summarized concepts to %s", len(summary), args.out_csv)


if __name__ == "__main__":
    main()

"""
Assemble the clinical-QA head-to-head rating data into a pair CSV
consumable by the forked bc-llm trainer.

Each original rating produces TWO rows: the original pair and a swap-
augmented sibling with A and B exchanged and the label flipped. Both
rows carry the same `qa_assignment_id`, which the trainer uses as the
cluster key to keep siblings on the same side of the partial-posterior
split.

Eligibility filters (prior users, subversion, test emails, completion,
etc.) are delegated to `src.judge_answers.build_eligible_assignments`
so this pipeline sees exactly the same set of ratings as the LLM-as-a-
judge gold standard.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the gold-standard loading + eligibility logic.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.judge_answers import build_eligible_assignments, load_data

# Choices from the rating UI → sign of the preference for answer A.
CHOICE_TO_LABEL = {
    "strongly_a": 0,
    "slightly_a": 0,
    "tie": 1,
    "slightly_b": 2,
    "strongly_b": 2,
}
LABEL_FLIP = {0: 2, 1: 1, 2: 0}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True,
                   help="Directory containing ratings.csv, qa_assignments.csv, "
                        "qa_questions.csv, users.csv (and optionally "
                        "blinded_prior_users.csv, recruitment_combined.csv).")
    p.add_argument("--answers-dir", required=True,
                   help="Directory containing answers_<variant>.jsonl.")
    p.add_argument("--variant", required=True,
                   help="Answer variant, e.g. 'text_only' or 'text_with_citations'.")
    p.add_argument("--axis", required=True,
                   help="Rating axis to filter to (e.g. clinical_utility)")
    p.add_argument("--max-n", type=int, default=0,
                   help="Max number of original ratings to keep (0 = all). "
                        "Augmentation doubles the output row count.")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for shuffling before --max-n and for the "
                        "train/test cluster split")
    p.add_argument("--user-stratification", default=None,
                   help="Optional OE Status filter ('On OE' or 'Off OE'); "
                        "requires recruitment_combined.csv in --data-dir.")
    p.add_argument("--apply-submission-time-filter", action="store_true",
                   help="Drop ratings with submission_time_seconds < 10. "
                        "Off by default (matches judge_answers.py).")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--out-indices-csv", required=True,
                   help="Train/test partition CSV grouped by qa_assignment_id "
                        "so augmented siblings stay together.")
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--log", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log)],
        force=True,
    )

    # Load everything via the gold-standard loader so we see exactly the
    # same files judge_answers.py uses.
    (ratings, assignments, questions, users, answers_by_key,
     prior_users, user_registrations) = load_data(
        args.data_dir, args.answers_dir, args.variant
    )
    logging.info(
        "loaded %d ratings, %d assignments, %d questions, %d users, %d answers",
        len(ratings), len(assignments), len(questions), len(users), len(answers_by_key),
    )

    # Apply the same inclusion criteria as judge_answers.py.
    eligible_assignments = build_eligible_assignments(
        ratings, assignments, users,
        prior_users=prior_users,
        user_registrations=user_registrations,
        user_stratification=args.user_stratification,
        apply_submission_time_filter=args.apply_submission_time_filter,
        variant=args.variant,
    )
    logging.info("eligible assignments: %d (unique questions: %d)",
                 len(eligible_assignments),
                 eligible_assignments["question_id"].nunique())

    # Filter ratings to the chosen axis AND to eligible assignments.
    axis_ratings = ratings[ratings["axis"] == args.axis].copy()
    logging.info("after axis=%s filter: %d ratings", args.axis, len(axis_ratings))

    eligible_ids = set(eligible_assignments["id"].tolist())
    axis_ratings = axis_ratings[axis_ratings["qa_assignment_id"].isin(eligible_ids)]
    logging.info("after eligibility filter: %d ratings", len(axis_ratings))

    axis_ratings["y"] = axis_ratings["choice"].map(CHOICE_TO_LABEL)
    assert axis_ratings["y"].notna().all(), (
        "unmapped choice values: "
        f"{sorted(axis_ratings.loc[axis_ratings['y'].isna(), 'choice'].unique())}"
    )

    # Join in question text + assignment metadata.
    df = (
        axis_ratings[["id", "qa_assignment_id", "y"]]
        .rename(columns={"id": "rating_id"})
        .merge(
            eligible_assignments[["id", "user_id", "question_id",
                                  "slot_a_provider", "slot_b_provider"]]
            .rename(columns={"id": "qa_assignment_id"}),
            on="qa_assignment_id", how="inner",
        )
        .merge(
            questions.rename(columns={"id": "question_id"})[["question_id", "question_text"]],
            on="question_id", how="inner",
        )
    )
    logging.info("after joins: %d rows", len(df))

    df["text_a"] = df.apply(
        lambda r: answers_by_key.get((r["question_id"], r["slot_a_provider"])), axis=1
    )
    df["text_b"] = df.apply(
        lambda r: answers_by_key.get((r["question_id"], r["slot_b_provider"])), axis=1
    )
    missing = df["text_a"].isna() | df["text_b"].isna()
    if missing.any():
        logging.warning("dropping %d rows with missing answer text", missing.sum())
        df = df[~missing].reset_index(drop=True)

    rng = np.random.default_rng(args.seed)
    unique_clusters = np.asarray(df["qa_assignment_id"].unique(), dtype=object)
    rng.shuffle(unique_clusters)
    n_test_clusters = int(round(args.test_frac * len(unique_clusters)))
    test_clusters = set(unique_clusters[:n_test_clusters])
    df["partition"] = np.where(df["qa_assignment_id"].isin(test_clusters), "test", "train")
    logging.info("pre-subsample train/test split: %s",
                 df["partition"].value_counts().to_dict())

    if args.max_n and args.max_n < len(df):
        df = df.sample(n=args.max_n, random_state=args.seed).reset_index(drop=True)
        logging.info("subsampled to %d rows", len(df))

    df["rater_id"] = df["user_id"]

    def pack_sentence(q: str, a: str, b: str) -> str:
        return (
            f"Clinical question:\n{q}\n\n"
            f"Answer A:\n{a}\n\n"
            f"Answer B:\n{b}"
        )

    original = pd.DataFrame({
        "qa_assignment_id": df["qa_assignment_id"].values,
        "question_id": df["question_id"].values,
        "rater_id": df["rater_id"].values,
        "question_text": df["question_text"].values,
        "text_a": df["text_a"].values,
        "text_b": df["text_b"].values,
        "sentence": [pack_sentence(q, a, b) for q, a, b in
                     zip(df["question_text"], df["text_a"], df["text_b"])],
        "y": df["y"].astype(int).values,
        "is_augmented": False,
        "partition": df["partition"].values,
    })
    swapped = pd.DataFrame({
        "qa_assignment_id": df["qa_assignment_id"].values,
        "question_id": df["question_id"].values,
        "rater_id": df["rater_id"].values,
        "question_text": df["question_text"].values,
        "text_a": df["text_b"].values,
        "text_b": df["text_a"].values,
        "sentence": [pack_sentence(q, b, a) for q, a, b in
                     zip(df["question_text"], df["text_a"], df["text_b"])],
        "y": df["y"].astype(int).map(LABEL_FLIP).values,
        "is_augmented": True,
        "partition": df["partition"].values,
    })

    out = pd.concat([original, swapped], ignore_index=True)
    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    logging.info("final pairs: %d rows (%d originals + %d augmented)",
                 len(out), len(original), len(swapped))
    logging.info("label distribution: %s", out["y"].value_counts().to_dict())

    partitions = out["partition"].values
    out = out.drop(columns=["partition"])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=True, index_label="row_id")

    indices_df = pd.DataFrame({"idx": np.arange(len(out)), "partition": partitions})
    indices_df = indices_df.sort_values("partition").reset_index(drop=True)
    Path(args.out_indices_csv).parent.mkdir(parents=True, exist_ok=True)
    indices_df.to_csv(args.out_indices_csv, index=True)
    logging.info("train/test split: %s", indices_df["partition"].value_counts().to_dict())


if __name__ == "__main__":
    sys.exit(main())

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
from typing import Set

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.judge_answers import build_eligible_assignments, load_data

# Choices from the rating UI → sign of the preference for answer A.
CHOICE_TO_LABEL_EXTREMES = {
    "strongly_a": 0,
    "strongly_b": 2,
}
CHOICE_TO_LABEL = {
    "strongly_a": 0,
    "slightly_a": 0,
    "tie": 1,
    "slightly_b": 2,
    "strongly_b": 2,
}
LABEL_FLIP = {0: 2, 1: 1, 2: 0}
LABEL_FLIP_BINARY = {0: 1, 1: 0}


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
    p.add_argument("--drop-ties", action="store_true", help="exclude ties")
    p.add_argument("--only-extremes", action="store_true", help="only including strong preferences")
    p.add_argument("--augment", dest="augment", action="store_true",
                   help="Emit a swap-augmented sibling (A↔B, label flipped) "
                        "for every original rating. On by default.")
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Disable swap augmentation; output originals only.")
    p.set_defaults(augment=True)
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


def setup_logging(log_path: str) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )


def assign_test_clusters(
    eligible_assignments: pd.DataFrame,
    test_frac: float,
    seed: int,
) -> Set:
    """Pick test clusters over the full eligible pool so the split is
    axis-independent (maximizes LLM-cache reuse across axes)."""
    rng = np.random.default_rng(seed)
    all_clusters = np.asarray(
        np.sort(eligible_assignments["id"].unique()), dtype=object
    )
    rng.shuffle(all_clusters)
    n_test = int(round(test_frac * len(all_clusters)))
    return set(all_clusters[:n_test])


def filter_ratings_to_axis(
    ratings: pd.DataFrame,
    eligible_assignments: pd.DataFrame,
    axis: str,
    mapping_dict: dict,
) -> pd.DataFrame:
    axis_ratings = ratings[ratings["axis"] == axis].copy()
    logging.info("after axis=%s filter: %d ratings", axis, len(axis_ratings))

    eligible_ids = set(eligible_assignments["id"].tolist())
    axis_ratings = axis_ratings[axis_ratings["qa_assignment_id"].isin(eligible_ids)]
    logging.info("after eligibility filter: %d ratings", len(axis_ratings))

    axis_ratings = axis_ratings[axis_ratings.choice.isin(list(mapping_dict.keys()))]

    axis_ratings["y"] = axis_ratings["choice"].map(mapping_dict)
    assert axis_ratings["y"].notna().all(), (
        "unmapped choice values: "
        f"{sorted(axis_ratings.loc[axis_ratings['y'].isna(), 'choice'].unique())}"
    )
    return axis_ratings


def join_rating_context(
    axis_ratings: pd.DataFrame,
    eligible_assignments: pd.DataFrame,
    questions: pd.DataFrame,
    answers_by_key: dict,
) -> pd.DataFrame:
    """Attach question text, slot providers, and resolved A/B answer text."""
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
    return df


def apply_drop_ties(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["y"] != 1].copy()
    # Collapse {0, 2} → {0, 1} so the label space is binary.
    df["y"] = (df["y"] // 2).astype(int)
    logging.info("drop-ties: %d → %d rows (binary labels); split now %s",
                 before, len(df), df["partition"].value_counts().to_dict())
    return df


def subsample(df: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    df = df.sample(n=len(df), random_state=seed).reset_index(drop=True)
    df = df.iloc[:max_n]
    logging.info("subsampled to %d rows", len(df))
    return df


def pack_sentence(q: str, a: str, b: str) -> str:
    return (
        f"Clinical question:\n{q}\n\n"
        f"Answer A:\n{a}\n\n"
        f"Answer B:\n{b}"
    )


def build_original_rows(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
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


def build_swapped_rows(df: pd.DataFrame, drop_ties: bool) -> pd.DataFrame:
    label_flip = LABEL_FLIP_BINARY if drop_ties else LABEL_FLIP
    return pd.DataFrame({
        "qa_assignment_id": df["qa_assignment_id"].values,
        "question_id": df["question_id"].values,
        "rater_id": df["rater_id"].values,
        "question_text": df["question_text"].values,
        "text_a": df["text_b"].values,
        "text_b": df["text_a"].values,
        "sentence": [pack_sentence(q, b, a) for q, a, b in
                     zip(df["question_text"], df["text_a"], df["text_b"])],
        "y": df["y"].astype(int).map(label_flip).values,
        "is_augmented": True,
        "partition": df["partition"].values,
    })


def log_inter_rater_agreement(df: pd.DataFrame) -> None:
    """Sanity-check inter-rater agreement for the same (question, {provider_a,
    provider_b}) set, canonicalizing A/B order by flipping labels when the
    providers appear in the opposite slot."""
    work = df[["qa_assignment_id", "question_id", "user_id",
               "slot_a_provider", "slot_b_provider", "y"]].copy()

    # Canonical provider ordering: (min, max). If the raw (a, b) is reversed
    # relative to canonical, flip the label.
    providers = np.sort(
        work[["slot_a_provider", "slot_b_provider"]].to_numpy(), axis=1
    )
    work["provider_lo"] = providers[:, 0]
    work["provider_hi"] = providers[:, 1]
    is_flipped = work["slot_a_provider"].values != work["provider_lo"].values

    y_vals = work["y"].astype(int).values
    unique_labels = set(np.unique(y_vals).tolist())
    if unique_labels <= {0, 1}:
        flip_map = LABEL_FLIP_BINARY
    else:
        flip_map = LABEL_FLIP
    work["y_canon"] = np.where(
        is_flipped,
        pd.Series(y_vals).map(flip_map).values,
        y_vals,
    )

    group_cols = ["question_id", "provider_lo", "provider_hi"]
    sizes = work.groupby(group_cols).size()
    multi = sizes[sizes > 1]
    if multi.empty:
        logging.info("inter-rater agreement: no (question, provider-pair) "
                     "groups with >1 rating")
        return

    multi_groups = work.merge(
        multi.rename("n_ratings").reset_index(), on=group_cols, how="inner"
    )

    # Pairwise agreement within each group: fraction of ordered rater pairs
    # (i != j) with matching canonical labels.
    agree_num = 0
    agree_den = 0
    for _, g in multi_groups.groupby(group_cols):
        labels = g["y_canon"].to_numpy()
        n = len(labels)
        same = (labels[:, None] == labels[None, :]).sum() - n  # exclude diagonal
        agree_num += same
        agree_den += n * (n - 1)

    pct = 100.0 * agree_num / agree_den if agree_den else float("nan")
    logging.info(
        "inter-rater agreement (A/B-swap canonicalized): %.1f%% "
        "pairwise agreement across %d (question, provider-pair) groups "
        "with >1 rating (%d ratings total)",
        pct, len(multi), len(multi_groups),
    )


def write_outputs(out: pd.DataFrame, out_csv: str, out_indices_csv: str) -> None:
    partitions = out["partition"].values
    out = out.drop(columns=["partition"])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=True, index_label="row_id")

    indices_df = pd.DataFrame({"idx": np.arange(len(out)), "partition": partitions})
    indices_df = indices_df.sort_values("partition").reset_index(drop=True)
    Path(out_indices_csv).parent.mkdir(parents=True, exist_ok=True)
    indices_df.to_csv(out_indices_csv, index=True)
    logging.info("train/test split: %s", indices_df["partition"].value_counts().to_dict())


def main():
    args = parse_args()
    setup_logging(args.log)

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

    test_clusters = assign_test_clusters(
        eligible_assignments, args.test_frac, args.seed
    )

    axis_ratings = filter_ratings_to_axis(
        ratings,
        eligible_assignments,
        args.axis,
        mapping_dict=CHOICE_TO_LABEL_EXTREMES if args.only_extremes else CHOICE_TO_LABEL)
    df = join_rating_context(axis_ratings, eligible_assignments, questions, answers_by_key)

    df["partition"] = np.where(df["qa_assignment_id"].isin(test_clusters), "test", "train")
    logging.info("pre-subsample train/test split: %s",
                 df["partition"].value_counts().to_dict())

    if args.drop_ties:
        df = apply_drop_ties(df)

    df = subsample(df, args.max_n, args.seed)
    df["rater_id"] = df["user_id"]

    original = build_original_rows(df)
    if args.augment:
        swapped = build_swapped_rows(df, args.drop_ties)
        out = pd.concat([original, swapped], ignore_index=True)
    else:
        swapped = original.iloc[0:0]
        out = original

    out = out.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    logging.info("final pairs: %d rows (%d originals + %d augmented)",
                 len(out), len(original), len(swapped))
    logging.info("label distribution: %s", out["y"].value_counts().to_dict())

    log_inter_rater_agreement(df)

    write_outputs(out, args.out_csv, args.out_indices_csv)


if __name__ == "__main__":
    sys.exit(main())

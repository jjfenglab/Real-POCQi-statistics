"""LLM-as-a-judge evaluation for clinical Q&A answers.

Replays the human Q&A preference pairs to LLM graders using higher reasoning effort.
This is a Python reimplementation of the original TypeScript machineRate.ts.

Usage:
    python src/judge_answers.py \
        --data-dir data_6-21-3-56-00pm-nohb \
        --answers-dir public_release_2026_09_22 \
        --output _output/judge_ratings.jsonl \
        --log _output/judge.log \
        --judge claude-opus-4-8 \
        --variant text_only \
        --batch-size 10 \
        --limit 5
"""
import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Add llm-api to path
sys.path.insert(0, str(Path(__file__).parent.parent / "llm-api"))

from lab_llm import LLMApi, wrap_completion_function, CachingCompletion
from lab_llm.versa.openai import make_versa_openai_completion
from lab_llm.versa.claude import make_versa_claude_completion
import litellm

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

RATING_AXES = [
    "clinical_utility",
    "accuracy",
    "authoritativeness",
    "verifiability",
    "completeness",
]

RATING_CHOICES = [
    "strongly_a",
    "slightly_a",
    "tie",
    "slightly_b",
    "strongly_b",
]

RatingChoice = Literal["strongly_a", "slightly_a", "tie", "slightly_b", "strongly_b"]

# Axis prompts (matching the human study)
AXIS_PROMPTS = {
    "clinical_utility": "How useful is the answer for clinical decision-making?",
    "accuracy": "How accurate is the medical information provided?",
    "authoritativeness": "How authoritative and trustworthy does the answer appear?",
    "verifiability": "How well does the answer support claims with verifiable sources?",
    "completeness": "How complete and thorough is the answer?",
}

CHOICE_GLOSS = {
    "strongly_a": "Answer A is much better",
    "slightly_a": "Answer A is somewhat better",
    "tie": "about equal",
    "slightly_b": "Answer B is somewhat better",
    "strongly_b": "Answer B is much better",
}

# -----------------------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------------------

# TODO: Update these model mappings when llm-api supports the exact versions
GRADER_MODELS = {
    "claude-opus-4-8": "anthropic/claude-opus-4-8",
    "gemini-3.1-pro": "gemini/gemini-3.1-pro-preview",
    "gpt-5.5": "openai/gpt-5.5-2026-04-23",
}

GRADER_LABELS = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "gemini-3.1-pro": "Gemini 3.1 Pro",
    "gpt-5.5": "GPT-5.5",
}

# Per-model reasoning configuration (default reasoning modes)
# - Claude: extended thinking with "enabled" type uses adaptive thinking (model decides depth)
# - GPT: reasoning_effort "medium" is the default for reasoning models
# - Gemini: thinking_config with thinking_budget=0 means dynamic (model decides)
GRADER_REASONING_PARAMS = {
    "claude-opus-4-8": {
        # Adaptive extended thinking - model decides how much thinking based on task
        # See: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
        "thinking": {"type": "adaptive"},
    },
    "gemini-3.1-pro": {
        # Dynamic thinking - model decides thinking depth
        # See: https://ai.google.dev/gemini-api/docs/thinking
        # thinking_config = on (high) by default
    },
    "gpt-5.5": {
        # Medium reasoning effort is the default
        # See: https://platform.openai.com/docs/guides/reasoning
        # "reasoning_effort": "medium",
    },
}

VARIANTS = ["text_only"] #, "text_with_citations"]

# -----------------------------------------------------------------------------
# Pydantic Response Model
# -----------------------------------------------------------------------------

class JudgeRating(BaseModel):
    """Structured output for LLM judge ratings on all 5 axes."""

    clinical_utility: RatingChoice = Field(
        description="Rating for clinical utility axis"
    )
    accuracy: RatingChoice = Field(
        description="Rating for accuracy axis"
    )
    authoritativeness: RatingChoice = Field(
        description="Rating for authoritativeness axis"
    )
    verifiability: RatingChoice = Field(
        description="Rating for verifiability axis"
    )
    completeness: RatingChoice = Field(
        description="Rating for completeness axis"
    )


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert US physician serving as a blinded evaluator in a head-to-head comparison of two AI-generated answers to a clinical question. You will see the question and two answers, labelled "Answer A" and "Answer B". You do not know which system wrote either answer; judge only their content.

Rate which answer is better on EACH of these five axes independently:
- clinical_utility: How useful is the answer for clinical decision-making?
- accuracy: How accurate is the medical information provided?
- authoritativeness: How authoritative and trustworthy does the answer appear?
- verifiability: How well does the answer support claims with verifiable sources?
- completeness: How complete and thorough is the answer?

For each axis choose exactly one option from this 5-point scale:
- "strongly_a": Answer A is much better
- "slightly_a": Answer A is somewhat better
- "tie": about equal
- "slightly_b": Answer B is somewhat better
- "strongly_b": Answer B is much better

Use the full scale; reserve "tie" for genuinely indistinguishable answers.
Reply with ONLY a JSON object whose keys are the five axis names above and whose values are the chosen scale option strings, e.g.:
{"clinical_utility":"slightly_a","accuracy":"tie","authoritativeness":"strongly_b","verifiability":"slightly_b","completeness":"slightly_a"}"""


def build_user_prompt(question: str, answer_a: str, answer_b: str) -> str:
    """Build the user prompt with question and both answers."""
    return f"""# Clinical question
{question}

# Answer A
{answer_a}

# Answer B
{answer_b}

Return only the JSON object."""




# -----------------------------------------------------------------------------
# Constants for Inclusion Criteria (matching R data_helpers.R)
# -----------------------------------------------------------------------------

MIN_SUBMISSION_TIME_SECONDS = 10
TEST_EMAIL_PATTERNS = [r"eval\.test$", r"example\.com$", r"openevidence\.com$"]
ELIGIBLE_SUBVERSIONS = ["qa_text_only", "qa_text_citations"]

# Map answer-variant label -> user subversion label
VARIANT_TO_SUBVERSION = {
    "text_only": "qa_text_only",
    "text_with_citations": "qa_text_citations",
}


# -----------------------------------------------------------------------------
# Data Loading and Preprocessing (matching R analysis.Rmd logic)
# -----------------------------------------------------------------------------

def get_eligible_users(
    users: pd.DataFrame,
    user_registrations: Optional[pd.DataFrame] = None,
    user_stratification: Optional[str] = None,
    variant: Optional[str] = None,
) -> pd.DataFrame:
    """Filter users to eligible ones for the experiment.

    Mirrors R get_eligible_users() in data_helpers.R:
    - Filter to subversion in ["qa_text_only", "qa_text_citations"]
    - If `variant` is given, restrict further to the matching subversion
      (text_only -> qa_text_only, text_with_citations -> qa_text_citations)
    - Exclude test email patterns
    - Optionally filter by OE Status stratification
    """
    # Filter to experiment subversions (optionally restricted to the active variant)
    if variant is not None:
        if variant not in VARIANT_TO_SUBVERSION:
            raise ValueError(
                f"Unknown variant: {variant}. Available: {list(VARIANT_TO_SUBVERSION)}"
            )
        allowed_subversions = [VARIANT_TO_SUBVERSION[variant]]
    else:
        allowed_subversions = ELIGIBLE_SUBVERSIONS
    eligible = users[users["subversion"].isin(allowed_subversions)].copy()

    # Exclude test email patterns
    test_pattern = "|".join(TEST_EMAIL_PATTERNS)
    is_test_email = eligible["email"].str.contains(test_pattern, case=False, regex=True)
    n_excluded = is_test_email.sum()
    if n_excluded > 0:
        logging.info(
            f"  Excluded {n_excluded} users with test/example email addresses"
        )
    eligible = eligible[~is_test_email]

    # Optional stratification by OE Status
    if user_stratification and user_registrations is not None:
        eligible_emails = user_registrations[
            user_registrations["OE Status"] == user_stratification
        ]["Email"].tolist()
        eligible = eligible[eligible["email"].isin(eligible_emails)]

    return eligible


def build_eligible_assignments(
    ratings: pd.DataFrame,
    assignments: pd.DataFrame,
    users: pd.DataFrame,
    prior_users: Optional[pd.DataFrame] = None,
    user_registrations: Optional[pd.DataFrame] = None,
    user_stratification: Optional[str] = None,
    apply_submission_time_filter: bool = False,
    variant: Optional[str] = None,
) -> pd.DataFrame:
    """Build the set of eligible assignments for judging.

    Mirrors R build_qa_ratings() in data_helpers.R:
    - Exclude prior users
    - Filter to eligible users (by subversion, excluding test emails)
    - Filter to ratings with valid qa_assignment_id
    - Optionally apply submission time filter (>= 10 seconds)
    - Return unique assignments (not individual ratings)

    Note: For LLM-as-a-judge, submission time filter is typically disabled since
    it's designed to filter out rushed human ratings.
    """
    # Exclude prior users if provided
    if prior_users is not None and len(prior_users) > 0:
        prior_user_ids = set(prior_users["id"].tolist())
        n_prior = assignments["user_id"].isin(prior_user_ids).sum()
        if n_prior > 0:
            logging.info(f"  Dropped {n_prior} assignments from prior users")
        assignments = assignments[~assignments["user_id"].isin(prior_user_ids)]

    # Get eligible users
    eligible_users = get_eligible_users(
        users,
        user_registrations,
        user_stratification,
        variant=variant #(if you want to restrict grading to exactly the QAs that were judged by a human)
    )
    eligible_user_ids = set(eligible_users["id"].tolist())

    # Filter ratings to those with valid qa_assignment_id
    valid_ratings = ratings[ratings["qa_assignment_id"].notna()].copy()

    # Optionally apply submission time filter (inclusion criteria from human analysis)
    if apply_submission_time_filter:
        n_before = len(valid_ratings)
        valid_ratings = valid_ratings[
            valid_ratings["submission_time_seconds"].notna() &
            (valid_ratings["submission_time_seconds"] >= MIN_SUBMISSION_TIME_SECONDS)
        ]
        n_after = len(valid_ratings)
        logging.info(f"  Applied submission time filter: {n_before} -> {n_after} ratings")

    # Get unique assignment IDs that have valid ratings
    valid_assignment_ids = set(valid_ratings["qa_assignment_id"].unique())

    # Filter assignments to:
    # 1. Completed assignments (completed_at not null)
    # 2. From eligible users
    # 3. Have valid ratings
    eligible_assignments = assignments[
        assignments["completed_at"].notna() &
        assignments["user_id"].isin(eligible_user_ids) &
        assignments["id"].isin(valid_assignment_ids)
    ].copy()

    # Add subversion from users
    user_subversion = eligible_users.set_index("id")["subversion"].to_dict()
    eligible_assignments["subversion"] = eligible_assignments["user_id"].map(user_subversion)

    return eligible_assignments


def load_data(
    data_dir: str,
    answers_dir: str,
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load all required data files.

    Args:
        data_dir: Directory containing assignment/user data (e.g., data_6-21-3-56-00pm-nohb)
        answers_dir: Directory containing pre-cleaned answer JSONL files (e.g., public_release_2026_09_22)
        variant: Text variant ('text_only' or 'with_citations') - determines which answer file to load

    Returns:
        - ratings: ratings.csv
        - assignments: qa_assignments.csv
        - questions: qa_questions.csv
        - users: users.csv
        - answers_by_key: dict mapping (question_id, provider_key) -> answer_markdown
        - prior_users: blinded_prior_users.csv if exists, else None
        - user_registrations: recruitment_combined.csv if exists, else None
    """
    data_path = Path(data_dir)
    answers_path = Path(answers_dir)

    # Load ratings
    ratings = pd.read_csv(data_path / "ratings.csv")

    # Load assignments
    assignments = pd.read_csv(data_path / "qa_assignments.csv")

    # Load questions
    questions = pd.read_csv(data_path / "qa_questions.csv")

    # Load users
    users = pd.read_csv(data_path / "users.csv")

    # Load prior users if exists
    prior_users_path = data_path / "blinded_prior_users.csv"
    prior_users = pd.read_csv(prior_users_path) if prior_users_path.exists() else None

    # Load user registrations if exists
    registrations_path = data_path / "recruitment_combined.csv"
    user_registrations = pd.read_csv(registrations_path) if registrations_path.exists() else None

    # Load pre-cleaned answers based on variant
    answer_file = f"answers_{variant}.jsonl"
    answers_by_key = {}
    with open(answers_path / answer_file) as f:
        for line in f:
            data = json.loads(line)
            question_id = data["question_id"]
            provider_key = data["provider_key"]
            answer_markdown = data["answer_markdown"]
            answers_by_key[(question_id, provider_key)] = answer_markdown

    return ratings, assignments, questions, users, answers_by_key, prior_users, user_registrations


# -----------------------------------------------------------------------------
# LLM API Setup
# -----------------------------------------------------------------------------

def setup_llm_api(cache_db_path: str, model_str: str) -> LLMApi:
    """Setup LLM API with caching.

    Args:
        cache_db_path: Path to cache database
        model_str: Model string for litellm

    Returns:
        Configured LLMApi instance
    """
    load_dotenv()

    cache = CachingCompletion(cache_db_path)

    # Drop unsupported params (litellm will ignore params not supported by provider)
    litellm.drop_params = True

    if model_str.startswith("azure"):
        llm_completion = make_versa_openai_completion()
    elif model_str.startswith("bedrock"):
        llm_completion = make_versa_claude_completion()
    else:
        llm_completion = litellm.completion
    api = LLMApi(
        wrap_completion_function(
            llm_completion,
            cache=cache,
            model=model_str,
        )
    )

    return api


# -----------------------------------------------------------------------------
# Main Judging Logic
# -----------------------------------------------------------------------------

def build_job_list(
    eligible_assignments: pd.DataFrame,
    questions: pd.DataFrame,
    answers_by_key: dict,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[dict]:
    """Build list of judging jobs to run.

    Each job contains all info needed to judge one assignment.

    Args:
        eligible_assignments: Pre-filtered assignments from build_eligible_assignments()
        questions: Questions dataframe
        answers_by_key: Dict mapping (question_id, provider_key) -> answer_markdown
        limit: Optional limit on number of jobs
        seed: Random seed for shuffling (if None, no shuffling)
    """
    # Merge to get question text
    merged = eligible_assignments.merge(
        questions[["id", "question_text", "external_key"]],
        left_on="question_id",
        right_on="id",
        suffixes=("", "_q"),
    )

    jobs = []
    for _, row in merged.iterrows():
        assignment_id = row["id"]

        # Get answer texts (already cleaned markdown)
        slot_a_provider = row["slot_a_provider"]
        slot_b_provider = row["slot_b_provider"]
        question_id = row["question_id"]

        text_a = answers_by_key.get((question_id, slot_a_provider))
        text_b = answers_by_key.get((question_id, slot_b_provider))

        if not text_a or not text_b:
            continue

        jobs.append({
            "assignment_id": assignment_id,
            "question_id": question_id,
            "question_text": row["question_text"],
            "question_external_key": row["external_key"],
            "subversion": row["subversion"],
            "slot_a_provider": slot_a_provider,
            "slot_b_provider": slot_b_provider,
            "text_a": text_a,
            "text_b": text_b,
        })

    # Shuffle if seed provided, then apply limit
    if seed is not None:
        random.seed(seed)
        random.shuffle(jobs)

    if limit:
        jobs = jobs[:limit]

    return jobs


async def run_judging(
    api: LLMApi,
    jobs: list[dict],
    grader: str,
    variant: str,
    output_path: str,
    batch_size: int = 10,
    temperature: float = 1,
) -> tuple[int, int, pd.DataFrame]:
    """Run judging for all jobs and write results.

    Returns (completed, failed, judge_ratings) where judge_ratings is a long-form
    DataFrame with columns ['qa_assignment_id', 'axis', 'choice'] — one row per
    (assignment, axis) successfully scored by the judge.
    """
    # Build prompts
    prompts = []
    for job in jobs:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(
                job["question_text"],
                job["text_a"],
                job["text_b"],
            )},
        ]
        prompts.append(messages)

    logging.info(f"Running {len(prompts)} judging calls (batch_size={batch_size})")

    # Get per-model reasoning parameters
    reasoning_params = GRADER_REASONING_PARAMS.get(grader, {})
    if reasoning_params:
        logging.info(f"Using reasoning params: {reasoning_params}")

    # Run batch with model-specific reasoning parameters
    results = await api.run_batch(
        prompts,
        max_parallel_jobs=batch_size,
        temperature=temperature,
        response_format=JudgeRating,
        **reasoning_params,
    )

    # Process results and write to output
    completed = 0
    failed = 0
    judge_rows: list[dict] = []

    with open(output_path, "a") as f:
        for job, result in zip(jobs, results):
            rating = {
                "assignmentId": job["assignment_id"],
                "grader": grader,
                "variant": variant,
                "questionExternalKey": job["question_external_key"],
                "subversion": job["subversion"],
                "slotAProvider": job["slot_a_provider"],
                "slotBProvider": job["slot_b_provider"],
                "ts": datetime.utcnow().isoformat() + "Z",
            }

            if isinstance(result, JudgeRating):
                choices = {axis: getattr(result, axis) for axis in RATING_AXES}
                rating["choices"] = choices
                for axis, choice in choices.items():
                    judge_rows.append({
                        "qa_assignment_id": job["assignment_id"],
                        "axis": axis,
                        "choice": choice,
                    })
                completed += 1
            elif isinstance(result, Exception):
                rating["choices"] = None
                rating["error"] = str(result)
                failed += 1
            else:
                # Unexpected result type
                rating["choices"] = None
                rating["error"] = f"Unexpected result type: {type(result)}"
                failed += 1

            f.write(json.dumps(rating) + "\n")

            if (completed + failed) % 25 == 0:
                logging.info(f"  ... {completed + failed}/{len(jobs)} ({failed} failed)")

    judge_ratings = pd.DataFrame(judge_rows, columns=["qa_assignment_id", "axis", "choice"])
    return completed, failed, judge_ratings


def log_agreement_summary(
    judge_ratings: pd.DataFrame,
    ratings: pd.DataFrame,
) -> None:
    """Log how often judge ratings agree with human ratings, per axis + overall.

    `judge_ratings` must have columns ['qa_assignment_id', 'axis', 'choice'] —
    one row per (assignment, axis) scored by the judge. Each human rating in
    `ratings` is matched against the judge's choice for the same
    (assignment, axis) and counted as an independent comparison.
    """
    merged = ratings.merge(
        judge_ratings,
        on=["qa_assignment_id", "axis"],
        suffixes=("_human", "_judge"),
    )
    if merged.empty:
        logging.info("No overlapping (assignment, axis) pairs; skipping agreement summary.")
        return

    merged["agree"] = merged["choice_human"] == merged["choice_judge"]

    logging.info("=" * 60)
    logging.info("Agreement with human ratings (exact 5-point choice):")
    logging.info("-" * 60)
    per_axis = merged.groupby("axis")["agree"].agg(["sum", "count"])
    for axis in RATING_AXES:
        if axis not in per_axis.index:
            logging.info(f"  {axis:20s}: no overlapping human ratings")
            continue
        agree, total = int(per_axis.loc[axis, "sum"]), int(per_axis.loc[axis, "count"])
        logging.info(f"  {axis:20s}: {agree/total:6.1%} ({agree}/{total})")
    total_agree = int(merged["agree"].sum())
    total_n = len(merged)
    logging.info("-" * 60)
    logging.info(f"  {'OVERALL':20s}: {total_agree/total_n:6.1%} ({total_agree}/{total_n})")
    logging.info("=" * 60)


def judge_answers(
    data_dir: str,
    answers_dir: str,
    output_path: str,
    judge: str,
    variant: str = "text_only",
    cache_db: str = "cache.db",
    batch_size: int = 10,
    temperature: float = 1,
    limit: Optional[int] = None,
    seed: Optional[int] = None,
    user_stratification: Optional[str] = None,
    log_file: Optional[str] = None,
):
    """Main entry point for judging answers.

    Args:
        data_dir: Directory containing assignment/user data files
        answers_dir: Directory containing pre-cleaned answer JSONL files
        output_path: Path to output JSONL file
        judge: Judge model key (e.g., 'claude-opus-4-8')
        variant: Text variant ('text_only' or 'with_citations')
        cache_db: Path to LLM cache database
        batch_size: Number of parallel LLM calls
        temperature: LLM temperature
        limit: Optional limit on number of judgments
        seed: Random seed for shuffling data before limiting
        user_stratification: Optional OE Status filter ('On OE' or 'Off OE')
        log_file: Optional path to log file. Defaults to output_path with
            `.log` suffix.
    """
    # Setup logging (stream + file)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file),
            ],
            force=True,
        )
    logging.info(f"Logging to {log_file}")

    # Validate inputs
    if judge not in GRADER_MODELS:
        raise ValueError(f"Unknown judge: {judge}. Available: {list(GRADER_MODELS.keys())}")
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Available: {VARIANTS}")

    model_str = GRADER_MODELS[judge]
    logging.info(f"Judge: {judge} -> {model_str}")
    logging.info(f"Variant: {variant}")

    # Load data
    logging.info(f"Loading data from {data_dir}, answers from {answers_dir}")
    ratings, assignments, questions, users, answers_by_key, prior_users, user_registrations = load_data(
        data_dir, answers_dir, variant
    )
    logging.info(f"  {len(ratings)} total ratings")
    logging.info(f"  {len(assignments)} total assignments")
    logging.info(f"  {len(questions)} questions")
    logging.info(f"  {len(answers_by_key)} answers loaded")

    # Build eligible assignments (applying all filters from R analysis)
    logging.info("Applying inclusion criteria (matching R analysis.Rmd)...")
    eligible_assignments = build_eligible_assignments(
        ratings, assignments, users,
        prior_users=prior_users,
        user_registrations=user_registrations,
        user_stratification=user_stratification,
        variant=variant,
    )
    n_unique_questions = eligible_assignments["question_id"].nunique()
    logging.info(f"  {len(eligible_assignments)} eligible assignments (pairwise comparisons)")
    logging.info(f"  {n_unique_questions} unique questions")

    # Build job list
    jobs = build_job_list(
        eligible_assignments, questions, answers_by_key,
        limit, seed
    )
    logging.info(f"  {len(jobs)} jobs to run")
    
    if not jobs:
        logging.info("No jobs to run. Done.")
        return

    # Setup LLM API
    api = setup_llm_api(cache_db, model_str)

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Run judging
    completed, failed, judge_ratings = asyncio.run(
        run_judging(api, jobs, judge, variant, output_path, batch_size, temperature)
    )

    logging.info(f"\nDone. {completed} completed, {failed} failed.")
    logging.info(f"Output: {output_path}")

    # Agreement summary vs. human ratings
    return judge_ratings, ratings


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge evaluation on clinical Q&A answers"
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing assignment/user data files (qa_assignments.csv, etc.)",
    )
    parser.add_argument(
        "--answers-dir",
        required=True,
        help="Directory containing pre-cleaned answer JSONL files (answers_text_only.jsonl, etc.)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--judge",
        required=True,
        choices=list(GRADER_MODELS.keys()),
        help="Judge model to use",
    )
    parser.add_argument(
        "--variant",
        default="text_only",
        choices=VARIANTS,
        help="Text variant (default: text_only)",
    )
    parser.add_argument(
        "--cache-db",
        default=None,
        required=True,
        help="Path to LLM cache database (default: cache.db)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of parallel LLM calls (default: 10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="LLM temperature (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of judgments (for testing)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling data before applying limit",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file (default: output path with .log suffix)",
    )
    parser.add_argument(
        "--list-judges",
        action="store_true",
        help="List available judge models and exit",
    )

    args = parser.parse_args()
    load_dotenv()

    if args.list_judges:
        print("Available judges:")
        for key, model in GRADER_MODELS.items():
            label = GRADER_LABELS.get(key, key)
            print(f"  {key}: {label} -> {model}")
        return

    judge_ratings, ratings = judge_answers(
        data_dir=args.data_dir,
        answers_dir=args.answers_dir,
        output_path=args.output,
        judge=args.judge,
        variant=args.variant,
        cache_db=args.cache_db,
        batch_size=args.batch_size,
        temperature=args.temperature,
        limit=args.limit,
        seed=args.seed,
        log_file=args.log_file,
    )

    log_agreement_summary(judge_ratings, ratings)


if __name__ == "__main__":
    main()

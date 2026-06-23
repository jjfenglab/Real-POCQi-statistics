"""Filter questions to only those with at least one rating.

Usage:
    python src/filter_rated_questions.py --questions-csv data/qa_questions.csv --assignments-csv data/qa_assignments.csv --ratings-csv data/blinded_ratings.csv --output-csv output/rated_questions.csv
"""

import argparse
from pathlib import Path
import pandas as pd


def filter_rated_questions(
    questions_csv: str,
    assignments_csv: str,
    ratings_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """Filter questions to only those with at least one rating.

    Join path: ratings.qa_assignment_id -> assignments.id -> assignments.question_id -> questions.id

    Args:
        questions_csv: Path to questions CSV
        assignments_csv: Path to assignments CSV
        ratings_csv: Path to ratings CSV
        output_csv: Path to output CSV

    Returns:
        DataFrame with filtered questions
    """
    questions = pd.read_csv(questions_csv)
    assignments = pd.read_csv(assignments_csv)
    ratings = pd.read_csv(ratings_csv)

    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(assignments)} assignments")
    print(f"Loaded {len(ratings)} ratings")

    # Get unique assignment IDs that have ratings
    rated_assignment_ids = ratings['qa_assignment_id'].dropna().unique()
    print(f"Found {len(rated_assignment_ids)} assignments with ratings")

    # Get question IDs from those assignments
    rated_question_ids = assignments[
        assignments['id'].isin(rated_assignment_ids)
    ]['question_id'].unique()
    print(f"Found {len(rated_question_ids)} questions with ratings")

    # Filter questions
    filtered_questions = questions[questions['id'].isin(rated_question_ids)]
    print(f"Filtered to {len(filtered_questions)} questions")

    # Save output
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    filtered_questions.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

    return filtered_questions


def main():
    parser = argparse.ArgumentParser(
        description="Filter questions to only those with at least one rating"
    )
    parser.add_argument("--questions-csv", required=True, help="Path to questions CSV")
    parser.add_argument("--assignments-csv", required=True, help="Path to assignments CSV")
    parser.add_argument("--ratings-csv", required=True, help="Path to ratings CSV")
    parser.add_argument("--output-csv", required=True, help="Path to output CSV")

    args = parser.parse_args()

    filter_rated_questions(
        questions_csv=args.questions_csv,
        assignments_csv=args.assignments_csv,
        ratings_csv=args.ratings_csv,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()

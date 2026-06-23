"""Extract structured question types from clinical questions using LLM.

This script uses the lab_llm library to extract structured question categories
with caching and batch processing.

Usage:
    python src/extract_themes.py --input-csv data/qa_questions.csv --output-csv output/extracted_types.csv --prompt-template exp_themes/prompts/v1_extraction_template.txt
    python src/extract_themes.py --model us.anthropic.claude-opus-4-5-20251101-v1:0 --limit 10
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

# Add llm-api to path
sys.path.insert(0, str(Path(__file__).parent.parent / "llm-api"))

from common import (
    list_available_models,
    load_prompt_template,
    create_prompt_from_template,
    parse_model_string,
    setup_llm_api,
)
import models  # Import module to access all model classes dynamically


# Default column name for the text to process
DEFAULT_NOTE_TEXT_COLUMN = "question_text"


def get_response_model(model_name: str):
    """Get response model class by name from models module."""
    if not hasattr(models, model_name):
        available = [name for name in dir(models) if not name.startswith('_') and isinstance(getattr(models, name), type)]
        raise ValueError(f"Unknown response model: {model_name}. Available: {available}")
    return getattr(models, model_name)


def extract_themes(
    input_csv_path: str,
    output_csv_path: str,
    prompt_template_path: str,
    cache_db_path: str,
    log_file_path: str = "output/extraction.txt",
    model_str: str = "us.anthropic.claude-opus-4-5-20251101-v1:0",
    batch_size: int = 10,
    max_retries: int = 1,
    temperature: float = 1.0,
    limit: int = None,
    random_seed: int = None,
    max_char: int = None,
    id_column: str = None,
    response_model_name: str = "QuestionTypeExtraction",
    note_column: str = None,
) -> pd.DataFrame:
    """Extract structured data from FYI flag narratives using LLM.

    Args:
        input_csv_path: Path to CSV file with FYI flag narratives
        output_csv_path: Path to output CSV file for extracted data
        prompt_template_path: Path to prompt template file
        cache_db_path: Path to cache database
        log_file_path: Path to log file (txt format)
        model_str: Model string (e.g., 'us.anthropic.claude-opus-4-5-20251101-v1:0')
        batch_size: Batch size for LLM processing
        max_retries: Maximum number of retries for failed extractions
        temperature: Temperature for LLM sampling
        limit: Optional limit on number of notes to process (for testing)
        random_seed: Random seed for shuffling data before applying limit
        max_char: Optional maximum character length for note text
        id_column: Optional column name to use as ID (if not provided, uses row index)
        response_model_name: Name of pydantic model class in models.py (default: QuestionTypeExtraction)
        note_column: Column name containing narrative text (default: "Note Text")

    Returns:
        DataFrame with extracted data
    """
    # Validate inputs
    assert Path(input_csv_path).exists(), f"Input CSV file not found: {input_csv_path}"

    # Set note column name
    note_text_column = note_column if note_column else DEFAULT_NOTE_TEXT_COLUMN

    # Setup logging to both file and console
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Load data
    logger.info(f"Loading data from CSV: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # Validate column exists
    assert note_text_column in df.columns, f"Missing '{note_text_column}' column in CSV"

    # Create ID column if not specified
    if id_column and id_column in df.columns:
        df['_row_id'] = df[id_column]
    else:
        df['_row_id'] = df.index

    # Filter by max_char if specified
    if max_char is not None:
        pre_filter_count = len(df)
        df = df[df[note_text_column].str.len() <= max_char]
        logger.info(f"Filtered by max_char={max_char}: {pre_filter_count} -> {len(df)} notes")

    # Apply limit with optional shuffle
    if limit:
        logger.info(f"Limiting to {limit} notes (no shuffle)")
        df = df.head(limit)

    assert len(df) > 0, "No notes to process after filtering"

    # Get response model class
    response_model = get_response_model(response_model_name)
    logger.info(f"Using response model: {response_model_name}")

    # Setup LLM API
    logger.info("Setting up LLM API")
    api = setup_llm_api(cache_db_path, log_file_path, model_str)

    # Load prompt template
    logger.info(f"Loading prompt template from {prompt_template_path}")
    prompt_template = load_prompt_template(prompt_template_path)

    # Create prompts
    logger.info("Creating prompts")
    prompts = [create_prompt_from_template(prompt_template, note) for note in df[note_text_column]]
    assert len(prompts) == len(df), "Mismatch between prompts and notes"

    # Run batch extraction
    logger.info(f"Extracting data (batch_size={batch_size}, model={model_str})")
    results = asyncio.run(
        api.run_batch(
            prompts,
            max_parallel_jobs=batch_size,
            temperature=temperature,
            response_format=response_model,
        )
    )

    # Validate results
    assert len(results) == len(df), f"Expected {len(df)} results, got {len(results)}"

    # Convert results to DataFrame
    logger.info("Processing results")
    extracted_data = []
    none_count = 0

    # Get field names from the response model
    model_fields = list(response_model.model_fields.keys())

    for row_id, result in zip(df['_row_id'], results):
        if result is None or isinstance(result, Exception):
            # LLM call failed
            none_count += 1
            row_data = {'row_id': row_id, 'extraction_failed': True}
            for field in model_fields:
                row_data[field] = None
            extracted_data.append(row_data)
        elif isinstance(result, response_model):
            row_data = {'row_id': row_id, 'extraction_failed': False}
            for field in model_fields:
                value = getattr(result, field)
                # JSON-encode lists, keep other types as-is
                if isinstance(value, list):
                    row_data[field] = json.dumps(value)
                else:
                    row_data[field] = value
            extracted_data.append(row_data)
        else:
            logger.warning(f"Unexpected result type for row {row_id}: {type(result)}")
            none_count += 1
            row_data = {'row_id': row_id, 'extraction_failed': True}
            for field in model_fields:
                row_data[field] = None
            extracted_data.append(row_data)

    results_df = pd.DataFrame(extracted_data)

    # Merge with original data to preserve all columns
    df_out = df.merge(results_df, left_on='_row_id', right_on='row_id', how='left')
    df_out = df_out.drop(columns=['_row_id', 'row_id'])

    # Report statistics
    logger.info("Extraction Statistics")
    logger.info(f"Total notes processed: {len(df)}")
    logger.info(f"Failed extractions: {none_count}")
    logger.info(f"Success rate: {(len(df) - none_count) / len(df):.1%}")

    # Save results
    logger.info(f"Saving results to CSV: {output_csv_path}")
    Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv_path, index=False)

    return df_out


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract structured themes from FYI flag narratives using LLM"
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to CSV file with FYI flag narratives"
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to output CSV file for extracted themes"
    )
    parser.add_argument(
        "--prompt-template",
        required=True,
        help="Path to prompt template file (must contain {note_text} placeholder)"
    )
    parser.add_argument(
        "--cache-db",
        default="cache.db",
        help="Path to LLM cache database"
    )
    parser.add_argument(
        "--log-file",
        default="output/extraction.txt",
        help="Path to log file (txt format)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for LLM processing"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum retries for failed extractions"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for LLM sampling"
    )
    parser.add_argument(
        "--model",
        default="us.anthropic.claude-opus-4-5-20251101-v1:0",
        help="LLM model to use"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of notes to process (for testing)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for shuffling data before applying --limit"
    )
    parser.add_argument(
        "--max-char",
        type=int,
        default=None,
        help="Maximum character length for note text"
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Column name to use as ID (if not provided, uses row index)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--response-model",
        default="QuestionTypeExtraction",
        help="Pydantic response model class name from models.py (default: QuestionTypeExtraction)"
    )
    parser.add_argument(
        "--note-column",
        default=None,
        help="Column name containing narrative text (default: 'Note Text')"
    )

    args = parser.parse_args()
    load_dotenv()

    # Handle list-models command
    if args.list_models:
        print("Available models:")
        for model in list_available_models():
            print(f"  - {model}")
        return

    # Validate model string early
    try:
        model = parse_model_string(args.model)
        print(f"Model validation passed: {model}")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create directories if needed
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.cache_db).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Run extraction
    df = extract_themes(
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv,
        prompt_template_path=args.prompt_template,
        cache_db_path=args.cache_db,
        log_file_path=args.log_file,
        model_str=args.model,
        batch_size=args.batch_size,
        # max_retries=args.max_retries,
        temperature=args.temperature,
        limit=args.limit,
        random_seed=args.random_seed,
        max_char=args.max_char,
        id_column=args.id_column,
        response_model_name=args.response_model,
        note_column=args.note_column,
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Successfully extracted themes from {len(df)} notes")
    logger.info(f"Model used: {args.model}")
    logger.info(f"Output: {args.output_csv}")


if __name__ == "__main__":
    main()

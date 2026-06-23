"""Common utilities for analysis scripts.

This module contains shared functions used across multiple analysis scripts.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


def setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to file and console.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_safe_data(filepath: str, logger: logging.Logger) -> pd.DataFrame:
    """Load cleaned SAFE reports data, dropping records with missing MRN or date.

    Args:
        filepath: Path to cleaned SAFE CSV
        logger: Logger instance

    Returns:
        DataFrame with parsed dates and normalized MRN
    """
    logger.info(f"Loading SAFE data from {filepath}")
    df = pd.read_csv(filepath, parse_dates=['safe_event_date'])
    n_total = len(df)

    # Drop records with missing MRN
    n_missing_mrn = df['mrn'].isna().sum()
    df = df.dropna(subset=['mrn'])
    df['mrn'] = df['mrn'].astype(str)

    # Drop records with missing date
    n_missing_date = df['safe_event_date'].isna().sum()
    df = df.dropna(subset=['safe_event_date'])

    n_dropped = n_total - len(df)
    logger.info(f"  Loaded {len(df)} SAFE events (dropped {n_dropped}: {n_missing_mrn} missing MRN, {n_missing_date} missing date)")
    return df


def load_flags_data(filepath: str, logger: logging.Logger) -> pd.DataFrame:
    """Load cleaned FYI flags data, dropping records with missing MRN or date.

    Args:
        filepath: Path to cleaned flags CSV
        logger: Logger instance

    Returns:
        DataFrame with parsed dates and normalized MRN
    """
    logger.info(f"Loading flags data from {filepath}")
    df = pd.read_csv(filepath, parse_dates=['flag_datetime'])
    n_total = len(df)

    # Drop records with missing MRN
    n_missing_mrn = df['mrn'].isna().sum()
    df = df.dropna(subset=['mrn'])
    df['mrn'] = df['mrn'].astype(str)

    # Drop records with missing date
    n_missing_date = df['flag_datetime'].isna().sum()
    df = df.dropna(subset=['flag_datetime'])

    n_dropped = n_total - len(df)
    logger.info(f"  Loaded {len(df)} FYI flags (dropped {n_dropped}: {n_missing_mrn} missing MRN, {n_missing_date} missing date)")
    return df


def load_patient_data(filepath: str, logger: logging.Logger) -> pd.DataFrame:
    """Load patient summary data.

    Args:
        filepath: Path to patient summary CSV
        logger: Logger instance

    Returns:
        DataFrame with normalized MRN
    """
    logger.info(f"Loading patient data from {filepath}")
    df = pd.read_csv(filepath)
    df['mrn'] = df['mrn'].astype(str)
    logger.info(f"  Loaded {len(df)} patients")
    return df

# Add llm-api to path
sys.path.insert(0, str(Path(__file__).parent.parent / "llm-api"))

from lab_llm import LLMApi, wrap_completion_function, CachingCompletion, ErrorTracker
from lab_llm.versa import make_versa_claude_completion
from lab_llm.constants import VersaClaude, Claude
import litellm


def list_available_models() -> list:
    """List all available model strings.

    Returns:
        List of available model strings
    """
    return [
        VersaClaude.CLAUDE_SONNET_4,
        VersaClaude.CLAUDE_OPUS_4_5,
        VersaClaude.CLAUDE_HAIKU_4_5,
        Claude.SONNET_4,
        Claude.OPUS_4_5,
        Claude.HAIKU_4_5,
    ]


def load_prompt_template(template_path: str) -> str:
    """Load prompt template from file.

    Args:
        template_path: Path to the prompt template file

    Returns:
        Template string with placeholder for note text
    """
    assert Path(template_path).exists(), f"Prompt template file not found: {template_path}"

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read().strip()

    assert "{note_text}" in template, "Prompt template must contain {note_text} placeholder"

    return template


def create_prompt_from_template(template: str, note_text: str) -> str:
    """Create a prompt from template by substituting note text.

    Args:
        template: Prompt template with {note_text} placeholder
        note_text: The clinical note text to analyze

    Returns:
        Formatted prompt string for LLM extraction
    """
    return template.format(note_text=note_text)


def parse_model_string(model_str: str) -> str:
    """Parse and validate model string.

    Args:
        model_str: Model string (e.g., 'anthropic/claude-sonnet-4', 'bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0')

    Returns:
        Validated model string

    Raises:
        ValueError: If model string is not recognized
    """
    valid_models = list_available_models()
    if model_str in valid_models:
        return model_str
    # Allow pass-through for litellm-compatible model strings
    return model_str


def setup_llm_api(
    cache_db_path: str,
    log_file_path: str,
    model_str: str = "anthropic/claude-sonnet-4",
    seed: int = 42
) -> LLMApi:
    """Setup LLM API with caching.

    Args:
        cache_db_path: Path to cache database
        log_file_path: Path to log file (txt format)
        model_str: Model string (e.g., 'anthropic/claude-sonnet-4', 'bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0')
        seed: Random seed for reproducibility

    Returns:
        Configured LLMApi instance
    """
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Setting up cache: {cache_db_path}")
    cache = CachingCompletion(cache_db_path)

    model = parse_model_string(model_str)
    logger.info(f"Using model: {model}")

    # Drop unsupported params (e.g., seed not supported by bedrock)
    litellm.drop_params = True

    api = LLMApi(
        wrap_completion_function(
            make_versa_claude_completion(), #litellm.completion,
            cache=cache,
            model=model,
        )
    )

    return api

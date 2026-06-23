"""Pydantic models for clinical question categorization."""

from pydantic import BaseModel, Field
from typing import List


class QuestionTypeExtraction(BaseModel):
    """Structured model for extracting question type from clinical questions.

    Each field contains a list of keyphrases extracted from the question.
    Empty lists indicate no relevant information was found for that category.
    """

    question_types: List[str] = Field(
        default_factory=list,
        description="Types of clinical information being requested. Examples: treatment, diagnosis, mechanism, drug interaction, prognosis, epidemiology, differential diagnosis, clinical presentation, dosing, monitoring, contraindication, side effect, workup/evaluation, guideline/recommendation, imaging interpretation, lab interpretation, procedure technique, risk factors, prevention, screening, patient education"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "question_types": ["treatment", "first-line therapy"],
                },
                {
                    "question_types": ["drug interaction", "mechanism"],
                },
                {
                    "question_types": ["differential diagnosis", "clinical presentation"],
                }
            ]
        }


class QuestionTagging(BaseModel):
    """Model for tagging questions with predefined categories.

    Used after clustering to tag all questions with cleaned category labels.
    """

    primary_question_type: str = Field(
        description="The primary type of clinical information being requested (select one from the provided list)"
    )

    secondary_question_types: List[str] = Field(
        default_factory=list,
        description="Additional question types if the question spans multiple categories"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "primary_question_type": "treatment",
                    "secondary_question_types": ["drug selection"],
                }
            ]
        }

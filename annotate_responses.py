from lab_llm import LLMApi, wrap_completion_function, CachingCompletion, VersaClaude, UsageTracker
from lab_llm.versa import make_versa_claude_completion
import pandas as pd
import json
from pathlib import Path
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import tqdm

load_dotenv()

type AgreementRating = Literal["yes", "no", "n/a"]

class NoteAnnotationResponse(BaseModel):
    address_as_patient: AgreementRating
    verbatim_guidelines: AgreementRating
    specific_study_details: AgreementRating
    conflicting_evidence: AgreementRating
    dont_miss_items: AgreementRating
    gives_definitive_answer: AgreementRating
    begin_clinical_background: AgreementRating
    disclaimer: AgreementRating

prompt = """
You are an expert US physician serving as an evaluator of a clinical decision support system that is used by physicians in a real clinical setting.

You will be given a question followed by the system's response. Annotate the response by answering the following questions with "yes", "no", or "n/a":
* address_as_patient: Does the response imply or state that the question-asker would be receiving medical care related to the topic, or that the question-asker is a patient?
* verbatim_guidelines: Does the response directly reference or quote clinical guidelines?
* specific_study_details: Does the response provide evidence and details from a specific prior study?
* conflicting_evidence: Does the response suggest that multiple possible responses are plausible, e.g. multiple diagnoses or treatments?
* dont_miss_items: Does the response call out specific diagnoses as being important to check for? Answer 'n/a' if the question does not involve a diagnosis.
* gives_definitive_answer: Does the response give a single definitive answer to the user's question within the first paragraph?
* begin_clinical_background: Does the response begin with clinical background information such as defining conditions in the user's question?
* disclaimer: Does the response include a note disclaiming the system's authority and requesting the user to confirm with other sources such as their own clinical reasoning?
Provide your answer as a JSON structured object where the possible answers are "yes", "no", or "n/a" (only if the question clearly does not apply).

>>> QUESTION

{question}

>>> ANSWER

{answer}

>>> YOUR ANNOTATIONS

"""
if __name__ == '__main__':
    usage = UsageTracker()
    api = LLMApi(wrap_completion_function(make_versa_claude_completion(), 
                                          cache=CachingCompletion("cache.db"), 
                                          usage_tracker=usage,
                                          model=VersaClaude.CLAUDE_OPUS_4_6))

    base_path = Path("data_cleaned")
    responses = pd.read_json(base_path / "answers.jsonl", lines=True)
    questions = pd.read_csv(base_path / "questions.csv")

    questions_answers = pd.merge(
        questions,
        responses,
        how='left',
        on='question_id'
    )

    annotations = []
    bar = tqdm.tqdm(questions_answers.iterrows())
    for i, row in bar:
        if i % 10 == 0:
            pd.DataFrame(annotations).to_csv("response_annotations.csv", index=False)
        response = api.run(prompt.format(
            question=row['question_text'],
            answer=row['answer_markdown']
        ), response_format=NoteAnnotationResponse)
        bar.set_description(f"{usage.last_usage()['total_tokens']} tokens")
        annotations.append({
            'question_id': row['question_id'],
            'answer_id': i,
            'provider_key': row['provider_key'],
            'specialty': row['specialty'],
            **response.model_dump()
        })

# from lab_llm import LLMApi, wrap_completion_function, CachingCompletion, VersaClaude, UsageTracker
# from lab_llm.versa import make_versa_claude_completion
# import pandas as pd
# import json
# from pathlib import Path
# from pydantic import BaseModel
# from typing import Literal
# from dotenv import load_dotenv
# import tqdm
# import multiprocessing as mp
# from functools import partial
# import os
# import asyncio
# import time

# # Configuration
# NUM_WORKERS = 4  # Number of worker processes
# BATCH_SIZE = 32

# load_dotenv()

# type AgreementRating = Literal["yes", "no", "n/a"]

# class NoteAnnotationResponse(BaseModel):
#     address_as_patient: AgreementRating
#     verbatim_guidelines: AgreementRating
#     specific_study_details: AgreementRating
#     conflicting_evidence: AgreementRating
#     dont_miss_items: AgreementRating
#     gives_definitive_answer: AgreementRating
#     begin_clinical_background: AgreementRating
#     disclaimer: AgreementRating

# prompt = """
# You are an expert US physician serving as an evaluator of a clinical decision support system that is used by physicians in a real clinical setting.

# You will be given a question followed by the system's response. Annotate the response by answering the following questions with "yes", "no", or "n/a":
# * address_as_patient: Does the response imply or state that the question-asker would be receiving medical care related to the topic, or that the question-asker is a patient?
# * verbatim_guidelines: Does the response directly reference or quote clinical guidelines?
# * specific_study_details: Does the response provide evidence and details from a specific prior study?
# * conflicting_evidence: Does the response suggest that multiple possible responses are plausible, e.g. multiple diagnoses or treatments?
# * dont_miss_items: Does the response call out specific diagnoses as being important to check for? Answer 'n/a' if the question does not involve a diagnosis.
# * gives_definitive_answer: Does the response give a single definitive answer to the user's question within the first paragraph?
# * begin_clinical_background: Does the response begin with clinical background information such as defining conditions in the user's question?
# * disclaimer: Does the response include a note disclaiming the system's authority and requesting the user to confirm with other sources such as their own clinical reasoning?
# Provide your answer as a JSON structured object where the possible answers are "yes", "no", or "n/a" (only if the question clearly does not apply).

# >>> QUESTION

# {question}

# >>> ANSWER

# {answer}

# >>> YOUR ANNOTATIONS

# """

# async def main():
#     usage = UsageTracker()
#     api = LLMApi(wrap_completion_function(make_versa_claude_completion(), 
#                                           cache=CachingCompletion("cache.db"), 
#                                           usage_tracker=usage,
#                                           model=VersaClaude.CLAUDE_OPUS_4_6))

#     base_path = Path("data_cleaned")
#     responses = pd.read_json(base_path / "answers.jsonl", lines=True)
#     questions = pd.read_csv(base_path / "questions.csv")

#     questions_answers = pd.merge(
#         questions,
#         responses,
#         how='left',
#         on='question_id'
#     )

#     annotations = []
#     for batch_start in range(0, len(questions_answers), BATCH_SIZE):
#         batch = questions_answers.iloc[batch_start:batch_start + BATCH_SIZE]
#         while True:
#             try:
#                 usage.reset()
#                 responses = await api.run_batch([prompt.format(
#                     question=row['question_text'],
#                     answer=row['answer_markdown']
#                 ) for _, row in batch.iterrows()], response_format=NoteAnnotationResponse)
#                 print(f"{usage.total_usage()['total_tokens']} tokens")
#                 break
#             except:
#                 print("Received exception, waiting 10s...")
#                 time.sleep(10)
#         annotations.extend([{
#             'question_id': row['question_id'],
#             'answer_id': batch_start + i,
#             'provider_key': row['provider_key'],
#             'specialty': row['specialty'],
#             **response.model_dump()
#         } for i, ((_, row), response) in enumerate(zip(batch.iterrows(), responses))])
#         pd.DataFrame(annotations).to_csv("response_annotations.csv", index=False)

# if __name__ == '__main__':
#     asyncio.run(main())
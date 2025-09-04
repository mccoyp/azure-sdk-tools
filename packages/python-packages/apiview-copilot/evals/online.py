import argparse
import json
import os
import pathlib
import sys
from typing import Any

# Ensure the parent directory is in the path to import from `cli`
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# set before azure.ai.evaluation import to make PF output less noisy
os.environ["PF_LOGGING_LEVEL"] = "CRITICAL"

import dotenv
from azure.ai.evaluation import evaluate, GroundednessEvaluator, TaskAdherenceEvaluator

from cli import get_apiview_comments
from src._search_manager import SearchManager


dotenv.load_dotenv()

DEFAULT_NUM_RUNS: int = 1
# for best results, this should always be a different model from the one we are evaluating
MODEL_JUDGE = "gpt-4.1-nano"

model_config: dict[str, str] = {
    "azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
    "azure_deployment": MODEL_JUDGE,
    "api_version": "2025-03-01-preview",
}


def _clean_comments(comments: dict[str, Any]) -> dict[str, Any]:
    cleaned = []
    for _, comment_list in comments.items():
        for c in comment_list:
            text = c.get("commentText")
            if text.startswith("### API Summary"):
                continue
            cleaned.append(c.get("commentText"))
    return {"comments": cleaned}


class OnlineEvaluator:
    """Evaluator for the groundedness live-site review comments, based on SDK guidelines."""

    def __init__(self):
        self._groundedness_eval = GroundednessEvaluator(model_config=model_config)
        self._task_adherence_eval = TaskAdherenceEvaluator(model_config=model_config)

    def _groundedness(self, actual: dict[str, Any], context: str) -> None:
        actual = [c for c in actual["comments"] if c["rule_ids"]]
        if not actual:
            return {"groundedness": 0.0, "groundedness_reason": "No comments found."}
        groundedness = self._groundedness_eval(response=json.dumps(actual), context=context)
        return groundedness

    def _task_adherence(self, actual: dict[str, Any], context: str) -> dict[str, Any]:
        actual = [c for c in actual["comments"] if c["rule_ids"]]
        if not actual:
            return {"task_adherence": 0.0, "task_adherence_reason": "No comments found."}
        task_adherence = self._task_adherence_eval(query=context, response=json.dumps(actual))
        return task_adherence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evals for APIview copilot.")
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="python",
        help="The language to run evals for. Defaults to python.",
    )
    parser.add_argument(
        "--num-runs",
        "-n",
        type=int,
        default=DEFAULT_NUM_RUNS,
        help=f"The number of runs to perform, with the median of results kept. Defaults to {DEFAULT_NUM_RUNS}.",
    )
    parser.add_argument(
        "--review",
        "-r",
        type=str,
        default="d48830fcdd2c4713b6344d61ae626d5e",
        help="The ID of the APIview review to get comments from. Defaults to 'd48830fcdd2c4713b6344d61ae626d5e' (azure-monitor-healthmodels).",
    )
    parser.add_argument(
        "--guidelines",
        "-g",
        type=str,
        default="../prompts/api_review/guidelines_review_sample.jsonl",
        help="The path to the guidelines file to use for evaluation. Defaults to '../prompts/api_review/guidelines_review_sample.jsonl'.",
    )
    args = parser.parse_args()

    custom_eval = OnlineEvaluator()
    rule_ids = set()

    azure_ai_project = {
        "subscription_id": os.environ["AZURE_SUBSCRIPTION_ID"],
        "resource_group_name": os.environ["AZURE_FOUNDRY_RESOURCE_GROUP"],
        "project_name": os.environ["AZURE_FOUNDRY_PROJECT_NAME"],
    }
    kwargs = {}

    comments = get_apiview_comments(args.review, use_api=True)
    comments = _clean_comments(comments)
    breakpoint()
    # search_manager = SearchManager(language=args.language, include_general_guidelines=True)
    # guidelines = search_manager.search_guidelines()
    run_results = []
    for run in range(args.num_runs):
        print(f"Running evals {run + 1}/{args.num_runs} for {args.review}...")
        with open(pathlib.Path(args.guidelines).expanduser(), "r") as file:
            guidelines = file.read()
        groundedness_result = custom_eval._groundedness_eval(response=json.dumps(comments), context=guidelines)
        task_adherence_result = custom_eval._task_adherence_eval(query=guidelines, response=json.dumps(comments))
        print("Groundedness result:", groundedness_result)
        print("Task adherence result:", task_adherence_result)

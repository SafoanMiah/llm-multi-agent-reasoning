import json
import random
from pathlib import Path


def load_gsm8k(n: int = 100, seed: int = 42) -> list[dict]:
    """
    Load a random subset of GSM8K test questions from local JSONL file.

    Args: n: Number of questions to sample.
    Returns: List of dicts with 'question', 'expected_answer', and 'raw_answer'.
    """
    path = Path(__file__).parent / "gsm8k_test.jsonl"

    with open(path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    random.seed(seed)
    sample = random.sample(data, min(n, len(data)))

    questions = []
    for item in sample:
        try:
            # Extract just the numerical answer after the ####
            expected = float(item["answer"].split("#### ")[-1])
            questions.append(
                {
                    "question": item["question"],
                    "expected_answer": expected,
                    "raw_answer": item["answer"],
                }
            )
        except ValueError as e:
            print(f"Skipping question: {e}")

    print(f"Loaded {len(questions)} GSM8K questions (seed={seed})")
    return questions


if __name__ == "__main__":
    questions = load_gsm8k(n=5)
    for i, q in enumerate(questions):
        print(f"\n--- Question {i + 1} ---")
        print(f"Q: {q['question']}")
        print(f"A: {q['expected_answer']}")

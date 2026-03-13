"""
Experiment Runner

Runs a single topology across all questions and saves results to CSV.

Usage:
    python -m experiments.run_experiment --topology independent
    python -m experiments.run_experiment --topology full
    python -m experiments.run_experiment --topology mediator
    python -m experiments.run_experiment --topology chain
"""

import argparse
import csv
import time
from pathlib import Path
from datetime import datetime

from core.llm_client import LLMClient
from core.agent import Agent
from core.config import MODELS, NUM_QUESTIONS, DATASET_SEED, TOPOLOGIES
from dataset.gsm8k_loader import load_gsm8k
from topology.independent import run_independent
from topology.fully_connected import run_fully_connected
from topology.mediator import run_mediator
from topology.chain import run_chain

TOPOLOGY_RUNNERS = {
    "independent": run_independent,
    "full": run_fully_connected,
    "mediator": run_mediator,
    "chain": run_chain,
}

FIELDNAMES = [
    "topology",
    "question_idx",
    "question",
    "expected_answer",
    "group_answer",
    "correct",
    "round",
    "agent_id",
    "model",
    "answer",
    "confidence",
    "reasoning",
    "parse_failed",
    "prompt_tokens",
    "completion_tokens",
]


def create_agents() -> list[Agent]:
    """Create one agent per model defined in config."""
    return [
        Agent(agent_id=i, client=LLMClient(model=model))
        for i, model in enumerate(MODELS)
    ]


def flatten_results(topology, q_idx, question, expected, result) -> list[dict]:
    """Flatten a single question's results into CSV rows."""
    group_answer = result["group_answer"]
    if group_answer is None:
        correct_val = None
    else:
        correct_val = int(group_answer == expected)

    base = {
        "topology": topology,
        "question_idx": q_idx,
        "question": question,
        "expected_answer": expected,
        "group_answer": group_answer,
        "correct": correct_val,
    }

    rows = []

    if topology == "independent":
        round_blocks = [{"round": 1, "responses": result["agent_responses"]}]
    elif topology == "chain":
        round_blocks = [
            {"round": c["chain"], "responses": c["responses"]} for c in result["chains"]
        ]
    else:
        round_blocks = result["rounds"]

    for block in round_blocks:
        for agent_r in block["responses"]:
            rows.append(
                {
                    **base,
                    "round": block["round"],
                    "agent_id": agent_r["agent_id"],
                    "model": MODELS[agent_r["agent_id"]],
                    "answer": agent_r["answer"],
                    "confidence": agent_r["confidence"],
                    "reasoning": agent_r["reasoning"],
                    "parse_failed": agent_r.get("parse_failed", False),
                    "prompt_tokens": agent_r.get("prompt_tokens", 0),
                    "completion_tokens": agent_r.get("completion_tokens", 0),
                }
            )

    return rows


def run_experiment(topology: str):
    """Run a full experiment for one topology and save results."""
    print(f"\n{'=' * 50}")
    print(f"Running experiment: {topology}")
    print(f"{'=' * 50}")

    agents = create_agents()
    questions = load_gsm8k(n=NUM_QUESTIONS, seed=DATASET_SEED)
    runner = TOPOLOGY_RUNNERS[topology]

    # Output file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"{topology}_{timestamp}.csv"

    total_correct = 0
    start_time = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, q in enumerate(questions):
            q_start = time.time()

            result = runner(agents, q["question"])
            rows = flatten_results(
                topology, i, q["question"], q["expected_answer"], result
            )

            for row in rows:
                writer.writerow(row)

            is_correct = result["group_answer"] == q["expected_answer"]
            total_correct += int(is_correct)
            elapsed = time.time() - q_start

            print(
                f"  [{i + 1}/{len(questions)}] "
                f"{'✔️' if is_correct else '❌'} "
                f"expected={q['expected_answer']}, got={result['group_answer']} "
                f"({elapsed:.1f}s)"
            )

    total_time = time.time() - start_time
    accuracy = total_correct / len(questions) * 100

    print(f"\n{'=' * 50}")
    print(f"Finished: {topology}")
    print(f"Accuracy: {total_correct}/{len(questions)} ({accuracy:.1f}%)")
    print(f"Time: {total_time:.0f}s")
    print(f"Saved to: {output_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent topology experiment")
    parser.add_argument(
        "--topology",
        type=str,
        choices=TOPOLOGIES,
        default=None,
        help="Topology to run. If not set, runs all topologies.",
    )
    args = parser.parse_args()

    topologies = [args.topology] if args.topology else TOPOLOGIES
    for t in topologies:
        run_experiment(t)

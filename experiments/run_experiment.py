"""
Experiment Runner

Runs a single topology across all questions and saves results to CSV.

Usage:
    python -m experiments.run_experiment --topology independent
    python -m experiments.run_experiment --topology full
    python -m experiments.run_experiment --topology mediator
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


def create_agents() -> list[Agent]:
    """Create one agent per model defined in config."""
    agents = []
    for i, model in enumerate(MODELS):
        client = LLMClient(model=model)
        agents.append(Agent(agent_id=i, client=client))
    return agents


def run_topology(topology: str, agents: list[Agent], question: str) -> dict:
    """Dispatch to the correct topology runner."""
    if topology == "independent":
        return run_independent(agents, question)
    elif topology == "full":
        return run_fully_connected(agents, question)
    elif topology == "mediator":
        return run_mediator(agents, question)
    else:
        raise ValueError(f"Unknown topology: {topology}")


def flatten_results(
    topology: str, q_idx: int, question: str, expected: float, result: dict
) -> list[dict]:
    """Flatten a single question's results into CSV rows (one row per agent per round)."""
    rows = []

    if topology == "independent":
        # Single round only
        for agent_r in result["agent_responses"]:
            rows.append(
                {
                    "topology": topology,
                    "question_idx": q_idx,
                    "question": question,
                    "expected_answer": expected,
                    "group_answer": result["group_answer"],
                    "correct": float(result["group_answer"] == expected)
                    if result["group_answer"] is not None
                    else 0,
                    "round": 1,
                    "agent_id": agent_r["agent_id"],
                    "model": MODELS[agent_r["agent_id"]],
                    "answer": agent_r["answer"],
                    "confidence": agent_r["confidence"],
                    "reasoning": agent_r["reasoning"],
                }
            )
    else:
        # Multi-round topologies
        for round_data in result["rounds"]:
            for agent_r in round_data["responses"]:
                rows.append(
                    {
                        "topology": topology,
                        "question_idx": q_idx,
                        "question": question,
                        "expected_answer": expected,
                        "group_answer": result["group_answer"],
                        "correct": float(result["group_answer"] == expected)
                        if result["group_answer"] is not None
                        else 0,
                        "round": round_data["round"],
                        "agent_id": agent_r["agent_id"],
                        "model": MODELS[agent_r["agent_id"]],
                        "answer": agent_r["answer"],
                        "confidence": agent_r["confidence"],
                        "reasoning": agent_r["reasoning"],
                    }
                )

    return rows


def run_experiment(topology: str):
    """Run a full experiment for one topology and save results."""
    assert topology in TOPOLOGIES, f"Topology must be one of {TOPOLOGIES}"

    print(f"\n{'=' * 60}")
    print(f"Running experiment: {topology}")
    print(f"{'=' * 60}")

    agents = create_agents()
    questions = load_gsm8k(n=NUM_QUESTIONS, seed=DATASET_SEED)

    # Output file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"{topology}_{timestamp}.csv"

    fieldnames = [
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
    ]

    total_correct = 0
    start_time = time.time()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, q in enumerate(questions):
            q_start = time.time()

            result = run_topology(topology, agents, q["question"])
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
                f"{'✓' if is_correct else '✗'} "
                f"expected={q['expected_answer']}, got={result['group_answer']} "
                f"({elapsed:.1f}s)"
            )

    total_time = time.time() - start_time
    accuracy = total_correct / len(questions) * 100

    print(f"\n{'=' * 60}")
    print(f"Finished: {topology}")
    print(f"Accuracy: {total_correct}/{len(questions)} ({accuracy:.1f}%)")
    print(f"Time: {total_time:.0f}s")
    print(f"Saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent topology experiment")
    parser.add_argument(
        "--topology",
        type=str,
        required=True,
        choices=TOPOLOGIES,
        help="Topology to run: independent, full, or mediator",
    )
    args = parser.parse_args()

    run_experiment(args.topology)

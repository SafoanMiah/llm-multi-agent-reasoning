"""
Experiment Runner

Runs a single topology across all questions and saves results to CSV.

Usage:
    python -m experiments.run_experiment --topology independent
    python -m experiments.run_experiment --topology full
    python -m experiments.run_experiment --topology mediator
    python -m experiments.run_experiment --topology chain
    python -m experiments.run_experiment --topology self_refine
"""

import argparse
import csv
from pathlib import Path
from datetime import datetime

from core.llm_client import LLMClient
from core.agent import Agent
from core.config import (
    MODELS,
    NUM_ROUNDS,
    NUM_QUESTIONS,
    DATASET_SEED,
    TOPOLOGIES,
    TEMPERATURE,
    OLLAMA_BASE_URL,
)
from dataset.gsm8k_loader import load_gsm8k
from topology.independent import run_independent
from topology.fully_connected import run_fully_connected
from topology.mediator import run_mediator
from topology.chain import run_chain
from topology.self_refine import run_self_refine

TOPOLOGY_RUNNERS = {
    "independent": run_independent,
    "full": run_fully_connected,
    "mediator": run_mediator,
    "chain": run_chain,
    "self_refine": run_self_refine,
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


def create_agents(temperature: float = None, base_url: str = None) -> list[Agent]:
    """Create one agent per model defined in config."""
    return [
        Agent(
            agent_id=i,
            client=LLMClient(model=model, base_url=base_url),
            temperature=temperature,
        )
        for i, model in enumerate(MODELS)
    ]


def get_config_folder_name(
    num_rounds: int,
    num_questions: int,
    seed: int,
    start_idx: int = 0,
) -> str:
    """Generate folder name based on config settings (e.g., A4-R3-Q150-S0-START100)."""
    start_suffix = f"-START{start_idx}" if start_idx > 0 else ""
    return f"R{num_rounds}-Q{num_questions}-S{seed}{start_suffix}"


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


def run_experiment(
    topology: str,
    start_idx: int = 0,
    num_questions: int = None,
    num_rounds: int = None,
    seed: int = None,
    temperature: float = None,
    base_url: str = None,
):
    """Run a full experiment for one topology and save results."""
    # Use CLI overrides if provided, otherwise use config defaults
    n_questions = num_questions if num_questions is not None else NUM_QUESTIONS
    n_rounds = num_rounds if num_rounds is not None else NUM_ROUNDS
    dataset_seed = seed if seed is not None else DATASET_SEED
    n_temperature = temperature if temperature is not None else TEMPERATURE
    n_base_url = base_url if base_url is not None else OLLAMA_BASE_URL

    print(f"\n{'=' * 50}")
    print(f"Running experiment: {topology}")
    print(f"Questions: {n_questions} (starting at {start_idx})")
    if topology in ("full", "mediator", "self_refine"):
        print(f"Rounds: {n_rounds}")
    else:
        print("Rounds: N/A (topology-specific)")
    print(f"Seed: {dataset_seed}")
    print(f"Temperature: {n_temperature}")
    print(f"Base URL: {n_base_url}")
    print(f"{'=' * 50}")

    agents = create_agents(temperature=n_temperature, base_url=n_base_url)
    questions = load_gsm8k(n=n_questions, seed=dataset_seed)[start_idx:]
    runner = TOPOLOGY_RUNNERS[topology]

    # Output file - use config-based folder
    config_folder = get_config_folder_name(
        num_rounds=n_rounds,
        num_questions=n_questions,
        seed=dataset_seed,
        start_idx=start_idx,
    )
    results_dir = Path("results") / config_folder
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"{topology}_{timestamp}.csv"

    total_correct = 0
    total_tokens = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, q in enumerate(questions):
            actual_idx = i + start_idx
            # Pass num_rounds only to topologies that support it
            if topology in ("full", "mediator", "self_refine"):
                result = runner(agents, q["question"], num_rounds=n_rounds)
            else:
                result = runner(agents, q["question"])
            rows = flatten_results(
                topology, actual_idx, q["question"], q["expected_answer"], result
            )

            for row in rows:
                writer.writerow(row)

            is_correct = result["group_answer"] == q["expected_answer"]
            total_correct += int(is_correct)

            # Count tokens for this question
            q_tokens = sum(
                r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in rows
            )
            total_tokens += q_tokens

            print(
                f"  [{actual_idx + 1}/{len(questions) + start_idx}] "
                f"{'✔️' if is_correct else '❌'} "
                f"expected={q['expected_answer']}, got={result['group_answer']} "
                f"({q_tokens:,} tokens)"
            )

    accuracy = total_correct / len(questions) * 100

    print(f"\n{'=' * 50}")
    print(f"Finished: {topology}")
    print(f"Accuracy: {total_correct}/{len(questions)} ({accuracy:.1f}%)")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Saved to: {output_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multi-agent topology experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all topologies with config defaults:
    python -m experiments.run_experiment

  Run specific topology:
    python -m experiments.run_experiment --topology full

  Run with 200 questions, 10 rounds, custom seed:
    python -m experiments.run_experiment --questions 200 --rounds 10 --seed 42

  Run questions 100-200 (skip first 100):
    python -m experiments.run_experiment --questions 200 --start 100
        """,
    )
    parser.add_argument(
        "--topology",
        type=str,
        choices=TOPOLOGIES,
        default=None,
        help="Topology to run. If not set, runs all topologies.",
    )
    parser.add_argument(
        "--questions",
        "-q",
        type=int,
        default=None,
        help=f"Number of questions to run (default: {NUM_QUESTIONS} from config)",
    )
    parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=None,
        help=f"Number of reasoning rounds (default: {NUM_ROUNDS} from config)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help=f"Dataset seed (default: {DATASET_SEED} from config)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting question index (0-based). Skips this many questions.",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=None,
        help=f"Temperature for LLM sampling (default: {TEMPERATURE} from config)",
    )
    parser.add_argument(
        "--base-url",
        "-u",
        type=str,
        default=None,
        help=f"Ollama base URL (default: {OLLAMA_BASE_URL} from config)",
    )
    args = parser.parse_args()

    topologies = [args.topology] if args.topology else TOPOLOGIES
    for t in topologies:
        run_experiment(
            t,
            start_idx=args.start,
            num_questions=args.questions,
            num_rounds=args.rounds,
            seed=args.seed,
            temperature=args.temperature,
            base_url=args.base_url,
        )

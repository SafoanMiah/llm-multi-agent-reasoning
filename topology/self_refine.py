"""
Self-Refine Topology (Baseline)

A single agent answers, then reviews and revises its own response
over multiple rounds — no peer information, no other models.

This serves as a token-matched baseline: does spending more tokens
on self-refinement with one model match multi-agent collaboration?
"""

from core.agent import Agent
from core.config import NUM_ROUNDS


def run_self_refine(
    agents: list[Agent], question: str, num_rounds: int = NUM_ROUNDS
) -> dict:
    """
    Run self-refine for a single question.

    Uses only the first agent in the list (single model baseline).
    The agent answers independently, then revises its own response
    each round with no external information.
    """
    agent = agents[0]
    agent.reset()

    round_log = []

    # Round 1: initial answer
    r = agent.initial_response(question)
    responses = [
        {
            "agent_id": agent.agent_id,
            "model": agent.client.model,
            "answer": r["answer"],
            "confidence": r["confidence"],
            "reasoning": r["reasoning"],
            "parse_failed": r["parse_failed"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
        }
    ]
    round_log.append({"round": 1, "responses": responses})

    # Rounds 2+: self-review with no peer info
    for round_num in range(2, num_rounds + 1):
        peer_info = (
            "No other agents were consulted. "
            "Review your own reasoning carefully for any errors, "
            "then provide your revised answer."
        )
        r = agent.revise(question, peer_info=peer_info)
        responses = [
            {
                "agent_id": agent.agent_id,
                "model": agent.client.model,
                "answer": r["answer"],
                "confidence": r["confidence"],
                "reasoning": r["reasoning"],
                "parse_failed": r["parse_failed"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
            }
        ]
        round_log.append({"round": round_num, "responses": responses})

    # Final answer is just the last round's answer (single agent, no vote)
    group_answer = responses[0]["answer"]

    return {
        "group_answer": group_answer,
        "rounds": round_log,
    }

"""
Mediator Topology

A mediator agent summarises all responses each round.
Agents only see the mediator's summary, never each other's raw responses.
"""

from core.agent import Agent
from core.llm_client import LLMClient
from core.config import MODELS, NUM_ROUNDS

MEDIATOR_PROMPT = """
You are a mediator summarising the responses of {n} mathematical reasoning agents.

Here are their responses to this problem:

Question:
{question}

Agent Responses:
{agent_responses}

Write a brief, neutral summary that includes:
- What answers were proposed and how many agents chose each
- The key reasoning behind each answer
- Any notable disagreements or errors in reasoning

Be factual and concise. Do not state which answer is correct.
"""


def format_agent_responses(responses: list[dict]) -> str:
    """Format agent responses for the mediator prompt."""
    lines = []
    for r in responses:
        lines.append(
            f"Agent {r['agent_id']}: answer={r['answer']}, "
            f"confidence={r['confidence']}, "
            f"reasoning: {r['reasoning']}"
        )
    return "\n".join(lines)


def run_mediator(
    agents: list[Agent], question: str, num_rounds: int = NUM_ROUNDS
) -> dict:
    """
    Run mediator topology for a single question across multiple rounds.

    Returns:
        dict with 'group_answer' and 'rounds' (list of per-round data).
    """
    mediator_client = LLMClient(model=MODELS[0])

    for agent in agents:
        agent.reset()

    round_log = []

    # Round 1: all agents answer independently
    responses = []
    for agent in agents:
        r = agent.initial_response(question)
        responses.append(
            {
                "agent_id": agent.agent_id,
                "model": agent.client.model,
                "answer": r["answer"],
                "confidence": r["confidence"],
                "reasoning": r["reasoning"],
            }
        )

    round_log.append({"round": 1, "responses": responses, "summary": None})

    # Rounds 2+: mediator summarises, agents revise
    for round_num in range(2, num_rounds + 1):
        # Mediator generates summary
        summary = mediator_client.prompt(
            MEDIATOR_PROMPT.format(
                n=len(agents),
                question=question,
                agent_responses=format_agent_responses(responses),
            )
        )

        # Each agent revises based on the mediator summary
        responses = []
        for agent in agents:
            r = agent.revise(question, peer_info=f"Mediator summary:\n{summary}")
            responses.append(
                {
                    "agent_id": agent.agent_id,
                    "model": agent.client.model,
                    "answer": r["answer"],
                    "confidence": r["confidence"],
                    "reasoning": r["reasoning"],
                }
            )

        round_log.append(
            {
                "round": round_num,
                "responses": responses,
                "summary": summary,
            }
        )

    # Final answer: majority vote from last round
    final_answers = [r["answer"] for r in responses if r["answer"] is not None]

    if not final_answers:
        return {"group_answer": None, "rounds": round_log}

    counts = {}
    for a in final_answers:
        counts[a] = counts.get(a, 0) + 1

    # Pick answer with most votes (first if multiple share max)
    group_answer = max(counts, key=counts.get)

    return {
        "group_answer": group_answer,
        "rounds": round_log,
    }

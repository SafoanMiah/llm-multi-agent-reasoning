"""
Fully Connected Topology

Each agent sees all other agents responses before revising.
No mediator, agents communicate directly with each other.
"""

from core.agent import Agent
from core.config import NUM_ROUNDS


def format_peer_responses(agent_id: int, responses: list[dict]) -> str:
    """Format all other agents' responses as peer info for a given agent."""
    lines = []
    for r in responses:
        if r["agent_id"] != agent_id:
            lines.append(
                f"Agent {r['agent_id']}: answer={r['answer']}, "
                f"confidence={r['confidence']}, "
                f"reasoning: {r['reasoning']}"
            )
    return "Other agents' responses:\n" + "\n".join(lines)


def run_fully_connected(
    agents: list[Agent], question: str, num_rounds: int = NUM_ROUNDS
) -> dict:
    """Run fully connected topology for a single question across multiple rounds."""
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
                "answer": r["answer"],
                "confidence": r["confidence"],
                "reasoning": r["reasoning"],
            }
        )

    round_log.append({"round": 1, "responses": responses})

    # Rounds 2+: each agent sees all others and revises
    for round_num in range(2, num_rounds + 1):
        new_responses = []
        for agent in agents:
            peer_info = format_peer_responses(agent.agent_id, responses)
            r = agent.revise(question, peer_info=peer_info)
            new_responses.append(
                {
                    "agent_id": agent.agent_id,
                    "answer": r["answer"],
                    "confidence": r["confidence"],
                    "reasoning": r["reasoning"],
                }
            )

        responses = new_responses
        round_log.append({"round": round_num, "responses": responses})

    # Final answer: majority vote from last round
    final_answers = [r["answer"] for r in responses if r["answer"] is not None]
    counts = {}
    for a in final_answers:
        counts[a] = counts.get(a, 0) + 1
    group_answer = max(counts, key=counts.get)

    return {
        "group_answer": group_answer,
        "rounds": round_log,
    }

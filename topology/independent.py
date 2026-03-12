"""
Independent Topology

Agents solve problems without seeing each other's responses.
Final answer determined by majority vote, first tiebreaker.
"""

from core.agent import Agent


def majority_vote(responses: list[dict]) -> dict:
    """Pick the most common answer. If tied, pick first."""
    answers = [r["answer"] for r in responses if r["answer"] is not None]

    if not answers:
        return {"answer": None, "method": "none"}

    # Count occurrences
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1

    # Pick answer with most votes (first if multiple share max)
    return {"answer": max(counts, key=counts.get), "method": "majority"}


def run_independent(agents: list[Agent], question: str) -> dict:
    """Run independent topology for a single question. Single round, no communication."""
    for agent in agents:
        agent.reset()

    responses = [agent.initial_response(question) for agent in agents]

    vote = majority_vote(responses)

    return {
        "group_answer": vote["answer"],
        "vote_method": vote["method"],
        "agent_responses": [
            {
                "agent_id": agent.agent_id,
                "model": agent.client.model,
                "answer": responses[i]["answer"],
                "confidence": responses[i]["confidence"],
                "reasoning": responses[i]["reasoning"],
            }
            for i, agent in enumerate(agents)
        ],
    }

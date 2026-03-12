"""
Independent Topology

Agents solve problems without seeing each other's responses.
Final answer determined by majority vote, highest confidence as tiebreaker.
"""

from core.agent import Agent


def majority_vote(responses: list[dict]) -> dict:
    """Pick the most common answer. If tied, pick highest confidence."""
    answers = [r["answer"] for r in responses]

    for a in answers:
        if answers.count(a) > len(answers) // 2:
            return {"answer": a, "method": "majority"}

    # No majority case
    best = max(responses, key=lambda r: r["confidence"])
    return {"answer": best["answer"], "method": "confidence"}


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
                "answer": responses[i]["answer"],
                "confidence": responses[i]["confidence"],
                "reasoning": responses[i]["reasoning"],
            }
            for i, agent in enumerate(agents)
        ],
    }

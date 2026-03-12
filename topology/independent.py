"""
Independent Topology

Agents solve problems without seeing each other's responses.
Final answer determined by majority vote, first tiebreaker.
"""

from core.agent import Agent
from topology.voting import VOTING_METHOD


def run_independent(agents: list[Agent], question: str) -> dict:
    """Run independent topology for a single question. Single round, no communication."""
    for agent in agents:
        agent.reset()

    responses = [agent.initial_response(question) for agent in agents]

    vote = VOTING_METHOD([r["answer"] for r in responses])

    return {
        "group_answer": vote,
        "agent_responses": [
            {
                "agent_id": agent.agent_id,
                "model": agent.client.model,
                "answer": responses[i]["answer"],
                "confidence": responses[i]["confidence"],
                "reasoning": responses[i]["reasoning"],
                "parse_failed": responses[i]["parse_failed"],
                "prompt_tokens": responses[i]["prompt_tokens"],
                "completion_tokens": responses[i]["completion_tokens"],
                "response_time_s": responses[i]["response_time_s"],
            }
            for i, agent in enumerate(agents)
        ],
    }

"""
Chain Topology

Agents answer sequentially, each seeing only the previous agent's response.
The chain runs 4 times with a different starting agent each time,
so every agent gets a turn as the final (most informed) agent.
Final answer is majority vote from the 4 chain-ending agents.
"""

from core.agent import Agent


def run_single_chain(
    agents: list[Agent], question: str, order: list[int]
) -> list[dict]:
    """Run one chain pass in the given agent order."""
    responses = []

    for i, agent_idx in enumerate(order):
        agent = agents[agent_idx]
        agent.reset()

        if i == 0:
            # First in chain: answer blind
            r = agent.initial_response(question)
        else:
            # See previous agent's response only
            prev = responses[-1]
            peer_info = (
                f"Previous agent's response:\n"
                f"Agent {prev['agent_id']}: answer={prev['answer']}, "
                f"confidence={prev['confidence']}, "
                f"reasoning: {prev['reasoning']}"
            )
            r = agent.initial_response(question)
            # Reset and use revise instead so peer info is included
            agent.history.pop()
            r = agent.revise(question, peer_info=peer_info)

        responses.append(
            {
                "agent_id": agent_idx,
                "answer": r["answer"],
                "confidence": r["confidence"],
                "reasoning": r["reasoning"],
            }
        )

    return responses


def run_chain(agents: list[Agent], question: str) -> dict:
    """Run chain topology: 4 rotations, each agent ends once."""
    n = len(agents)
    chain_log = []
    final_answers = []

    for start in range(n):
        order = [(start + i) % n for i in range(n)]
        responses = run_single_chain(agents, question, order)

        chain_log.append(
            {
                "chain": start + 1,
                "order": order,
                "responses": responses,
            }
        )

        # Last agent in chain is the most informed
        last = responses[-1]
        if last["answer"] is not None:
            final_answers.append(last["answer"])

    # Majority vote from the 4 chain-ending agents
    if not final_answers:
        group_answer = None
    else:
        counts = {}
        for a in final_answers:
            counts[a] = counts.get(a, 0) + 1
        group_answer = max(counts, key=counts.get)

    return {
        "group_answer": group_answer,
        "chains": chain_log,
    }

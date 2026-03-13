"""
Mediator Topology

A mediator agent summarises all responses each round.
Agents only see of mediator's summary, never each other's raw responses.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from core.agent import Agent
from core.llm_client import LLMClient
from core.config import MODELS, NUM_ROUNDS
from topology.voting import VOTING_METHOD

MEDIATOR_PROMPT = """
You are a mediator summarising the responses of {n} mathematical reasoning agents.

Here are their responses to this problem:

{question}

{agent_responses}

Write a brief, neutral summary that includes:
- What answers were proposed and how many agents chose each
- The key reasoning behind each answer
- Any notable disagreements or errors in reasoning

Be factual and concise. Do not state which answer is correct.
"""


def format_agent_responses(responses: list[dict]) -> str:
    """Format agent responses for mediator prompt."""
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
    """Run mediator topology for a single question across multiple rounds."""
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
                "answer": r["answer"],
                "confidence": r["confidence"],
                "reasoning": r["reasoning"],
                "parse_failed": r["parse_failed"],
                "prompt_tokens": r["prompt_tokens"],
                "completion_tokens": r["completion_tokens"],
            }
        )

    round_log.append({"round": 1, "responses": responses, "summary": None})

    # Rounds 2+: mediator summarises, agents revise (parallelized)
    for round_num in range(2, num_rounds + 1):
        # Mediator generates summary
        summary = mediator_client.prompt(
            MEDIATOR_PROMPT.format(
                n=len(agents),
                question=question,
                agent_responses=format_agent_responses(responses),
            )
        )

        # Each agent revises based on mediator summary (in parallel)
        responses = []
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            peer_info = f"Mediator summary:\n{summary}"
            future_to_agent = {
                executor.submit(agent.revise, question, peer_info): agent
                for agent in agents
            }

            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                r = future.result()
                responses.append(
                    {
                        "agent_id": agent.agent_id,
                        "answer": r["answer"],
                        "confidence": r["confidence"],
                        "reasoning": r["reasoning"],
                        "parse_failed": r["parse_failed"],
                        "prompt_tokens": r["prompt_tokens"],
                        "completion_tokens": r["completion_tokens"],
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
    group_answer = VOTING_METHOD([r["answer"] for r in responses])

    return {
        "group_answer": group_answer,
        "rounds": round_log,
    }

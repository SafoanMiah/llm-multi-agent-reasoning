from core.llm_client import LLMClient
from core.agent import Agent
from core.config import MODELS
from topology.independent import run_independent

QUESTION = "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"


def test_independent():
    """Test independent topology with 4 agents."""
    agents = [Agent(i, LLMClient(model=model)) for i, model in enumerate(MODELS)]

    result = run_independent(agents, QUESTION)

    print(f"Group answer: {result['group_answer']}")
    print(f"Vote method: {result['vote_method']}")
    for agent in result["agent_responses"]:
        print(
            f"  Agent {agent['agent_id']}: {agent['answer']} (confidence: {agent['confidence']})"
        )

    assert result["group_answer"] is not None, "No group answer"
    print("Independent topology test passed.")


if __name__ == "__main__":
    test_independent()

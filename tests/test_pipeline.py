from core.llm_client import LLMClient
from core.agent import Agent
from core.config import MODELS
from topology.independent import run_independent
from topology.mediator import run_mediator
from topology.fully_connected import run_fully_connected

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


def test_mediator():
    """Test mediator topology with 4 agents and 2 rounds."""
    agents = [Agent(i, LLMClient(model=model)) for i, model in enumerate(MODELS)]

    result = run_mediator(agents, QUESTION, num_rounds=2)

    print(f"Group answer: {result['group_answer']}")
    for round_data in result["rounds"]:
        print(f"\n--- Round {round_data['round']} ---")
        if round_data["summary"]:
            print(f"Mediator: {round_data['summary']}")
        for agent in round_data["responses"]:
            print(
                f"  Agent {agent['agent_id']} ({agent['model']}): "
                f"{agent['answer']} (confidence: {agent['confidence']})"
            )

    assert result["group_answer"] is not None, "No group answer"
    assert len(result["rounds"]) == 2, "Expected 2 rounds"
    print("Mediator topology test passed.")


def test_fully_connected():
    """Test fully connected topology with 4 agents and 2 rounds."""
    agents = [Agent(i, LLMClient(model=model)) for i, model in enumerate(MODELS)]

    result = run_fully_connected(agents, QUESTION, num_rounds=2)

    print(f"Group answer: {result['group_answer']}")
    for round_data in result["rounds"]:
        print(f"\n--- Round {round_data['round']} ---")
        for agent in round_data["responses"]:
            print(
                f"  Agent {agent['agent_id']} ({agent['model']}): "
                f"{agent['answer']} (confidence: {agent['confidence']})"
            )

    assert result["group_answer"] is not None, "No group answer"
    assert len(result["rounds"]) == 2, "Expected 2 rounds"
    print("Fully connected topology test passed.")


if __name__ == "__main__":
    test_fully_connected()
    # print("\n" + "="*50 + "\n")
    # test_mediator()
    # print("\n" + "="*50 + "\n")
    # test_independent()

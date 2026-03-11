from core.llm_client import LLMClient
from core.agent import Agent


def test_initial_response():
    """Test that an agent produces a structured response."""
    client = LLMClient()
    agent = Agent(agent_id=0, client=client)
    result = agent.initial_response("If a book costs £8 and you buy 3, how much do you spend?")
    print(f"Agent response: {result}")

    assert result["answer"] is not None, "Agent returned no answer"
    assert 0 <= result["confidence"] <= 100, "Confidence error"
    print("Initial response test passed.")


def test_revision():
    """Test that an agent can revise given peer info."""
    client = LLMClient()
    agent = Agent(agent_id=0, client=client)

    question = "If a book costs £8 and you buy 3, how much do you spend?"
    agent.initial_response(question)

    peer_info = "Agent 1 answered 24 with confidence 90. Agent 2 answered 24 with confidence 85."
    revised = agent.revise(question, peer_info)
    print(f"Revised response: {revised}")

    assert revised["answer"] is not None, "Revised answer is None"
    assert len(agent.history) == 2, "History should have 2 entries"
    print("Revision test passed.")


if __name__ == "__main__":
    test_initial_response()
    # test_revision()
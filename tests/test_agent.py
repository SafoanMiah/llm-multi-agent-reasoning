from core.llm_client import LLMClient
from core.agent import Agent


def test_initial_response():
    """Test that an agent produces a structured response."""
    client = LLMClient()
    agent = Agent(agent_id=0, client=client)
    result = agent.initial_response(
        "If a book costs £8 and you buy 3, how much do you spend?"
    )
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


def test_parse_valid_json():
    """Test parsing a valid JSON response."""
    client = LLMClient()
    agent = Agent(agent_id=0, client=client)

    raw = '{"answer": 42, "confidence": 95, "reasoning": "Simple calculation"}'
    result = agent.parse_response(raw)

    assert result["answer"] == 42, f"Expected 42, got {result['answer']}"
    assert result["confidence"] == 95, f"Expected 95, got {result['confidence']}"
    assert result["reasoning"] == "Simple calculation"
    assert result["raw"] == raw
    print("Valid JSON parsing test passed.")


def test_parse_json_with_extra_text():
    """Test parsing JSON embedded in extra text."""
    client = LLMClient()
    agent = Agent(agent_id=0, client=client)

    raw = 'Here\'s my answer: {"answer": 24, "confidence": 80, "reasoning": "8 * 3"} That\'s it!'
    result = agent.parse_response(raw)

    assert result["answer"] == 24
    assert result["confidence"] == 80
    assert result["reasoning"] == "8 * 3"
    assert result["raw"] == raw
    print("JSON with extra text parsing test passed.")


if __name__ == "__main__":
    test_initial_response()
    test_revision()
    test_parse_valid_json()
    test_parse_json_with_extra_text()

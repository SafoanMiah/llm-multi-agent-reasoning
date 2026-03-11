from core.llm_client import LLMClient


def test_connection():
    """Check with simple prompt."""
    client = LLMClient()
    response = client.prompt("What is 2 + 2?")
    print(f"Model: {client.model}")
    print(f"Response: {response}")

    assert response.strip(), "Empty response from model"
    print("Connection test passed.")


def test_system_message():
    """Check system messages are respected."""
    client = LLMClient()
    response = client.prompt(
        "What is 10 + 5?",
        system_message="You are a maths assistant. Reply with only the numerical answer."
    )
    print(f"System message test response: {response}")
    
    assert response.strip(), "Empty response from model"
    print("System message test passed.")


if __name__ == "__main__":
    test_connection()
    test_system_message()
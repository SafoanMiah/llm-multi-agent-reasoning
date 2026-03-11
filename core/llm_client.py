import requests


class LLMClient:
    """Wrapper for Ollama API to send prompts and get responses."""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
    ):
        """
        model:
        - qwen2.5:7b-instruct
        - llama3.1:8b
        - mistral:7b-instruct
        """
        self.model = model
        self.base_url = base_url

    def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Send messages to the model and return the response text.
        0.7 temp is a good default for a varied responses, lower for more deterministic output."""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature},
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def prompt(
        self, user_message: str, system_message: str = None, temperature: float = 0.7
    ) -> str:
        """Method for a single user prompt with system message."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return self.chat(messages, temperature=temperature)

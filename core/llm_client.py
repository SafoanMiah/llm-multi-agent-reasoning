import requests
from core.config import MODELS, OLLAMA_BASE_URL, TEMPERATURE


class LLMClient:
    """Wrapper for Ollama API to send prompts and get responses."""

    def __init__(
        self,
        model: str = None,
        base_url: str = OLLAMA_BASE_URL,
    ):
        """
        model:
        - gemma3:4b (Google)
        - phi4-mini (Microsoft)
        - llama3.2:3b (Meta)
        - qwen2.5:3b-instruct (Alibaba)
        """
        self.model = model if model else MODELS[0]
        self.base_url = base_url

    def chat(self, messages: list[dict], temperature: float = TEMPERATURE) -> str:
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
        self, user_message: str, system_message: str = None, temperature: float = TEMPERATURE
    ) -> str:
        """Method for a single user prompt with system message."""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return self.chat(messages, temperature=temperature)

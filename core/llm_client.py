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

        # Stats from the last call (for backward compatibility)
        self.last_prompt_tokens: int = 0
        self.last_completion_tokens: int = 0

    def chat(self, messages: list[dict], temperature: float = TEMPERATURE) -> dict:
        """Send messages to the model and return a dict with response and stats.

        Returns:
            {
                "response_text": str,
                "prompt_tokens": int,
                "completion_tokens": int,
            }
        """
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
        data = response.json()

        # Extract token counts from Ollama response
        result = {
            "response_text": data["message"]["content"],
            "prompt_tokens": int(data.get("prompt_eval_count", 0)),
            "completion_tokens": int(data.get("eval_count", 0)),
        }

        # Store as instance attributes for backward compatibility
        self.last_prompt_tokens = result["prompt_tokens"]
        self.last_completion_tokens = result["completion_tokens"]

        return result

    def prompt(
        self, user_message: str, system_message: str = None, temperature: float = TEMPERATURE
    ) -> str:
        """Method for a single user prompt with system message.

        Returns just the response text for backward compatibility.
        Use self.last_prompt_tokens, self.last_completion_tokens
        to access stats after calling this method.
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        result = self.chat(messages, temperature=temperature)
        return result["response_text"]

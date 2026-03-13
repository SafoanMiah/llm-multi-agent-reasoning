import json
import re
from core.llm_client import LLMClient

AGENT_SYSTEM_PROMPT = """
You are a mathematical reasoning agent. Solve the given problem step by step.

Respond in EXACTLY this JSON format and nothing else:

{
    "answer": <your numerical answer>,
    "confidence": <0-100>,
    "reasoning": "<brief step-by-step reasoning>"
}

Rules:
- "answer" must be a single number (integer or float), no units or text
- "confidence" reflects how certain you are in your answer:
    - 90-100: You are very confident the answer is correct
    - 70-89: Fairly confident but some steps feel uncertain
    - 50-69: Unsure, multiple approaches seem plausible
    - Below 50: Guessing or very uncertain
  Do NOT default to 100. Be honest about your certainty.
- "reasoning" must show your working clearly in 2-3 sentences
- Output ONLY valid JSON, no extra text before or after
"""

REVISION_PROMPT_TEMPLATE = """
You previously answered this maths problem:

{question}

Your previous response:
{previous_response}

Here is what other agents think:
{peer_info}

Now reconsider. You may keep your answer if you believe it is correct, or change it if you see a mistake in your reasoning. Do not change your answer just because others disagree.

Respond in EXACTLY this JSON format and nothing else:

{{
    "answer": <your numerical answer>,
    "confidence": <0-100>,
    "reasoning": "<brief step-by-step reasoning>"
}}
"""


class Agent:
    """Reasoning agent that answers questions and tracks belief across rounds."""

    def __init__(self, agent_id: int, client: LLMClient):
        self.agent_id = agent_id
        self.client = client
        self.history = []  # list of parsed responses per round

    def parse_response(self, raw: str) -> dict:
        """Extract JSON from model response, with fallback."""
        try:
            # Find JSON in the response
            match = re.search(r"\{.*\}", raw, re.DOTALL)  # Incase of extra text
            if not match:
                raise ValueError("No JSON found in response")
            parsed = json.loads(match.group())
            return {
                "answer": float(parsed["answer"]) if "answer" in parsed else None,
                "confidence": int(parsed.get("confidence", 0)),
                "reasoning": str(parsed.get("reasoning", "")),
                "raw": raw,
                "parse_failed": False,
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            # Fallback for unparseable responses
            return {
                "answer": None,
                "confidence": None,  # Use None instead of 0 to distinguish parse failure
                "reasoning": "Failed to parse",
                "raw": raw,
                "parse_failed": True,
            }

    def initial_response(self, question: str) -> dict:
        """Generate first-round answer for a question."""
        raw = self.client.prompt(
            question, system_message=AGENT_SYSTEM_PROMPT, temperature=0.7
        )
        parsed = self.parse_response(raw)

        # Add token stats from the LLM call
        parsed["prompt_tokens"] = self.client.last_prompt_tokens
        parsed["completion_tokens"] = self.client.last_completion_tokens

        self.history.append(parsed)
        return parsed

    def revise(self, question: str, peer_info: str) -> dict:
        """Revise answer given peer information from the topology."""
        previous = json.dumps(self.history[-1]) if self.history else "{}"
        prompt = REVISION_PROMPT_TEMPLATE.format(
            question=question, previous_response=previous, peer_info=peer_info
        )
        raw = self.client.prompt(prompt, temperature=0.7)
        parsed = self.parse_response(raw)

        # Add token stats from the LLM call
        parsed["prompt_tokens"] = self.client.last_prompt_tokens
        parsed["completion_tokens"] = self.client.last_completion_tokens

        self.history.append(parsed)
        return parsed

    def reset(self):
        """Clear history for a new question."""
        self.history = []

import json
import re
from core.llm_client import LLMClient

AGENT_SYSTEM_PROMPT = """

You are a mathematical reasoning agent. When given a problem, respond in EXACTLY this JSON format and nothing else:

{
    "answer": <your numerical answer>,
    "confidence": <0-100>,
    "reasoning": "<brief step-by-step reasoning>"
}


Rules:
- "answer" must be a number (integer or float)
- "confidence" must be an whole integer from 0 to 100
- "reasoning" must be a short string explaining your steps
- Output ONLY valid JSON, no extra text
"""

REVISION_PROMPT_TEMPLATE = """
You are a mathematical reasoning agent. Here is the problem:

{question}

Your previous response was:
{previous_response}

{peer_info}

Reconsider your answer given the above. Respond in EXACTLY this JSON format and nothing else:

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
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {
                    "answer": parsed.get("answer"),
                    "confidence": parsed.get("confidence", 50),
                    "reasoning": parsed.get("reasoning", ""),
                    "raw": raw,
                }
        except:
            # Fallback for unparseable responses
            return {
                "answer": None,
                "confidence": 0,
                "reasoning": "Failed to parse",
                "raw": raw,
            }

    def initial_response(self, question: str) -> dict:
        """Generate first-round answer for a question."""
        raw = self.client.prompt(
            question, system_message=AGENT_SYSTEM_PROMPT, temperature=0.7
        )
        parsed = self.parse_response(raw)
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
        self.history.append(parsed)
        return parsed

    def reset(self):
        """Clear history for a new question."""
        self.history = []

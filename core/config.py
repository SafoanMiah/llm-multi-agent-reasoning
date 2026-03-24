"""
Centralized configuration for multi-agent LLM reasoning experiments.
"""

# Models used by the 4 agents

# Small
MODELS = [
    "phi4:14b",
    "phi4-mini",  # Microsoft
    "qwen2.5:3b-instruct",  # Alibaba
    "gemma3:4b",  # Google
    "llama3.2:3b",  # Meta
]

# Medium
# MODELS = ["qwen2.5:7b-instruct", "llama3.1:8b", "mistral:7b-instruct"]

# Rounds for mediator and full, intermidiate is 1, chain rounds = models
NUM_ROUNDS = 5
NUM_QUESTIONS = 200
DATASET_SEED = 0

# LLM inference parameters
TEMPERATURE = 0.4
OLLAMA_BASE_URL = "http://localhost:11434"

# Topologies available
TOPOLOGIES = ["independent", "chain", "full", "mediator", "self_refine"]

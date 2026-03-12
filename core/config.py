"""
Centralized configuration for multi-agent LLM reasoning experiments.
"""

# Models used by the 4 agents

# Small
MODELS = [
    "gemma3:4b",
    "phi4-mini",
    "llama3.2:3b",
    "qwen2.5:3b-instruct",
]

# MODELS = ["qwen2.5:7b-instruct", "llama3.1:8b", "mistral:7b-instruct"]

# Experimental parameters
NUM_AGENTS = 4
NUM_ROUNDS = 5
NUM_QUESTIONS = 150
DATASET_SEED = 42

# LLM inference parameters
TEMPERATURE = 0.7
OLLAMA_BASE_URL = "http://localhost:11434"

# Topologies available
TOPOLOGIES = ["independent", "full", "mediator"]

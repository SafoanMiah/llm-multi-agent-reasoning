### Communication Topology and Belief Dynamics in Multi-Agent LLM Reasoning Systems

#### Prerequisites

- Python 3.10+
- Ollama: https://ollama.com/
- `irm https://ollama.com/install.ps1 | iex`

#### Setup
1. Pull required models via Ollama:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull llama3.1:8b
   ollama pull mistral:7b-instruct
   ```

#### Usage
Run an experiment:
```bash
python experiments/run.py --topology full --model qwen2.5:7b-instruct
```

Available topologies: `independent`, `full`, `mediator`
### Communication Topology and Belief Dynamics in Multi-Agent LLM Reasoning Systems
This project investigates how communication topology affects reasoning in multi-agent LLM systems. Multiple agents solve GSM8K math problems under different communication structures (independent, fully connected, mediator-based) over iterative rounds. It'll compares accuracy, convergence, and cost across topologies, treating communication structure as the primary experimental variable.

#### Prerequisites

- Python 3.10+
- Ollama: https://ollama.com/
- Windows Install: `irm https://ollama.com/install.ps1 | iex`

#### Setup
1. Pull required models via Ollama:
   ```bash
   ollama pull qwen2.5:7b-instruct
   ollama pull llama3.1:8b
   ollama pull mistral:7b-instruct
   ```

#### Usage
Run an experiment [not yet implemented]:
```bash
python experiments/run.py --topology full --model qwen2.5:7b-instruct
```

Available topologies: `independent`, `full`, `mediator`
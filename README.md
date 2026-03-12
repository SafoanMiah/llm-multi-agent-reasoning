### Communication Topology and Belief Dynamics in Multi-Agent LLM Reasoning Systems
This project investigates how communication topology affects reasoning in multi-agent LLM systems. Multiple agents solve GSM8K math problems under different communication structures (independent, fully connected, mediator-based) over iterative rounds. It'll compares accuracy, convergence, and cost across topologies, treating communication structure as the primary experimental variable.

#### Prerequisites

- Python 3.10+
- Ollama: https://ollama.com/
- Windows Install: `irm https://ollama.com/install.ps1 | iex`

#### Setup
1. Pull required models via Ollama:
   ```bash
   ollama pull gemma3:4b
   ollama pull phi4-mini
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b-instruct
   ```

#### Usage
Run an experiment [not yet implemented]:
```bash
python experiments/run_experiment.py --topology independent --rounds 5 --questions 150
```

Available topologies: `independent`, `full`, `mediator`

**Configuration:**
- 4 agents (one each: gemma3:4b, phi4-mini, llama3.2:3b, qwen2.5:3b-instruct)
- 5 rounds of reasoning per question
- 150 questions from GSM8K test split
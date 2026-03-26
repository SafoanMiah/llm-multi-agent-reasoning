### Communication Topology and Belief Dynamics in Multi-Agent LLM Reasoning Systems
#### Notebook: [nbviewer link](https://nbviewer.org/github/SafoanMiah/llm-multi-agent-reasoning/blob/main/analysis/analysis.ipynb)   |   Project: [github link](https://github.com/SafoanMiah/llm-multi-agent-reasoning/tree/main)

This project investigates how communication topology affects reasoning in multi-agent LLM systems. Multiple agents solve GSM8K math problems under different communication structures (independent, fully connected, mediator-based, chain) over iterative rounds. It compares accuracy, convergence, and cost across topologies, treating communication structure as the primary experimental variable.

[![Topology Diagrams](figures/topology_diagrams.png)](analysis/figures/topology_diagrams.png)

#### Prerequisites

- Python 3.10+
- Ollama: https://ollama.com/
- Windows Install: `irm https://ollama.com/install.ps1 | iex`

#### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/SafoanMiah/llm-multi-agent-reasoning.git
   cd llm-multi-agent-reasoning
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

#### Setup
1. Pull required models via Ollama:
   ```bash
   ollama pull gemma3:4b
   ollama pull phi4-mini
   ollama pull llama3.2:3b
   ollama pull qwen2.5:3b-instruct
   ```

#### Usage

##### Basic Usage
Run all topologies with config defaults:
```bash
python -m experiments.run_experiment
```

Run specific topology:
```bash
python -m experiments.run_experiment --topology full
```

Available topologies: `independent`, `full`, `mediator`, `chain`, `self_refine`

##### Flags

| Flag | Short | Description | Default |
|------|-------|-------------|---------|
| `--topology` | - | Topology to run (runs all if omitted) | `independent, full, mediator, chain` |
| `--questions` | `-q` | Number of questions to run | `200` (from config) |
| `--rounds` | `-r` | Number of reasoning rounds per question | `5` (from config) |
| `--seed` | `-s` | Dataset random seed | `0` (from config) |
| `--start` | - | Starting question index (skip first N) | `0` |
| `--temperature` | `-t` | Temperature for LLM sampling | `0.4` (from config) |
| `--base-url` | `-u` | Ollama base URL | `http://localhost:11434` (from config) |

##### Examples

Run with custom questions and rounds:
```bash
python -m experiments.run_experiment --questions 100 --rounds 10
```

Run questions 100-200 (skip first 100):
```bash
python -m experiments.run_experiment --questions 200 --start 100
```

Run specific topology with custom settings:
```bash
python -m experiments.run_experiment --topology mediator --questions 50 --rounds 3
```

Run with higher temperature:
```bash
python -m experiments.run_experiment --temperature 0.8
```

Run with custom Ollama URL:
```bash
python -m experiments.run_experiment --base-url http://192.168.1.100:11434
```

##### Configuration

Base configuration is in [`core/config.py`](core/config.py). Command-line flags override config values.

Default settings:
- 4 agents (one each: gemma3:4b, phi4-mini, llama3.2:3b, qwen2.5:3b-instruct)
- 5 rounds of reasoning per question
- 200 questions from GSM8K test split
- Temperature: 0.4
- Ollama base URL: http://localhost:11434

##### Dataset

The GSM8K test dataset is included at [`dataset/gsm8k_test.jsonl`](dataset/gsm8k_test.jsonl).
No additional download required.

##### Testing

Run the test suite to verify your setup:
```bash
# Test individual agent behavior
python tests/test_agent.py

# Test topology pipelines
python tests/test_pipeline.py
```

**Note:** Tests require Ollama to be running with models already pulled.

##### Results

Experiment results are saved to `results/<config-folder>/`. Each run creates a timestamped CSV file with per-question, per-agent, and per-round data.

The analysis notebook [`analysis/analysis.ipynb`](analysis/analysis.ipynb) processes these results and generates figures in [`analysis/figures/`](analysis/figures/).
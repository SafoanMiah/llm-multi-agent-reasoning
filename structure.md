artefact/
main.py                 # entry point that runs the experiment pipeline
README.md               # instructions explaining how to run the project

core/
   llm_client.py       # wrapper that sends prompts to the LLM and returns responses
   agent.py            # reasoning agent answers questions, outputs answer/confidence

topology/
   independent.py      # agents solve problems without seeing each other
   fully_connected.py  # agents see all other agents' responses
   mediator.py         # mediator summarises responses and sends feedback to agents

dataset/
   gsm8k_loader.py     # loads a subset of GSM8K questions for experiments

experiments/
   run_experiment.py   # experiment loop that runs agents across questions and rounds

tests/
   test_model.py       # quick test to confirm model connection works
   test_agent.py       # test that an agent produces structured outputs
   test_pipeline.py    # small experiment to verify the whole system runs

results/                # saved experiment outputs (csv/json logs)

analysis/
   analysis.ipynb      # loads results and generates plots/figures
   figures/            # exported graphs used in the dissertation


PROJECT BUILD ORDER
3. Dataset loader
   dataset/gsm8k_loader.py

4. Communication structures
   topology/independent.py
   topology/fully_connected.py
   topology/mediator.py

5. Experiment pipeline
   experiments/run_experiment.py
   tests/test_pipeline.py

6. Main entry point
   main.py

7. Run experiments
   generate files in results/

8. Analysis
   analysis/analysis.ipynb
   export graphs to figures/

9. Documentation
   README.md
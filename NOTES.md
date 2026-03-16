Model choice

Models:
- Qwen2.5-7B-Instruct
- Llama-3.1-8B
- Mistral-7B-Instruct-v0.3

Reason:
- Strong reasoning performance
- Highly ranked on Hugging Face Open LLM Leaderboard
- 7–8B size fits local GPU
- Widely used open-source models

Note:
"Instruct" models are fine-tuned to follow instructions and answer tasks reliably.

Control:
Same model used across agents to isolate communication topology effects.

CHANGE:
Models (4 agents, all 3-4B parameters):
- Gemma 3 4B (Google, ~89% GSM8K) (Google)
- Phi-4-mini 3.8B (Microsoft, ~89% GSM8K) (Microsoft)
- Qwen 2.5 3B (Alibaba, ~79% GSM8K) (Alibaba) 
- Llama 3.2 3B (Meta, ~78% GSM8K) (Meta)

Why 4 agents instead of 3:
- With 3 agents, the structural difference between fully connected
  and mediator topologies is minimal (seeing 2 peers vs 1 summary).
- With 4 agents, fully connected means each agent sees 3 responses
  directly, while mediator compresses all 3 into a single summary.
  This makes the information filtering effect of the mediator
  much more measurable.

Why small (3-4B) models instead of 7-8B:
- Faster inference allows more rounds per question and more questions
  overall, producing stronger statistical results.
- Weaker individual agents leave more room for collaboration to
  demonstrate measurable improvement — if agents are already near
  perfect alone, topology effects get masked.
- All models are in a tight parameter range (3-4B), controlling for
  model size as a confounding variable.

Why 4 different models from 4 companies:
- Each model has different training data and architecture choices,
  introducing natural diversity in reasoning approaches.
- More realistic — real multi-agent systems often use heterogeneous
  models.
- Avoids the awkwardness of running duplicate instances of the
  same model.

Why 5 rounds:
- Smaller models run fast enough to support more rounds without
  excessive compute cost.
- More rounds provide richer convergence data — does agreement
  emerge by round 2, or does it take longer?
- Allows analysis of "what if we stopped at round 3" as a
  comparison point.

Dataset: ~100 questions from GSM8K test split
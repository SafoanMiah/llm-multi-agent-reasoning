import json
import random
from pathlib import Path

path = Path(__file__).parent / "gsm8k_test.jsonl"

with open(path, "r") as f:
    data = [json.loads(line) for line in f if line.strip()]

random.seed(42)
sample = random.sample(data, min(1, len(data)))

for item in sample:
    print(item["answer"].split("#### ")[-1])

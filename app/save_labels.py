import json
from datasets import load_dataset

dataset = load_dataset(
    "PolyAI/banking77",
    trust_remote_code=True
)

label_names = dataset["train"].features["label"].names

with open("data/label_names.json", "w", encoding="utf-8") as f:
    json.dump(label_names, f, ensure_ascii=False, indent=2)

print("Correspondance des intentions sauvegardée.")
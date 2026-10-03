from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Télécharger le jeu de données BANKING77
dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)

# Convertir les données en fichiers CSV
Path("data").mkdir(exist_ok=True)

for split in ["train", "test"]:
    df = pd.DataFrame(dataset[split])
    df.to_csv(f"data/banking77_{split}.csv", index=False)

print("BANKING77 téléchargé et enregistré dans le dossier data/")
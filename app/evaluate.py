import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# Charger les données
test_df = pd.read_csv("data/banking77_test.csv")

X_test = test_df["text"].tolist()
y_test = test_df["label"]


# Charger le modèle
encoder = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

classifier = joblib.load(
    "data/multilingual_intent_model.joblib"
)


# Transformer les questions en embeddings
X_test_embeddings = encoder.encode(
    X_test,
    show_progress_bar=True
)


# Prédictions
y_pred = classifier.predict(
    X_test_embeddings
)


# Métriques
accuracy = accuracy_score(
    y_test,
    y_pred
)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="weighted"
)


print("\n===== ÉVALUATION BANKBOT =====")
print(f"Accuracy  : {accuracy * 100:.2f} %")
print(f"Precision : {precision * 100:.2f} %")
print(f"Recall    : {recall * 100:.2f} %")
print(f"F1-score  : {f1 * 100:.2f} %")
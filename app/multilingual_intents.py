import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Charger les données
train_df = pd.read_csv("data/banking77_train.csv")
test_df = pd.read_csv("data/banking77_test.csv")

X_train = train_df["text"].tolist()
y_train = train_df["label"]

X_test = test_df["text"].tolist()
y_test = test_df["label"]


# Charger le modèle multilingue
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Modèle multilingue chargé.")


# Transformer les questions en représentations numériques
X_train_embeddings = encoder.encode(
    X_train,
    show_progress_bar=True
)

X_test_embeddings = encoder.encode(
    X_test,
    show_progress_bar=True
)


# Entraîner le classifieur
classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(X_train_embeddings, y_train)

print("Classifieur entraîné.")


# Évaluer le modèle
y_pred = classifier.predict(X_test_embeddings)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", round(accuracy * 100, 2), "%")


# Sauvegarder le classifieur
joblib.dump(
    classifier,
    "data/multilingual_intent_model.joblib"
)

print("Modèle sauvegardé.")
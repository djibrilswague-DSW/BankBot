import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report

# Charger les données
train_df = pd.read_csv("data/banking77_train.csv")

# Séparer les textes et les labels
X_train = train_df["text"]
y_train = train_df["label"]


# Créer le pipeline de classification
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])


# Entraîner le modèle
model.fit(X_train, y_train)

print("Modèle entraîné avec succès.")

# Tester une nouvelle demande
question = "Where is my bank card?"

prediction = model.predict([question])[0]

# Récupérer le nom correspondant au label
label_names = load_dataset(
    "PolyAI/banking77",
    trust_remote_code=True
)["train"].features["label"].names

print("Question :", question)
print("Intention prédite :", label_names[prediction])

# Évaluer le modèle sur les données de test
test_df = pd.read_csv("data/banking77_test.csv")

X_test = test_df["text"]
y_test = test_df["label"]

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")
print("\nRapport de classification :")
print(classification_report(y_test, y_pred, zero_division=0))

# Sauvegarder le modèle entraîné
joblib.dump(model, "data/intent_model.joblib")

print("\nModèle sauvegardé dans data/intent_model.joblib")
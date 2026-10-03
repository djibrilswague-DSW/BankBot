import joblib
from sentence_transformers import SentenceTransformer


# Charger le modèle multilingue
encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Charger le classifieur
classifier = joblib.load("data/multilingual_intent_model.joblib")


questions = [
    "Je n'ai toujours pas reçu ma carte bancaire.",
    "Ma carte bancaire ne fonctionne pas.",
    "Je voudrais changer mon code PIN.",
    "Je ne reconnais pas ce paiement.",
    "Comment puis-je approvisionner mon compte ?",
    "J'ai oublié mon mot de passe."
]


# Transformer les questions
embeddings = encoder.encode(questions)

# Prédire les intentions
predictions = classifier.predict(embeddings)


for question, prediction in zip(questions, predictions):
    print(f"\nQuestion : {question}")
    print(f"Intention prédite : {prediction}")
import json
import joblib
import faiss

from sentence_transformers import SentenceTransformer
from ollama import chat


# =========================
# 1. Chargement des modèles
# =========================

encoder = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

classifier = joblib.load(
    "data/multilingual_intent_model.joblib"
)

with open(
    "data/label_names.json",
    "r",
    encoding="utf-8"
) as f:
    label_names = json.load(f)


# =========================
# 2. Chargement du RAG
# =========================

index = faiss.read_index(
    "data/bank_faq.index"
)

with open(
    "data/bank_documents.txt",
    "r",
    encoding="utf-8"
) as f:
    documents = f.read().split("\n\n")


# =========================
# 3. Correspondance intention → document
# =========================

INTENT_TO_DOCUMENT = {
    "card_arrival": "Livraison de la carte",
    "card_not_working": "Carte bancaire non fonctionnelle",
    "lost_or_stolen_card": "Carte perdue ou volée",
    "change_pin": "Code PIN",
    "card_payment_not_recognised": "Paiement non reconnu",
    "declined_card_payment": "Paiement refusé",
    "transaction_charged_twice": "Paiement en double",
    "transfer_not_received_by_recipient": "Virement non reçu",
    "failed_transfer": "Virement refusé",
    "passcode_forgotten": "Mot de passe oublié",
    "verify_my_identity": "Vérification d'identité",
    "topping_up_by_card": "Approvisionner le compte",
    "top_up_failed": "Échec d'un approvisionnement",
}


# =========================
# 4. Analyse de la question
# =========================

def analyze_question(question):

    question_embedding = encoder.encode(
        [question],
        convert_to_numpy=True
    )

    probabilities = classifier.predict_proba(
        question_embedding
    )[0]

    prediction = probabilities.argmax()

    confidence = probabilities[prediction]

    intent = label_names[prediction]

    target = INTENT_TO_DOCUMENT.get(intent)

    if target is None:
        return intent, None, confidence

    for document in documents:

        if target.lower() in document.lower():

            return intent, document, confidence

    return intent, None, confidence

# =========================
# 5. Génération de la réponse
# =========================

def generate_answer(question, document):

    if document is None:

        return (
            "Je ne dispose pas actuellement d'informations "
            "suffisantes pour répondre à cette question."
        )

    prompt = f"""
Tu es BankBot, un assistant conversationnel pour la relation client bancaire.

Réponds en français.

Tu dois répondre UNIQUEMENT à partir des informations présentes
dans la documentation ci-dessous.

N'invente aucune information.
Ne complète pas la documentation avec tes propres connaissances.

Documentation :
{document}

Question du client :
{question}

Donne une réponse claire, courte et professionnelle.
"""

    response = chat(
        model="qwen2.5:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content


# =========================
# 6. Pipeline complet BankBot
# =========================

def bankbot(question):

    intent, document, confidence = analyze_question(question)

    if confidence < 0.30:

        answer = (
            "Je ne suis pas suffisamment certain de comprendre "
            "votre demande. Pouvez-vous reformuler votre question ?"
        )

    else:

        answer = generate_answer(
            question,
            document
        )

    return {
        "question": question,
        "intent": intent,
        "confidence": confidence,
        "document": document,
        "answer": answer
    }

# =========================
# 7. Test
# =========================

if __name__ == "__main__":

    questions = [
        "Ma carte bancaire ne fonctionne plus.",
        "Je ne reconnais pas ce paiement.",
        "Je voudrais changer mon code PIN.",
        "Ma carte n'est toujours pas arrivée.",
        "Je voudrais faire quelque chose avec mon compte."
    ]

    for question in questions:

        result = bankbot(question)

        print("\n" + "=" * 60)
        print("Question :", result["question"])
        print("Intention :", result["intent"])
        print("Confiance :", round(result["confidence"] * 100, 2), "%")
        print("Réponse :", result["answer"])

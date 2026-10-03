from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# Charger les documents
documents = Path("data/bank_documents.txt").read_text(
    encoding="utf-8"
).split("\n\n")


# Charger le modèle multilingue
encoder = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)


# Charger l'index FAISS
index = faiss.read_index("data/bank_faq.index")


def search(question, top_k=3):
    question_embedding = encoder.encode(
        [question],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for distance, index_position in zip(
        distances[0],
        indices[0]
    ):
        results.append({
            "document": documents[index_position],
            "distance": float(distance)
        })

    return results


if __name__ == "__main__":

    question = "Je n'ai toujours pas reçu ma carte bancaire."

    results = search(question)

    print("\nQuestion :", question)
    print("\nRésultats :")

    for i, result in enumerate(results, start=1):
        print(f"\n--- Résultat {i} ---")
        print(result["document"])
        print("Distance :", round(result["distance"], 4))
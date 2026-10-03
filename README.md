# BankBot — Agent IA conversationnel pour la relation client bancaire

BankBot est un agent conversationnel basé sur l'intelligence artificielle, conçu pour automatiser le traitement de demandes courantes dans le domaine bancaire.

Le système combine **classification d'intentions multilingue**, **recherche documentaire**, **LLM local** et **gestion de l'incertitude** afin de produire des réponses en français à partir d'une documentation bancaire contrôlée.

> Projet réalisé à des fins de démonstration avec des données publiques et une documentation bancaire fictive. Aucune donnée client réelle n'est utilisée.

---

## Objectif

L'objectif de BankBot est de transformer une question formulée naturellement par un client en une réponse contextualisée et contrôlée.

Exemples de demandes :

* « Je ne reconnais pas ce paiement. »
* « Ma carte bancaire ne fonctionne plus. »
* « Je voudrais changer mon code PIN. »
* « Ma carte n'est toujours pas arrivée. »
* « Un virement que j'ai effectué n'est toujours pas reçu. »

---

## Architecture

```text
Question utilisateur
        │
        ▼
Encodeur multilingue
Sentence Transformers
        │
        ▼
Classification d'intention
        │
        ▼
Intention BANKING77
        │
        ▼
Sélection de la documentation pertinente
        │
        ▼
Documentation bancaire
        │
        ▼
LLM local — Qwen2.5 3B
        │
        ▼
Réponse en français
        │
        ▼
Interface Gradio
```

BankBot utilise une approche hybride combinant **classification d'intention** et **retrieval documentaire guidé par l'intention**.

---

## Fonctionnalités

* Classification de **77 intentions bancaires**
* Compréhension de questions en français
* Encodeur multilingue `paraphrase-multilingual-MiniLM-L12-v2`
* Classification avec `LogisticRegression`
* Documentation bancaire utilisée comme source de réponse
* Génération de réponses avec **Qwen2.5 3B**
* LLM exécuté localement avec **Ollama**
* Seuil de confiance pour gérer les demandes ambiguës
* Interface conversationnelle avec **Gradio**
* Tests fonctionnels
* Conteneurisation avec **Docker**

---

## Évaluation

Le modèle de classification a été évalué sur le jeu de test de **BANKING77**.

| Métrique  |    Résultat |
| --------- | ----------: |
| Accuracy  | **91,98 %** |
| Precision | **92,28 %** |
| Recall    | **91,98 %** |
| F1-score  | **91,99 %** |

Évaluation réalisée sur **3 080 exemples de test**.

---

## Tests fonctionnels

BankBot dispose également de tests sur plusieurs cas d'utilisation :

| Question                | Intention                     | Résultat |
| ----------------------- | ----------------------------- | -------- |
| Paiement non reconnu    | `card_payment_not_recognised` | ✅        |
| Changement de PIN       | `change_pin`                  | ✅        |
| Carte non fonctionnelle | `card_not_working`            | ✅        |
| Carte non reçue         | `card_arrival`                | ✅        |
| Virement non reçu       | Gestion de l'incertitude      | ✅        |

**Résultat : 5/5 tests fonctionnels réussis.**

---

## Technologies

### Intelligence artificielle

* Python
* Scikit-learn
* Sentence Transformers
* BANKING77
* Qwen2.5 3B
* Ollama

### Recherche documentaire

* FAISS
* RAG / retrieval documentaire
* Documentation bancaire structurée

### Application

* Gradio
* Docker

### Développement

* Git
* GitHub
* Linux / Windows

---

## Structure du projet

```text
BankBot/
│
├── app/
│   ├── agent.py
│   ├── intents.py
│   ├── multilingual_intents.py
│   ├── rag.py
│   ├── evaluate.py
│   ├── main.py
│   └── ...
│
├── data/
│   ├── bank_faq.txt
│   ├── bank_documents.txt
│   └── label_names.json
│
├── tests/
│   └── test_bankbot.py
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

Les données d'entraînement BANKING77 et les modèles générés localement ne sont pas versionnés dans le dépôt.

---

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/VOTRE_USERNAME/BankBot.git
cd BankBot
```

### 2. Créer l'environnement Python

```bash
python -m venv .venv
```

Activation sous Windows :

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Installer Ollama

BankBot utilise un LLM local avec Ollama.

Installer Ollama puis télécharger le modèle :

```bash
ollama pull qwen2.5:3b
```

Lancer Ollama :

```bash
ollama serve
```

---

## Lancer BankBot

Depuis le dossier du projet :

```bash
python app/main.py
```

Puis ouvrir :

```text
http://localhost:7860
```

---

## Exécution avec Docker

Construire l'image :

```bash
docker build -t bankbot .
```

Lancer le conteneur :

```bash
docker run -d \
  --name bankbot \
  -p 7860:7860 \
  --add-host=host.docker.internal:host-gateway \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  bankbot
```

Puis accéder à :

```text
http://localhost:7860
```

Dans cette configuration, **BankBot est exécuté dans Docker tandis que le modèle Qwen est exécuté par Ollama sur la machine hôte**.

---

## Gestion de l'incertitude

BankBot ne génère pas systématiquement une réponse lorsqu'il est peu certain de l'intention détectée.

Lorsque la confiance du classifieur est inférieure au seuil défini, le système demande à l'utilisateur de reformuler sa demande.

Cette approche permet de limiter les réponses générées à partir d'une mauvaise interprétation de la question.

---

## Exemple

### Question

```text
Je ne reconnais pas ce paiement.
```

### Résultat

```text
Intention détectée :
card_payment_not_recognised

Confiance :
97,20 %
```

### Réponse

```text
Vérifiez la date et le montant de la transaction.
Si le paiement reste inconnu, veuillez contacter le service client.
```

---

## Perspectives

Plusieurs améliorations peuvent être ajoutées :

* amélioration de la classification des intentions proches ;
* enrichissement de la base documentaire ;
* véritable pipeline RAG avec recherche sémantique FAISS ;
* ajout de citations des sources dans les réponses ;
* mémoire conversationnelle ;
* authentification et gestion des utilisateurs ;
* observabilité et monitoring ;
* déploiement cloud ;
* API REST pour intégrer BankBot à une application bancaire ;
* amélioration du modèle LLM et comparaison de plusieurs modèles.

---

## Limites

BankBot est un **prototype de démonstration**.

Il utilise une documentation fictive et ne doit pas être utilisé pour traiter de véritables opérations bancaires ou des données personnelles.

Aucune connexion à un système bancaire réel n'est réalisée.

---

## Auteur

**Djibril Seye Wague**

Étudiant en Master 2 — Systèmes Intelligents & Vision par Ordinateur

Université Sorbonne Paris Nord — Institut Galilée

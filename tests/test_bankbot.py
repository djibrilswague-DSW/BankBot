import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.agent import bankbot


tests = [
    {
        "question": "Je ne reconnais pas ce paiement.",
        "expected_intent": "card_payment_not_recognised"
    },
    {
        "question": "Je voudrais changer mon code PIN.",
        "expected_intent": "change_pin"
    },
    {
        "question": "Ma carte bancaire ne fonctionne plus.",
        "expected_intent": "card_not_working"
    },
    {
        "question": "Ma carte n'est toujours pas arrivée.",
        "expected_intent": "card_arrival"
    },
    {
        "question": "Un virement que j'ai effectué n'est toujours pas reçu.",
        "expected_intent": "transfer_not_received_by_recipient",
        "expected_low_confidence": True
    }
]


print("\n===== TESTS BANKBOT =====")

success = 0

for test in tests:

    result = bankbot(test["question"])

    expected = test["expected_intent"]
    predicted = result["intent"]

    if test.get("expected_low_confidence"):
    
        if result["confidence"] < 0.30:
            status = "OK"
            success += 1
        else:
            status = "ERREUR"
    
    else:
    
        if predicted == expected:
            status = "OK"
            success += 1
        else:
            status = "ERREUR"

    print(
        f"\n[{status}] "
        f"{test['question']}"
    )

    print(
        f"Attendu : {expected}"
    )

    print(
        f"Prédit  : {predicted}"
    )

    print(
        f"Confiance : "
        f"{result['confidence'] * 100:.2f} %"
    )


print("\n========================")
print(
    f"Résultat : {success}/{len(tests)} tests réussis"
)
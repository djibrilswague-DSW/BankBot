from deep_translator import GoogleTranslator


def translate_to_english(text):
    translator = GoogleTranslator(source="fr", target="en")
    return translator.translate(text)


if __name__ == "__main__":
    question = "Je n'ai toujours pas reçu ma carte bancaire."

    translation = translate_to_english(question)

    print("Question :", question)
    print("Traduction :", translation)
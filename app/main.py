import gradio as gr

from agent import bankbot


def chat_response(message, history):
    if not message or not message.strip():
        return history or [], "", ""

    result = bankbot(message)

    history = history or []

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result["answer"]}
    ]

    info = (
        f"**Intention détectée :** `{result['intent']}`  \n"
        f"**Confiance :** {result['confidence'] * 100:.2f} %"
    )

    return history, "", info


with gr.Blocks(title="BankBot") as demo:

    gr.Markdown(
        """
        # 🏦 BankBot

        **Assistant IA conversationnel pour la relation client bancaire**

        Posez une question concernant votre carte, vos paiements,
        vos virements ou votre compte.
        """
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        type="messages"
    )

    message = gr.Textbox(
        label="Votre question",
        placeholder="Exemple : Je ne reconnais pas ce paiement..."
    )

    send = gr.Button("Envoyer")

    analysis = gr.Markdown(
        "L'intention et le niveau de confiance apparaîtront ici."
    )

    send.click(
        chat_response,
        inputs=[message, chatbot],
        outputs=[chatbot, message, analysis]
    )

    message.submit(
        chat_response,
        inputs=[message, chatbot],
        outputs=[chatbot, message, analysis]
    )


demo.launch(server_name="0.0.0.0", server_port=7860)
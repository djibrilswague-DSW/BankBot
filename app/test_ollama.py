from ollama import chat

response = chat(
    model="qwen2.5:3b",
    messages=[
        {
            "role": "user",
            "content": "Réponds en une phrase : combien de temps faut-il généralement pour recevoir une carte bancaire ?"
        }
    ]
)

print(response.message.content)
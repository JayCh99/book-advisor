from openai import OpenAI


client = OpenAI(api_key="sk-proj-1234567890")

response = client.chat.completions.create(
    model="gpt-4o-mini", messages=[{"role": "user", "content": "Hello, how are you?"}]
)

print(response.choices[0].message.content)

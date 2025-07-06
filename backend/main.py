from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()


def answer_query(client: OpenAI, query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": query}]
    )

    if response.choices[0].message.content is None:
        raise ValueError("No response from OpenAI")

    return response.choices[0].message.content


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print(answer_query(client, "Hello, how are you?"))

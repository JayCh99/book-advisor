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
    queries = [
        "I feel like I've always struggled with feeling good about myself and it's really affecting my life right now, what should I do?",
        "I want to buy my sister a nice gift but I'm not sure what she'd like, what should I get her?",
        "I don't think I've taken care of my health enough and it's negatively impacting my life, but my stress keeps me from making it a priority day to day. How should I handle this?",
        "I have frequent headaches, help?",
        "I really love love. There's this girl I really like, how could I spend more time with her?",
        "How do I think more about the things that excite me than the things that scare me?",
    ]
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print(answer_query(client, "Hello, how are you?"))

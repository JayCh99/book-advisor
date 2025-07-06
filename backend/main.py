from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
import os
from dotenv import load_dotenv
import base64

load_dotenv()


# Pricing table for OpenAI models (per 1M tokens)
MODEL_PRICING = {
    # GPT-4.1 models
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
}


def get_cost_from_response(response: ChatCompletion, model: str) -> float:
    if not response.usage:
        raise ValueError("No usage data in response")

    return (
        response.usage.prompt_tokens * MODEL_PRICING[response.model]["input"]
        + response.usage.completion_tokens * MODEL_PRICING[response.model]["output"]
    ) / 1000000


def answer_query(client: OpenAI, query: str) -> ChatCompletion:
    with open("book.pdf", "rb") as file:
        pdf_base64 = base64.b64encode(file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": f"Use only the advice in this book, which I'm providing as a base 64 encoded pdf, to answer my query: {pdf_base64}.",
            },
            {"role": "user", "content": query},
        ],
    )

    if response.choices[0].message.content is None:
        raise ValueError("No response from OpenAI")

    return response


if __name__ == "__main__":
    queries = [
        "I feel like I've always struggled with feeling good about myself and it's really affecting my life right now, what should I do?",
        "I want to buy my sister a nice gift but I'm not sure what she'd like, what should I get her?",
        "I don't think I've taken care of my health enough and it's negatively impacting my life, but my stress keeps me from making it a priority day to day. How should I handle this?",
        "I have frequent headaches, help?",
        "I really love love. There's this girl I really like, how could I spend more time with her?",
        "How do I think more about the things that excite me than the things that scare me?",
    ]

    with open("book.pdf", "rb") as file:
        pdf_base64 = base64.b64encode(file.read()).decode("utf-8")

    print(pdf_base64[: len(pdf_base64) // 100])

    # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # response = answer_query(client, queries[0])
    # print(get_cost_from_response(response, "gpt-4o-mini"))
    # print(response.choices[0].message.content)

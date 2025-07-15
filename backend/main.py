from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
import os
from dotenv import load_dotenv
import base64
import tiktoken
import PyPDF2


load_dotenv()


def count_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_cost_from_response(response: ChatCompletion, model: str) -> float:
    # Pricing table for OpenAI models (per 1M tokens)
    MODEL_PRICING = {
        # GPT-4.1 models
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
    }

    if not response.usage:
        raise ValueError("No usage data in response")

    return (
        response.usage.prompt_tokens * MODEL_PRICING[response.model]["input"]
        + response.usage.completion_tokens * MODEL_PRICING[response.model]["output"]
    ) / 1000000


def answer_query(client: OpenAI, query: str, book_text: str) -> ChatCompletion:
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": f"Use only the advice in this book, which I'm providing as a base 64 encoded pdf, to answer my query: {book_text}. Supply an answer, then specific quotes from the book that support your answer. Output your answer in markdown format.",
            },
            {"role": "user", "content": query},
        ],
    )

    if response.choices[0].message.content is None:
        raise ValueError("No response from OpenAI")

    return response


if __name__ == "__main__":
    # Elon Musk book: https://www.dirzon.com/file/telegram/eltstudentfiles/Elon_Musk_2023.pdf
    # AOL Tokens: 1,459,692 (Base64), 100,000 (Text)
    # Elon Musk Tokens: 47,693,743 (Base64), 411,047 (Text)

    # Alleged Gemini Tokenizer (gives 1/4 the tokens of tiktoken)

    SOURCE_BOOK = "inputs/aol_book.pdf"
    BOOK_TEXT_FILE = "outputs/aol_book_text.txt"
    RESPONSE_FILE = "outputs/response.md"

    queries = [
        "I feel like I've always struggled with feeling good about myself and it's really affecting my life right now, what should I do?",
        "I want to buy my sister a nice gift but I'm not sure what she'd like, what should I get her?",
        "I don't think I've taken care of my health enough and it's negatively impacting my life, but my stress keeps me from making it a priority day to day. How should I handle this?",
        "I have frequent headaches, help?",
        "I really love love. There's this girl I really like, how could I spend more time with her?",
        "How do I think more about the things that excite me than the things that scare me?",
        "I just feel kind of empty. I love when I can enjoy that feeling and just be here right now",
    ]

    query = queries[len(queries) - 1]

    with open(SOURCE_BOOK, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        book_text = ""
        for page in reader.pages:
            book_text += page.extract_text()

    with open(BOOK_TEXT_FILE, "w", encoding="utf-8") as f:
        f.write(book_text)
    print(f"Book text dumped to {BOOK_TEXT_FILE}")

    print(f"Book text tokens: {count_tokens(book_text, 'gpt-4o'):,}")

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = answer_query(client, query, book_text)
    # Save response as markdown file
    response_text = response.choices[0].message.content or ""
    with open("response.md", "w", encoding="utf-8") as f:
        f.write(response_text)
    print(f"Response saved to response.md. Text: {response_text}")
    print(f"Cost: {get_cost_from_response(response, 'gpt-4.1-nano')}")

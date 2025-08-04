from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

llm_readme = init_chat_model(
    model="qwen/qwen3-32b",
    model_provider="groq",
    temperature=0.1,
    reasoning_format="parsed"
)

llm_summary = init_chat_model(
    model="codestral-2405",
    model_provider="mistralai",
    temperature=0.3,
)

llm_commenting = init_chat_model(
    model="codestral-2501",
    model_provider="mistralai",
    temperature=0.3
)

parser = StrOutputParser()

def get_llm_response_summary(prompt: str, language: str) -> str:
    messages = [
        ("system", 
        f"You are a highly skilled senior {language} software engineer. "
        f"Always write precise, technical, and concise output without adding explanations or extra commentary."),
        ("user", prompt)
    ]
    chain = llm_summary | StrOutputParser()
    return chain.invoke(messages)

def get_llm_response_readme(prompt: str) -> str:
    messages = [
        ("system", 
        "You are a technical writer and documentation specialist. "
        "You create clean, professional, and well-structured Markdown documentation. "
        "Always be concise, precise, and avoid adding any extra commentary or text."),
        ("user", prompt)
    ]
    chain = llm_readme | StrOutputParser()
    return chain.invoke(messages)

def get_llm_response_commenting(prompt: str) -> str:
    messages = [("user", prompt)]
    chain = llm_commenting | StrOutputParser()
    return chain.invoke(messages)

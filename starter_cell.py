from dotenv import load_dotenv
from typing import Literal

# Load environment variables from .env
load_dotenv("../../.env")


from langchain_community.chat_models import ChatOllama
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from langchain_community.embeddings.ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model='nomic-embed-text')

def get_chat_model(type: Literal["gemma", "llama", "openai", "internlm2"] = "llama") -> BaseChatModel:
    if type == "gemma":
        return ChatOllama(model="gemma2")
    elif type == "llama":
        return ChatOllama(model="llama3")
    elif type == "openai":
        return ChatOpenAI()
    elif type == "internlm2":
        return ChatOllama(model="internlm2")


# llm = get_chat_model("llama")

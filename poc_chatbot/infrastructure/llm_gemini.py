from enum import Enum
from pydantic import SecretStr
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

class GeminiEmbeddingTask(str, Enum):
    TASK_TYPE_UNSPECIFIED = "task_type_unspecified"
    RETRIEVAL_QUERY = "retrieval_query"
    RETRIEVAL_DOCUMENT = "retrieval_document"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


def get_gemini_embedding_model(
    api_key: SecretStr, task_type: GeminiEmbeddingTask
) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key.get_secret_value(),
        task_type=task_type.value,
    )
    
    
def get_gemini_llm(api_key: SecretStr) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_output_tokens=1024,
        google_api_key=api_key.get_secret_value(),
    )


# if __name__ == "__main__":
#     import os

#     from app_config import app_config

#     gemini_api_key = app_config.gemini.api_key
#     os.environ["GOOGLE_API_KEY"] = gemini_api_key.get_secret_value()

#     embedding_model = get_gemini_embedding_model(
#         api_key=gemini_api_key, task_type=GeminiEmbeddingTask.RETRIEVAL_DOCUMENT
#     )
#     texts = ["Hello, world!", "Bonjour le monde!"]
#     embeddings = embedding_model.embed_documents(texts)
#     for text, embedding in zip(texts, embeddings):
#         print(f"Text: {text}\nEmbedding: {embedding}\n")
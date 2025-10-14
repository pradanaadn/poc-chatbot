from poc_chatbot.backend.document_processing_pipeline import DocumentProcessingPipeline
from poc_chatbot.backend.embedding_vector_store import EmbeddingVectorStore
from poc_chatbot.infrastructure.llm_gemini import (
    GeminiEmbeddingTask,
    get_gemini_embedding_model,
)
from app_config import app_config
from poc_chatbot.infrastructure.qdrant_client_instance import QdrantClientInstance
from poc_chatbot.backend.clean_text_pipeline import CleanTextPipeline
from yake import KeywordExtractor


def main():
    # Initialize Qdrant Client
    qdrant_config = app_config.qdrant
    qdrant_client = QdrantClientInstance.init(
        host=qdrant_config.host,
        port=qdrant_config.http_port,
        api_key=qdrant_config.api_key.get_secret_value(),
    )

    # Initialize Gemini Embedding Model
    gemini_api_key = app_config.gemini.api_key
    embedding_model = get_gemini_embedding_model(
        api_key=gemini_api_key, task_type=GeminiEmbeddingTask.RETRIEVAL_DOCUMENT
    )

    # Initialize Embedding Vector Store
    vector_store = EmbeddingVectorStore(
        qdrant_client=qdrant_client, embedding_model=embedding_model
    )
    text_cleaner = CleanTextPipeline()
    keyword_extractor = KeywordExtractor(lan="en", n=3, dedupLim=0.95, dedupFunc="jaro", top=3)
    # Process Document and Add to Vector Store
    document_path = "assets/example_documents.pdf"
    document_pipeline = DocumentProcessingPipeline(source=document_path, text_cleaner=text_cleaner, keyword_extractor=keyword_extractor)
    document_pipeline.load(exclude_pages=list(range(0, 10)))
    chunks = document_pipeline.chunk()
    print(f"Total Chunks Processed: {len(chunks)}")
    print(f"Sample Chunk: {chunks[0].model_dump()}")
    vector_store.add_documents(documents=chunks, collection_name="AWS-White-Paper")

    print(f"Successfully added {len(chunks)} document chunks to the vector store.")


def query_example():
    qdrant_config = app_config.qdrant
    qdrant_client = QdrantClientInstance.init(
        host=qdrant_config.host,
        port=qdrant_config.http_port,
        api_key=qdrant_config.api_key.get_secret_value(),
    )

    gemini_api_key = app_config.gemini.api_key
    llm = get_gemini_embedding_model(
        api_key=gemini_api_key, task_type=GeminiEmbeddingTask.RETRIEVAL_QUERY
    )

    query_text = "Explain about amazon sagemaker!"
    query_embedding = llm.embed_query(text=query_text)

    results = qdrant_client.search(
        collection_name="AWS-White-Paper", query_vector=query_embedding, limit=5
    )

    print("Query Results:")
    for result in results:
        print(result)


if __name__ == "__main__":
    # main()
    query_example()

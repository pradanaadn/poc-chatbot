from poc_chatbot.backend.document_processing_pipeline import DocumentProcessingPipeline
from poc_chatbot.backend.embedding_vector_store import EmbeddingVectorStore
from poc_chatbot.infrastructure.llm_gemini import (
    GeminiEmbeddingTask,
    get_gemini_embedding_model,
    get_gemini_llm,
)
from app_config import app_config
from poc_chatbot.infrastructure.qdrant_client_instance import QdrantClientInstance


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
    vector_store = EmbeddingVectorStore(qdrant_client=qdrant_client, embedding_model=embedding_model)

    # Process Document and Add to Vector Store
    document_path = "assets/example_documents.pdf"
    document_pipeline = DocumentProcessingPipeline(source=document_path)
    document_pipeline.load()
    chunks = document_pipeline.chunk()
    print(f"Total Chunks Processed: {len(chunks)}")
    print(f"Sample Chunk: {chunks[0].model_dump()}")
    vector_store.add_documents(documents=chunks, collection_name="Testing-Documents")

    print(f"Successfully added {len(chunks)} document chunks to the vector store.")

if __name__ == "__main__":
    main()
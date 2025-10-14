from yake import KeywordExtractor
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from app_config import app_config
from poc_chatbot.backend.clean_text_pipeline import CleanTextPipeline
from poc_chatbot.backend.document_processing_pipeline import DocumentProcessingPipeline
from poc_chatbot.backend.embedding_vector_store import HybridEmbeddingVectorStore
from poc_chatbot.infrastructure.llm_gemini import (
    GeminiEmbeddingTask,
    get_gemini_embedding_model,
    get_gemini_llm
)
from poc_chatbot.infrastructure.qdrant_client_instance import QdrantClientInstance


def add_documents():
    qdrant_config = app_config.qdrant
    qdrant_client = QdrantClientInstance.init(
        host=qdrant_config.host,
        port=qdrant_config.http_port,
        api_key=qdrant_config.api_key.get_secret_value(),
    )
    gemini_api_key = app_config.gemini.api_key
    embedding_model = get_gemini_embedding_model(
        api_key=gemini_api_key, task_type=GeminiEmbeddingTask.RETRIEVAL_DOCUMENT
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    vector_store = HybridEmbeddingVectorStore(
        qdrant_client=qdrant_client
    )
    vector_store.create_collection(
        collection_name="hybrid_documents")
    
    qdrant_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="hybrid_documents",
        embedding=embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )

   
    text_cleaner = CleanTextPipeline()
    keyword_extractor = KeywordExtractor(
        lan="en", n=3, dedupLim=0.95, dedupFunc="jaro", top=3
    )

    document_path = "assets/example_documents.pdf"
    document_pipeline = DocumentProcessingPipeline(
        source=document_path,
        text_cleaner=text_cleaner,
        keyword_extractor=keyword_extractor,
    )
    document_pipeline.load(exclude_pages=list(range(0, 10)))
    chunks = document_pipeline.chunk()
    print(f"Total Chunks Processed: {len(chunks)}")
    print(f"Sample Chunk: {chunks[0].model_dump()}")
    vector_store.add_documents(documents=chunks, vector_store=qdrant_vector_store)

    print(f"Successfully added {len(chunks)} document chunks to the vector store.")
    
def query():
    qdrant_config = app_config.qdrant
    qdrant_client = QdrantClientInstance.init(
        host=qdrant_config.host,
        port=qdrant_config.http_port,
        api_key=qdrant_config.api_key.get_secret_value(),
    )
    gemini_api_key = app_config.gemini.api_key
    embedding_model = get_gemini_embedding_model(
        api_key=gemini_api_key, task_type=GeminiEmbeddingTask.RETRIEVAL_DOCUMENT
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    llm = get_gemini_llm(
        api_key=gemini_api_key
    )
    vector_store = HybridEmbeddingVectorStore(
        qdrant_client=qdrant_client
    )
    vector_store.create_collection(
        collection_name="hybrid_documents")
    
    qdrant_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="hybrid_documents",
        embedding=embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    query_text = "I want to deploy a ultralystics yolo model. How can I do that?"
    found_docs = qdrant_vector_store.similarity_search(
        query_text, k=5
    )
    llm_response = llm.invoke(
        input=f"Answer the question based on the context below:\n\nContext: {found_docs}\n\nQuestion: {query_text}\n\nAnswer:",
    )
    llm_response.pretty_print()
    
if __name__ == "__main__":
    # add_documents()
    query()

from yake import KeywordExtractor
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from app_config import app_config
from poc_chatbot.backend.clean_text_pipeline import CleanTextPipeline
from poc_chatbot.backend.document_processing_pipeline import DocumentProcessingPipeline
from poc_chatbot.backend.embedding_vector_store import HybridEmbeddingVectorStore
from poc_chatbot.infrastructure.llm_gemini import (
    GeminiEmbeddingTask,
    get_gemini_embedding_model,
    get_gemini_llm,
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
    vector_store = HybridEmbeddingVectorStore(qdrant_client=qdrant_client)
    vector_store.create_collection(collection_name="hybrid_documents")

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

    document_path = "assets/IBM_TAM_ESSO_TroubleshootingGuide_pdf.pdf"
    document_pipeline = DocumentProcessingPipeline(
        source=document_path,
        text_cleaner=text_cleaner,
        keyword_extractor=keyword_extractor,
    )
    document_pipeline.load(exclude_pages=list(range(0, 7)))
    chunks = document_pipeline.chunk()
    print(f"Total Chunks Processed: {len(chunks)}")
    print(f"Sample Chunk: {chunks[0].model_dump()}")
    vector_store.add_documents(documents=chunks, vector_store=qdrant_vector_store)

    print(f"Successfully added {len(chunks)} document chunks to the vector store.")


# def query():
#     qdrant_config = app_config.qdrant
#     qdrant_client = QdrantClientInstance.init(
#         host=qdrant_config.host,
#         port=qdrant_config.http_port,
#         api_key=qdrant_config.api_key.get_secret_value(),
#     )
#     gemini_api_key = app_config.gemini.api_key
#     embedding_model = get_gemini_embedding_model(
#         api_key=gemini_api_key, task_type=GeminiEmbeddingTask.RETRIEVAL_DOCUMENT
#     )
#     sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
#     llm = get_gemini_llm(api_key=gemini_api_key)
#     vector_store = HybridEmbeddingVectorStore(qdrant_client=qdrant_client)
#     vector_store.create_collection(collection_name="hybrid_documents")

#     qdrant_vector_store = QdrantVectorStore(
#         client=qdrant_client,
#         collection_name="hybrid_documents",
#         embedding=embedding_model,
#         sparse_embedding=sparse_embeddings,
#         retrieval_mode=RetrievalMode.HYBRID,
#         vector_name="dense",
#         sparse_vector_name="sparse",
#     )
#     system_prompt = """
#     You are an AI customer service assistant built by Google. Your primary goal is to provide helpful, clear, and professional support.

# When formulating your response, you must structure the answer internally using the PREP (Point, Reason, Example/Evidence, Point Restated) or PEEL (Point, Explanation, Evidence, Link/Next Step) framework to ensure maximum clarity and easy understanding for the customer.
# Do not answer if the context is not relevant to the question. Answer with customer service way ( I can't find relevant information about ... in my knowledge base) if you are unsure of the answer. If the question is not clear, ask for clarification.
# Do not mention the frameworks (PREP/PEEL) in your response.


# CRITICAL INSTRUCTION: DO NOT INCLUDE ANY EXPLICIT HEADINGS OR LABELS such as 'Point,' 'Explanation,' 'Evidence,' or 'Link/Next Step' in the final output. The answer must read as a seamless, naturally flowing response.

# Key Requirements:

# Tone: Maintain a helpful, empathetic, and professional customer service voice.

# Structure: The response must be logically structured according to PREP/PEEL (Direct Answer → Supporting Details → Summary/Next Step), but the structure must be implicit.

# Clarity: Use clear, straightforward language, avoiding unnecessary jargon.

# Actionability: Ensure solutions or explanations are easy for the customer to understand and act upon.
#     """
#     list_chat_history = []
#     while True:
#         user_input = input("User: ")
#         if user_input.lower() in ["exit", "quit"]:
#             print("Exiting chat.")
#             break
#         list_chat_history.append({"role": "user", "content": user_input})
#         found_docs = qdrant_vector_store.similarity_search(user_input, k=5)
#         llm_response = llm.invoke(
#             [
#                 {"role": "system", "content": system_prompt},
#                 *list_chat_history,
#                 {"role": "assistant", "content": f"Context: {found_docs}"},
#             ]
#         )
#         llm_response.pretty_print()
#         list_chat_history.append(
#             {"role": "assistant", "content": llm_response.content}
#         )
   



if __name__ == "__main__":
    add_documents()
    # query()

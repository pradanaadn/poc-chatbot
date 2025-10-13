from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from poc_chatbot.backend.schema import DocumentChunk

class EmbeddingVectorStore:
    def __init__(self, qdrant_client: QdrantClient, embedding_model: GoogleGenerativeAIEmbeddings):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model

    def add_documents(self, documents: list[DocumentChunk], collection_name: str = "documents", distance: Distance = Distance.COSINE):
        texts = [doc.text for doc in documents]
        titles = [doc.metadata.title if doc.metadata and doc.metadata.title else "" for doc in documents]
        if not self.qdrant_client.collection_exists(collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(self.embedding_model.embed_query("test")), distance=distance),
            )
        embeddings = self.embedding_model.embed_documents(texts, batch_size=32, titles=titles)

        points = []
        for doc, embedding in zip(documents, embeddings):
            point = {
                "id": doc.chunk_id,
                "vector": embedding,
                "payload": {
                    "text": doc.text,
                    "metadata": doc.metadata.model_dump() if doc.metadata else {},
                },
            }
            points.append(point)
        
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
    
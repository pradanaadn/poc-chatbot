from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from poc_chatbot.backend.schema import DocumentChunk
from poc_chatbot.backend.utils import create_batches


class EmbeddingVectorStore:
    def __init__(
        self, qdrant_client: QdrantClient, embedding_model: GoogleGenerativeAIEmbeddings
    ):
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model

    def add_documents(
        self,
        documents: list[DocumentChunk],
        collection_name: str = "documents",
        distance: Distance = Distance.COSINE,
    ):
        """Convert documents to embedding, and save to vectore db.

        Args:
            documents (list[DocumentChunk]): List of DocumentChunk to be embedded and stored.
            collection_name (str, optional): Name of the collection to store the embeddings. Defaults to "documents".
            distance (Distance, optional): Distance metric for the vector search. Defaults to Distance.COSINE.
        """
        texts = [doc.text for doc in documents]
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        if not self.qdrant_client.collection_exists(collection_name):
            logger.info(f"Creating collection '{collection_name}' in Qdrant...")
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=len(self.embedding_model.embed_query("test")),
                    distance=distance,
                ),
            )
        logger.info(f"Collection '{collection_name}' already exists in Qdrant.")
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.embed_documents(texts, batch_size=64)
        logger.info("Embeddings generated.")
        points = []
        logger.info(f"Preparing {len(documents)} points for upsert...")
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
        logger.info(
            f"Upserting {len(points)} points into collection '{collection_name}'..."
        )
        for batch in create_batches(points, 64):
            logger.info(f"Upserting batch of {len(batch)} points...")
            self.qdrant_client.upsert(collection_name=collection_name, points=batch)
            logger.info(f"Upserted batch of {len(batch)} points.")
        logger.success(
            f"Successfully upserted {len(points)} points into collection '{collection_name}'."
        )

from loguru import logger
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams
from poc_chatbot.backend.schema import DocumentChunk
from poc_chatbot.backend.utils import create_batches, document_adapter


class HybridEmbeddingVectorStore:
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client

    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 3072,
        distance: Distance = Distance.COSINE,
    ):
        """Create a collection in Qdrant if it does not exist.

        Args:
            collection_name (str): Name of the collection to create.
            vector_size (int): Size of the vectors to be stored in the collection.
            distance (Distance, optional): Distance metric for the vector search. Defaults to Distance.COSINE.
        """
        if not self.qdrant_client.collection_exists(collection_name):
            logger.info(f"Creating collection '{collection_name}' in Qdrant...")
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=vector_size,
                        distance=distance,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            logger.info(f"Collection '{collection_name}' created.")
        else:
            logger.info(f"Collection '{collection_name}' already exists.")

    def add_documents(
        self,
        documents: list[DocumentChunk],
        vector_store: QdrantVectorStore,
        vector_size: int = 3072,
        distance: Distance = Distance.COSINE,
        batch=64,
    ):
        """Convert documents to embedding, and save to vectore db.

        Args:
            documents (list[DocumentChunk]): List of DocumentChunk to be embedded and stored.
            collection_name (str, optional): Name of the collection to store the embeddings. Defaults to "documents".
            distance (Distance, optional): Distance metric for the vector search. Defaults to Distance.COSINE.
        """
      

        documents, ids = document_adapter(documents)
        batches_documents, batches_ids = (
            create_batches(documents, batch),
            create_batches(ids, batch),
        )
        for docs_batch, ids_batch in zip(batches_documents, batches_ids):
            vector_store.add_documents(
                documents=docs_batch,
                ids=ids_batch,
            )

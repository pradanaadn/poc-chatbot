from langchain_qdrant import  QdrantVectorStore

class EmbeddingVectorQuery:
    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store
        

    def query(self, query_vector: list[float], top_k: int = 5, collection_name: str = "documents"):
        results = self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return results
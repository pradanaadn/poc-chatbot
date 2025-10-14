from qdrant_client import QdrantClient



class QdrantClientInstance:
    __client: QdrantClient = None

    @classmethod
    def init(cls, host: str, port: int, api_key: str):
        """Initialize Qdrant Client

        Args:
            host (str): Host of the qdrant
            port (int): Http port of the qdrant
            api_key (str): API key for authentication

        Returns:
            QdrantClient: Instance of the QdrantClient
        """
        if cls.__client is None:
            cls.__client = QdrantClient(host=host, port=port, api_key=api_key, https=False, timeout=120)
        return cls.__client

    @classmethod
    def get_client(cls):
        """Get the Qdrant Client instance
        Returns:
            QdrantClient: Instance of the QdrantClient
        """
        if cls.__client is None:
            raise ValueError("QdrantClient is not initialized. Call 'init' first.")
        return cls.__client


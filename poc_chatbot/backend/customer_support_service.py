from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from pydantic import TypeAdapter
from app_config import AppConfig, app_config
from poc_chatbot.backend.embedding_vector_store import HybridEmbeddingVectorStore
from poc_chatbot.infrastructure.llm_gemini import (
    GeminiEmbeddingTask,
    get_gemini_embedding_model,
    get_gemini_llm,
)
from poc_chatbot.backend.customer_support_workflow import (
    PromptCustomerSupport,
    CustomerSupportWorkflowState,
    CustomerSupportWorkflow,
)
from poc_chatbot.infrastructure.qdrant_client_instance import QdrantClientInstance


class CustomerSupportService:
    def __init__(self, config: AppConfig = app_config, prompt=PromptCustomerSupport()):
        self.config = config
        self.llm = get_gemini_llm(config.gemini.api_key)
        self.embedding_model = get_gemini_embedding_model(
            config.gemini.api_key, GeminiEmbeddingTask.RETRIEVAL_DOCUMENT
        )
        self.sparse_embedding = FastEmbedSparse(model_name="Qdrant/bm25")
        self.qdrant_client = QdrantClientInstance.init(
            host=config.qdrant.host,
            port=config.qdrant.http_port,
            api_key=config.qdrant.api_key.get_secret_value(),
        )
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="hybrid_documents",
            embedding=self.embedding_model,
            sparse_embedding=self.sparse_embedding,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        self.hybrid_vector_store = HybridEmbeddingVectorStore(self.qdrant_client)

        self.hybrid_vector_store.create_collection(collection_name="hybrid_documents")

        self.workflow = CustomerSupportWorkflow(
            llm_model=self.llm, vector_store=self.vector_store, prompt=prompt
        )
        self.graph = self.workflow.get_compiled_graph()

    def run(self, user_query: str) -> CustomerSupportWorkflowState:
        state = CustomerSupportWorkflowState(user_query=user_query)
        result = self.graph.invoke(input=state)
        state_adapter = TypeAdapter(CustomerSupportWorkflowState).validate_python(
            result
        )
        return state_adapter


if __name__ == "__main__":
    prompt = PromptCustomerSupport().load_prompt()
    service = CustomerSupportService(prompt=prompt)
    response = service.run(
        user_query="I found bug on the Wallet using Active RFID card. It keep raise error 'Card not detected'. I want to request for new card"
    )
    print(response.model_dump_json(indent=2))

import uuid
from datetime import datetime
from typing import Literal

from langchain.tools.retriever import create_retriever_tool
from langchain_core.language_models.base import BaseLanguageModel
from langchain_qdrant import QdrantVectorStore
from langgraph.graph.state import CompiledStateGraph

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, AIMessage
from pydantic import BaseModel, Field


class UserTicket(BaseModel):
    ticket_id: str = Field(
        default=str(uuid.uuid4()),
        description="Unique identifier for the ticket. Not passed by LLM.",
    )
    user_id: str = Field(
        default="not set", description="ID of the user, not passed by LLM"
    )
    title: str = Field(
        default="No Title",
        max_length=200,
        description="Title of the support ticket. Set by LLM. Max 200 characters.",
    )
    issue_description: str = Field(
        default="", description="Detailed description of the issue. Set by LLM."
    )
    labels: list[str] = Field(
        default=[], description="List of labels/tags for the ticket. Set by LLM."
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        description="Priority level of the ticket. Set by LLM."
    )
    create_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of ticket creation. Not passed by LLM.",
    )
    status: Literal["open", "closed"] = Field(
        default="open", description="Status of the ticket, not passed by LLM"
    )


class GradeDocument(BaseModel):
    relevantness: Literal[
        "relevant", "not_relevant", "query_not_match_with_document", "create_ticket"
    ] = Field(description="Whether the retrieved documents are relevant")

    def is_relevant(self) -> bool:
        return self.relevantness == "relevant"

    def is_not_relevant(self) -> bool:
        return self.relevantness == "not_relevant"

    def is_query_not_match_with_document(self) -> bool:
        return self.relevantness == "query_not_match_with_document"

    def is_create_ticket(self) -> bool:
        return self.relevantness == "create_ticket"


class PromptCustomerSupport(BaseModel):
    generate_query_or_response: str = Field(
        "You are a customer service AI. Use document_retriever to search the knowledge base when users ask questions."
        " Do NOT ask follow-up questions or engage in conversation—just retrieve relevant documents.",
        description="Prompt for node generate_query_or_response",
    )
    grade_document_relevance: str = Field(
        "Evaluate if context can answer the query from the document's perspective. Return ONE value:\n\n"
        "1. 'create_ticket' - Query needs ACTION or user-specific investigation:\n"
        "   - Errors/bugs, account actions, user-specific issues, feature requests\n\n"
        "2. 'relevant' - Context can answer query from its SPECIFIC PERSPECTIVE:\n"
        "   - General queries (e.g., 'best practices') can be answered with specific examples from context (e.g., AWS best practices)\n"
        "   - Context provides relevant information even if not comprehensive coverage\n"
        "   - Partial but useful answers are still relevant\n\n"
        "3. 'query_not_match_with_document' - Query and context are completely unrelated products/domains\n"
        "   (e.g., Google Cloud query + AWS context about different topics)\n\n"
        "4. 'not_relevant' - Same product BUT context doesn't address the specific aspect asked\n"
        "   (e.g., deployment query + only pricing context)\n\n"
        "Decision order: Check #1 first, then #2, then #3, then #4.",
        description="Prompt for node grade_document_relevance",
    )
    rewrite_query: str = Field(
        "Rewrite query to improve retrieval. Initial search found same product docs but didn't answer the specific question.\n\n"
        "Rules:\n"
        "- Preserve CORE TOPIC from original query\n"
        "- Make more specific while keeping same subject\n"
        "- Add clarifying terms (deployment, security, performance)\n"
        "- Use context terminology ONLY if relevant to user's intent\n"
        "- If context is off-topic, ignore it and focus on original query\n\n"
        "Examples:\n"
        "- 'best practices' → 'best practices for deployment and security'\n"
        "- 'how to deploy' → 'step-by-step deployment guide'\n\n"
        "Return only rewritten query.",
        description="Prompt for node rewrite_query",
    )
    create_ticket: str = Field(
        "Create support ticket from conversation. Include:\n"
        "- Title (max 200 chars)\n"
        "- Detailed issue description\n"
        "- Labels/tags\n"
        "- Priority: low/medium/high/urgent\n\n"
        "Capture essence of user's problem. Make reasonable assumptions if unclear (state assumptions). Exclude PII.",
        description="Prompt for node create_ticket",
    )
    generate_answer: str = Field(
        "Answer the user's question using the provided context. Extract ALL relevant details and present them directly.\n\n"
        "Rules:\n"
        "- If context contains ANY information related to the question, use it to provide a helpful answer\n"
        "- Present actual features, capabilities, and details - not meta-descriptions\n"
        "- For general questions, answer from the specific service/product perspective in context\n"
        "- ONLY say 'no information found' if context is completely unrelated or empty\n\n"
        "Examples:\n"
        "❌ BAD: 'I can't find specific information on how X improves Y. The context mentions...'\n"
        "✅ GOOD: 'X improves team collaboration through: 1. Feature A allows..., 2. Feature B enables...'\n\n"
        "Do not fabricate information not in context.",
        description="Prompt for node generate_answer",
    )
    generate_sorry_response: str = Field(
        "Inform user you couldn't find relevant information. Mention support ticket created and human agent will contact them."
        " Keep brief and professional. Do NOT ask follow-up questions.",
        description="Prompt for node generate_sorry_response",
    )

class CustomerSupportWorkflowState(BaseModel):
    user_query: str = Field(description="User's input query")
    rewritten_query: str | None = Field(None, description="Rewritten user query")
    generated_response: str | None = Field(
        None, description="Response generated for the query"
    )
    already_rewritten: bool = Field(
        default=False, description="Whether the query has been rewritten"
    )
    histories: list = Field(
        default_factory=list, description="History of user interactions"
    )
    ticket: UserTicket | None = Field(
        default=None, description="Generated support ticket if applicable"
    )
    context: str | None = Field(None, description="Context from retrieved documents")

    def set_context(self, value: str | None):
        self.context = value

    def set_already_rewritten(self, value: bool):
        self.already_rewritten = value

    def set_rewritten_query(self, value: str | None):
        self.rewritten_query = value

    def set_generated_response(self, value: str | None):
        self.generated_response = value

    def add_to_history(self, entry: BaseMessage | list[BaseMessage]):
        if isinstance(entry, list):
            self.histories.extend(entry)
        else:
            self.histories.append(entry)

    def set_ticket(self, ticket: UserTicket):
        self.ticket = ticket


class CustomerSupportWorkflow:
    def __init__(
        self,
        llm_model: BaseLanguageModel,
        vector_store: QdrantVectorStore,
        prompt: PromptCustomerSupport,
    ):
        self.llm_model = llm_model
        self.vector_store = vector_store
        self.prompt = prompt
        self.__graph = self.build()

    def build(self) -> CompiledStateGraph:
        """Build the workflow graph
        Returns:
            CompiledStateGraph: compiled state graph
        """
        workflow = StateGraph(CustomerSupportWorkflowState)
        workflow.add_node("generate_query_or_response", self.generate_query_or_response)
        workflow.add_node("grade_document_relevance", self.grade_document_relevance)
        retrieve_node = ToolNode([self.retrieve_tool()], messages_key="histories")
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("create_ticket", self.create_ticket)
        workflow.add_node("generate_answer", self.generate_answer)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("is_use_tool", self.is_use_tool)
        workflow.add_node("generate_sorry_response", self.generate_sorry_response)
        workflow.add_node("store_context", self.store_context)
        workflow.add_edge(START, "generate_query_or_response")
        workflow.add_conditional_edges(
            "generate_query_or_response",
            self.is_use_tool,
            {"tools": "retrieve", END: END},
        )
        workflow.add_edge("retrieve", "store_context")

        workflow.add_conditional_edges(
            "store_context",
            self.grade_document_relevance,
            {
                "generate_answer": "generate_answer",
                "create_ticket": "create_ticket",
                "rewrite_query": "rewrite_query",
                "generate_sorry_response": "generate_sorry_response",
            },
        )
        workflow.add_edge(
            "rewrite_query",
            "generate_query_or_response",
        )
        workflow.add_edge("generate_sorry_response", END)
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("create_ticket", END)

        graph = workflow.compile()
        return graph

    def is_use_tool(self, state: CustomerSupportWorkflowState):
        return tools_condition(state, messages_key="histories")

    def generate_query_or_response(
        self, state: CustomerSupportWorkflowState
    ) -> CustomerSupportWorkflowState:
        """Generate answer directly or tool calling

        Args:
            state (CustomerSupportWorkflowState): Workflow state

        Returns:
            CustomerSupportWorkflowState: Updated workflow state
        """
        if state.already_rewritten:
            question = state.rewritten_query
        else:
            question = state.user_query

        query = [
            {"role": "system", "content": self.prompt.generate_query_or_response},
            {"role": "user", "content": question},
        ]
        response = self.llm_model.bind_tools([self.retrieve_tool()]).invoke(query)
        if isinstance(response, list):
            for r in response:
                if isinstance(r, AIMessage):
                    state.set_generated_response(r.content)
                state.add_to_history(r)
        else:
            if isinstance(response, AIMessage):
                state.set_generated_response(response.content)
            state.add_to_history(response)

        return state

    def retrieve_tool(self):
        """Create a retriever tool for document retrieval."""
        return create_retriever_tool(
            name="document_retriever",
            description=(
                "Useful for retrieving relevant documents to answer user queries."
                "The parameter:"
                " - query: the input query to search for relevant documents."
            ),
            retriever=self.vector_store.as_retriever(
                search_type="mmr", search_kwargs={"k": 5, "fetch_k": 50}
            ),
        )

    def store_context(
        self, state: CustomerSupportWorkflowState
    ) -> CustomerSupportWorkflowState:
        """Store the retrieved context in state

        Args:
            state (CustomerSupportWorkflowState): state of the workflow

        Returns:
            CustomerSupportWorkflowState: updated state with context
        """
        if state.histories:
            last_message = state.histories[-1]
            # Extract the content from the tool message
            if hasattr(last_message, "content"):
                context = last_message.content
            else:
                context = str(last_message)
            state.set_context(context)
        return state

    def generate_sorry_response(self, state: CustomerSupportWorkflowState):
        """Generate a sorry response when no relevant documents are found

        Returns:
            str: sorry response
        """
        response = self.llm_model.invoke(
            [
                {"role": "system", "content": self.prompt.generate_sorry_response},
                {
                    "role": "user",
                    "content": f"The user query: {state.user_query} context: {state.context}",
                },
            ]
        )
        if response and hasattr(response, "content"):
            state.set_generated_response(response.content)
        else:
            state.set_generated_response(
                "I'm sorry, but I couldn't find relevant information in my knowledge base to answer your question."
            )
        return state

    def grade_document_relevance(self, state: CustomerSupportWorkflowState):
        """Grade the relevance of retrieved documents

        Args:
            state (CustomerSupportWorkflowState): state of the workflow
        Returns:
            str: next node to execute
        """
        if state.already_rewritten:
            question = state.rewritten_query
        else:
            question = state.user_query
        context = state.context if state.context else ""

        responses: GradeDocument = self.llm_model.with_structured_output(
            GradeDocument
        ).invoke(
            [
                {"role": "system", "content": self.prompt.grade_document_relevance},
                {
                    "role": "user",
                    "content": f"Determine if the context is relevant to the question. Question: {question} Context: {context}",
                },
            ]
        )

        if responses.is_relevant():
            return "generate_answer"
        elif responses.is_create_ticket() or (
            responses.is_not_relevant() and state.already_rewritten
        ):
            return "create_ticket"
        elif responses.is_query_not_match_with_document():
            return "generate_sorry_response"
        elif responses.is_not_relevant() and not state.already_rewritten:
            return "rewrite_query"
        else:
            return "create_ticket"

    def create_ticket(
        self, state: CustomerSupportWorkflowState
    ) -> CustomerSupportWorkflowState:
        """Create user problem ticket

        Args:
            state (CustomerSupportWorkflowState): state of the workflow

        Returns:
            CustomerSupportWorkflowState: updated state of the workflow
        """
        question = state.histories

        question.extend(
            [
                {
                    "role": "system",
                    "content": self.prompt.create_ticket,
                },
                {
                    "role": "user",
                    "content": f"Create a support ticket based on the above conversation. For use query: {state.user_query}",
                },
            ]
        )
        response: UserTicket = self.llm_model.with_structured_output(UserTicket).invoke(
            question
        )
        state.set_ticket(response)
        return state

    def generate_answer(self, state) -> CustomerSupportWorkflowState:
        """Generated answer based on retrieved documents

        Args:
            state (CustomerSupportWorkflowState): state of the workflow

        Returns:
            CustomerSupportWorkflowState: updated state of the workflow
        """
        if state.already_rewritten:
            question = state.rewritten_query
        else:
            question = state.user_query
        context = state.histories[-1].content
        response = self.llm_model.invoke(
            [
                {"role": "system", "content": self.prompt.generate_answer},
                {"role": "user", "content": f"Question: {question} Context: {context}"},
            ]
        )
        state.set_generated_response(response.content)
        return state

    def rewrite_query(
        self, state: CustomerSupportWorkflowState
    ) -> CustomerSupportWorkflowState:
        """Rewrite the user initial query

        Args:
            state (CustomerSupportWorkflowState): state of the workflow

        Returns:
            CustomerSupportWorkflowState: updated state of the workflow
        """
        initial_user_query = state.user_query
        rewrite_query = self.llm_model.invoke(
            [
                {"role": "system", "content": self.prompt.rewrite_query},
                {
                    "role": "user",
                    "content": f"Initial user query: {initial_user_query} initial_context: {state.context}",
                },
            ]
        )
        state.set_rewritten_query(rewrite_query.content)
        state.set_already_rewritten(True)

        return state

    def get_compiled_graph(self):
        """Get the compiled graph

        Returns:
            CompiledStateGraph: compiled state graph
        """
        return self.__graph


# if __name__ == "__main__":
#     from app_config import app_config
#     from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
#     from poc_chatbot.backend.embedding_vector_store import HybridEmbeddingVectorStore
#     from poc_chatbot.infrastructure.llm_gemini import (
#         get_gemini_llm,
#         get_gemini_embedding_model,
#         GeminiEmbeddingTask,
#     )
#     from poc_chatbot.infrastructure.qdrant_client_instance import QdrantClientInstance

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

#     qdrant_vector_store = QdrantVectorStore(
#         client=qdrant_client,
#         collection_name="hybrid_documents",
#         embedding=embedding_model,
#         sparse_embedding=sparse_embeddings,
#         retrieval_mode=RetrievalMode.HYBRID,
#         vector_name="dense",
#         sparse_vector_name="sparse",
#     )
#     prompt = PromptCustomerSupport()
#     workflow = CustomerSupportWorkflow(
#         llm_model=llm, vector_store=qdrant_vector_store, prompt=prompt
#     )
#     graph = workflow.get_compiled_graph()
#     #     state = CustomerSupportWorkflowState(
#     #         user_query="""
#     #    Hi, my account is blocked and I can't access my dashboard.
#     #    I tried resetting my password but didn't receive the reset email.
#     #    Can you help me resolve this issue quickly? My username is john_doe and my email is john_doe@example.com.

#     #         """,
#     #     )

#     #     state = CustomerSupportWorkflowState(
#     #         user_query="""
#     #    What service does Alibaba Cloud provide for deploying serverless applications?

#     #         """,
#     #     )

#     state = CustomerSupportWorkflowState(
#         user_query="""
#   What service that help collaborations?
        
#         """,
#     )
#     # result_state = graph.invoke(state)
#     # response = TypeAdapter(CustomerSupportWorkflowState).validate_python(result_state)
#     # print("User Query:", response.user_query)
#     # print("Rewritten Query:", response.rewritten_query)
#     # print("Is Rewritten:", response.already_rewritten)
#     # if response.generated_response:
#     #     print("Generated Response:", response.generated_response)
#     # if response.ticket:
#     #     print("Generated Ticket:", response.ticket.model_dump())
    
#     for message_chunk in graph.stream(state):
#        print(message_chunk)
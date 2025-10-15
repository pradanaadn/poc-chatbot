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
from pydantic import BaseModel, Field, TypeAdapter


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
        "You are an AI customer service assistant. Your task is to retrieve relevant information from the knowledge base to answer user queries."
        " Use the document_retriever tool to search for relevant documents when the user asks a question."
        " Do NOT engage in conversation, ask follow-up questions, or request additional information from users."
        " Simply retrieve documents that might contain the answer to their query."
        " If the user provides all necessary details (like email, username, issue description), still search for relevant documentation first.",
        description="Prompt for node generate_query_or_response",
    )
    grade_document_relevance: str = Field(
        "You are an AI assistant that evaluates whether retrieved documents contain sufficient information to answer a user's query."
        " Your task is to determine if you can provide a complete and accurate answer based solely on the provided context."
        "\n\nRespond with one of FOUR values:\n"
        " - 'relevant': The context contains sufficient information to fully answer the user's question."
        "   You can generate a complete, accurate response using ONLY the information in the context."
        "   Example: Query 'how to deploy AWS Lambda' + context with Lambda deployment steps → 'relevant'\n"
        " - 'not_relevant': The context is about the same product/service/company, but does NOT contain the specific information needed to answer the question."
        "   This means the query was TOO VAGUE about which aspect is needed, causing retrieval of related but unhelpful documents."
        "   Example 1: Query 'how to deploy AWS Lambda' + context only has 'AWS Lambda pricing' → 'not_relevant' (same product, wrong aspect)"
        "   Example 2: Query 'AWS billing issues' + context only has 'AWS deployment configs' → 'not_relevant' (same service, wrong topic)"
        "   Example 3: Query 'account access problems' + context only has 'API authentication methods' → 'not_relevant' (related but different aspect)"
        "   Example 4: Query 'how to secure my app' + context only has 'deployment steps' → 'not_relevant' (same app, but missing security info)"
        "   When 'not_relevant': The query needs MORE SPECIFICITY about the aspect (deployment vs pricing vs security vs troubleshooting).\n"
        " - 'create_ticket': The query requires ACTION, troubleshooting, or specific information that needs coordination with technical teams. Only for the query about errors that cannot be answered by context."
        "   Use this when:"
        "   - User reports an error, bug, or system malfunction (e.g., '500 error when deploying', 'API not responding')"
        "   - User requests account-specific actions (e.g., 'reset my password', 'change my billing plan', 'delete my account')"
        "   - User needs investigation of their specific case (e.g., 'why is my bill $1000', 'my deployment failed 3 times')"
        "   - User requests features or changes (e.g., 'add support for Python 3.12', 'enable custom domain')"
        "   - Query requires access to user's account data, logs, or system internals"
        "   - Issue needs human intervention or technical team expertise"
        "   Example 1: 'My deployment keeps failing with error XYZ' → 'create_ticket' (needs investigation)"
        "   Example 2: 'Can you reset my password for john@example.com' → 'create_ticket' (requires action)"
        "   Example 3: 'Why am I being charged $500 this month' → 'create_ticket' (needs account review)"
        "   Example 4: 'The dashboard is not loading' → 'create_ticket' (system issue requiring investigation)\n"
        " - 'query_not_match_with_document': The query and context are about completely different products, companies, or unrelated domains."
        "   Example 1: Query about 'Google Cloud' + context about 'AWS services' → 'query_not_match_with_document'"
        "   Example 2: Query about 'Microsoft Azure' + context about 'cooking recipes' → 'query_not_match_with_document'"
        "   Example 3: Query about 'Python programming' + context about 'gardening tips' → 'query_not_match_with_document'\n"
        "\nDecision criteria:"
        " 1. Does the query require ACTION or investigation of user-specific data? If YES → 'create_ticket'"
        " 2. Can you answer the question using ONLY the context? If YES → 'relevant'"
        " 3. Are query and context about the same product/service? If NO → 'query_not_match_with_document'"
        " 4. If same product but can't answer: Is it because the context covers a different aspect? If YES → 'not_relevant' (query too vague)"
        " 5. Remember: 'create_ticket' is for actionable requests or user-specific issues, NOT for general questions that just need better retrieval",
        description="Prompt for node grade_document_relevance",
    )
    rewrite_query: str = Field(
        "You are an AI assistant that rewrites user queries to improve retrieval accuracy."
        " You are rewriting because the initial search found documents about the SAME product/service/company, but they didn't answer the specific question asked."
        "\n\nYour task:"
        " - Analyze the initial query to understand what the user is actually asking about"
        " - Look at the context ONLY to identify relevant terminology or platform names that match the user's intent"
        " - DO NOT change the topic or subject of the user's query"
        " - If the context is about a different topic than the user's query, IGNORE it and focus on making the original query more specific"
        "\n\nIMPORTANT RULES:"
        " - Preserve the CORE TOPIC from the user's original query"
        " - Only use context if it's actually relevant to what the user asked"
        " - If the user asks about 'web application', keep it about web applications (NOT streaming, NOT databases, NOT other services)"
        " - If the user asks about 'deployment', focus on deployment (NOT pricing, NOT monitoring)"
        "\n\nHandling broad queries:"
        " - 'best practices for web application' → 'best practices for deploying web applications' or 'web application security best practices'"
        " - 'how to deploy' → 'how to deploy applications' or 'deployment steps for web services'"
        " - 'pricing information' → 'pricing plans and costs' or 'billing information'"
        "\n\nRewriting strategy:"
        " - Make the query more specific while keeping the SAME topic"
        " - Add clarifying terms that narrow the scope (e.g., 'deployment', 'security', 'performance')"
        " - Use platform/technology names from context ONLY if they match what the user is asking about"
        " - If context mentions a different topic (e.g., streaming when user asks about web apps), ignore the context"
        " - Keep the rewritten query focused on the user's ORIGINAL intent"
        "\n\nExamples:"
        " - User: 'best practices for web application' + Context about 'AWS streaming' → Rewrite: 'best practices for web application deployment and architecture' (IGNORE streaming context)"
        " - User: 'how to deploy' + Context about 'Lambda functions' → Rewrite: 'how to deploy applications using AWS Lambda' (USE context)"
        " - User: 'security tips' + Context about 'pricing' → Rewrite: 'security best practices and recommendations' (IGNORE pricing context)"
        "\n\nReturn only the rewritten query without explanations.",
        description="Prompt for node rewrite_query",
    )
    create_ticket: str = Field(
        "You are an AI assistant that creates support tickets based on user interactions."
        " Your task is to generate a concise and informative support ticket that summarizes the user's issue."
        " The ticket should include a title, a detailed description of the issue, relevant labels or tags, and a priority level (low, medium, high, urgent)."
        " Ensure the detail description captures the essence of the user's problem based on the conversation history provided."
        " If the user's issue is not clear from the conversation, make reasonable assumptions to fill in the details (state if this an assumption)."
        " Do not include any personally identifiable information (PII) in the ticket.",
        description="Prompt for node create_ticket",
    )
    generate_answer: str = Field(
        "You are an AI customer service assistant. Your task is to provide accurate and helpful responses to user inquiries."
        "You will be given some context to help you answer the question. If the context is relevant to the question, use it to formulate your answer."
        " If the context is not relevant, do not use it and respond with a polite apology indicating that you couldn't find relevant information in your knowledge base to answer the question."
        " Do not attempt to fabricate an answer or provide information that is not supported by the context.",
        description="Prompt for node generate_answer",
    )
    generate_sorry_response: str = Field(
        "You are an AI customer service assistant."
        " The system could not find relevant information in the knowledge base to answer the user's question."
        " Inform the user that you couldn't find the relevant information they need."
        " Tell them that a support ticket has been created and a human support agent will contact them to assist with their issue."
        " Do NOT ask follow-up questions, engage in conversation, or request additional details."
        " Keep your response brief, professional, and informative.",
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


if __name__ == "__main__":
    from app_config import app_config
    from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
    from poc_chatbot.backend.embedding_vector_store import HybridEmbeddingVectorStore
    from poc_chatbot.infrastructure.llm_gemini import (
        get_gemini_llm,
        get_gemini_embedding_model,
        GeminiEmbeddingTask,
    )
    from poc_chatbot.infrastructure.qdrant_client_instance import QdrantClientInstance

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
    llm = get_gemini_llm(api_key=gemini_api_key)

    qdrant_vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name="hybrid_documents",
        embedding=embedding_model,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
    )
    prompt = PromptCustomerSupport()
    workflow = CustomerSupportWorkflow(
        llm_model=llm, vector_store=qdrant_vector_store, prompt=prompt
    )
    graph = workflow.get_compiled_graph()
    #     state = CustomerSupportWorkflowState(
    #         user_query="""
    #    Hi, my account is blocked and I can't access my dashboard.
    #    I tried resetting my password but didn't receive the reset email.
    #    Can you help me resolve this issue quickly? My username is john_doe and my email is john_doe@example.com.

    #         """,
    #     )

    #     state = CustomerSupportWorkflowState(
    #         user_query="""
    #    What service does Alibaba Cloud provide for deploying serverless applications?

    #         """,
    #     )

    state = CustomerSupportWorkflowState(
        user_query="""
   What the best practice for web application?
        
        """,
    )
    result_state = graph.invoke(state)
    response = TypeAdapter(CustomerSupportWorkflowState).validate_python(result_state)
    print("User Query:", response.user_query)
    print("Rewritten Query:", response.rewritten_query)
    print("Is Rewritten:", response.already_rewritten)
    if response.generated_response:
        print("Generated Response:", response.generated_response)
    if response.ticket:
        print("Generated Ticket:", response.ticket.model_dump())

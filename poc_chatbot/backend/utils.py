import uuid
from langchain_core.documents import Document
from poc_chatbot.backend.schema import DocumentChunk


def generate_id(text: str) -> str:
    id = uuid.uuid5(uuid.NAMESPACE_OID, text)
    return str(id)


def create_batches(input_list, batch_size) -> list:
    """
    Splits a list into smaller batches of a specified size.

    Args:
        input_list (list): The original list to be batched.
        batch_size (int): The desired size of each batch.

    Returns:
        list: A list of lists, where each inner list is a batch.
    """
    batches = []
    for i in range(0, len(input_list), batch_size):
        batch = input_list[i : i + batch_size]
        batches.append(batch)
    return batches


def document_adapter(documents: list[DocumentChunk]) -> list[Document] | list[str]:
    langchain_docs = []
    ids = []
    for doc in documents:
        langchain_documents = Document(
            page_content=doc.text,
            metadata=doc.metadata.model_dump() if doc.metadata else {},
        )
        ids.append(doc.chunk_id)
        langchain_docs.append(langchain_documents)
    return langchain_docs, ids


# if __name__ == "__main__":
#     text = "Hello, world!"
#     document_title = "My Document"
#     page_number = 1
#     unique_id = generate_id(text, page_number, document_title)
#     print(f"Generated ID: {unique_id}")
#     text = "Hello, world!"
#     page_number = 1
#     document_title = "My Document"
#     unique_id = generate_id(text, page_number, document_title)
#     print(f"Generated ID: {unique_id}")

import pymupdf4llm
from pydantic import BaseModel, Field
import uuid
import hashlib
import datetime

class DocumentChunkMetadata(BaseModel):
    page_number: int = Field(ge=1)
    max_page: int = Field(ge=1)
    source: str = Field(description="Source file path or URL")
    subject: str | None = Field(default=None)
    title: str = Field(default="")
    keywords: list[str] = Field(default=[])
    authors: list[str] = Field(default=[])


class DocumentChunk(BaseModel):
    chunk_id: str = Field(description="Unique identifier for the document chunk")
    text: str = Field(description="Text content of the document chunk")
    metadata: DocumentChunkMetadata | None = Field(
        default=None, description="Metadata associated with the document chunk"
    )

def generate_id(text:str, page_number:int, create_date:datetime.datetime) -> str:
    return hashlib.sha256(f"{page_number}-{text}-{create_date}".encode()).hexdigest()

class DocumentProcessingPipeline:
    def __init__(self, source: str):
        self.source = source
        self.chunks: list[DocumentChunk] = []

    def load(self):
        self.load_document = pymupdf4llm.to_markdown(
            doc=self.source,
            page_chunks=True,
            show_progress=True,
            ignore_alpha=True,
            ignore_images=True,
            ignore_graphics=True,
            force_text=True,
        )

    def chunk(self)-> list[DocumentChunk]:
        for page_data in self.load_document:
            page_markdown = page_data["text"]
            page_metadata = page_data["metadata"]
            chunk = DocumentChunk(
                chunk_id=generate_id(page_markdown, page_metadata.get("page", 0), datetime.datetime.now()),
                text=page_markdown,
                metadata=DocumentChunkMetadata(
                    page_number=page_metadata.get("page", 0),
                    max_page=page_metadata.get("page_count", 0),
                    source=self.source,
                    subject=page_metadata.get("subject", None),
                    title=page_metadata.get("title", ""),
                    keywords=list(set(page_metadata.get("keywords", []).split(","))),
                    authors=list(set(page_metadata.get("author", []).split(","))),
                    ),
            )
            self.chunks.append(chunk)
        return self.chunks
    


# if __name__ == "__main__":
#     process = DocumentProcessingPipeline("assets/example_documents.pdf")
#     process.load()
#     chunks = process.chunk()
#     for chunk in chunks:
#         print(chunk.model_dump())
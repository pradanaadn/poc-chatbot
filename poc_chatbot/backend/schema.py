from pydantic import BaseModel, Field


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




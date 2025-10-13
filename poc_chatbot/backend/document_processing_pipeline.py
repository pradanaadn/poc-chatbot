import pymupdf4llm
from poc_chatbot.backend.schema import DocumentChunk, DocumentChunkMetadata
from poc_chatbot.backend.utils import generate_id

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
            pages=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],  
        )

    def chunk(self) -> list[DocumentChunk]:
        for page_data in self.load_document:
            page_markdown = page_data["text"]
            page_metadata = page_data["metadata"]
            keywords = set(map(lambda x: x.strip(), page_metadata.get("keywords", "").split(",")))
            authors = set(map(lambda x: x.strip(), page_metadata.get("author", "").split(",")))
            chunk = DocumentChunk(
                chunk_id=generate_id(page_markdown, page_metadata.get("page", 0), page_metadata.get("title", "")),
                text=page_markdown,
                metadata=DocumentChunkMetadata(
                    page_number=page_metadata.get("page", 0),
                    max_page=page_metadata.get("page_count", 0),
                    source=self.source,
                    subject=page_metadata.get("subject", None),
                    title=page_metadata.get("title", ""),
                    keywords=list(keywords),
                    authors=list(authors),
                ),
            )
            self.chunks.append(chunk)
        return self.chunks


# if __name__ == "__main__":
#     process = DocumentProcessingPipeline("assets/example_documents.pdf")
#     process.load()
#     chunks = process.chunk()
#     print(f"Total Chunks: {len(chunks)}")
#     for chunk in chunks:
#         print(chunk.model_dump())

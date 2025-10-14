import pymupdf4llm
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from yake import KeywordExtractor
from poc_chatbot.backend.schema import DocumentChunk, DocumentChunkMetadata
from poc_chatbot.backend.utils import generate_id
from poc_chatbot.backend.clean_text_pipeline import CleanTextPipeline


class DocumentProcessingPipeline:
    def __init__(self, source: str, text_cleaner: CleanTextPipeline = None,  keyword_extractor: KeywordExtractor = None):
        
        self.source = source
        self.chunks: list[DocumentChunk] = []
        self.text_cleaner = text_cleaner
        self.keyword_extractor = keyword_extractor

    def load(self, exclude_pages: list[int] = None):
        """Load PDF documents

        Args:
            exclude_pages (list[int], optional): List of page numbers to exclude from loading. Defaults to None.
        """
        page_count = pymupdf.open(self.source).page_count
        all_pages = list(range(page_count))

        if exclude_pages:

            all_pages = [p for p in all_pages if p not in exclude_pages]

        self.load_document = pymupdf4llm.to_markdown(
            doc=self.source,
            show_progress=True,
            ignore_alpha=True,
            ignore_images=True,
            ignore_graphics=True,
            force_text=True,
            pages=all_pages,
        )
        
    def keyword_extraction(self, text: str, max_keywords: int = 10) -> list[str]| None:
        """Extract keywords from text using YAKE.

        Args:
            text (str): Text to extract keywords from.
            max_keywords (int, optional): Maximum number of keywords to extract. Defaults to 10.
        Returns:
            list[str]: List of extracted keywords.
        """
        if self.keyword_extractor:
            keywords = self.keyword_extractor.extract_keywords(text)
            return [kw[0].upper() for kw in keywords]

    def chunk(self, chunk_size = 900, chunk_overlap=300) -> list[DocumentChunk]:
        """Chunk loaded text based on length.

        Raises:
            ValueError: If the document is not loaded. Call load() before chunk().

        Returns:
            list[DocumentChunk]: List of document chunks.
        """
        if not self.load_document:
            raise ValueError("Document not loaded. Call load() before chunk().")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        split_result = text_splitter.split_text(self.load_document)
        for chunk_text in split_result:
            if self.text_cleaner:
                chunk_text = self.text_cleaner.run(chunk_text)
            metadata = DocumentChunkMetadata(keywords=self.keyword_extraction(chunk_text) or [])
            chunk = DocumentChunk(chunk_id=generate_id(chunk_text), text=chunk_text, metadata=metadata)
            self.chunks.append(chunk)
        return self.chunks


# if __name__ == "__main__":
#     keyword_extractor = KeywordExtractor(lan="en", n=3, dedupLim=0.95, dedupFunc="jaro", top=3)
#     clean_text_pipeline = CleanTextPipeline()
#     process = DocumentProcessingPipeline("assets/example_documents.pdf", text_cleaner=clean_text_pipeline, keyword_extractor=keyword_extractor)
#     process.load(exclude_pages=list(range(0, 11)))
#     chunks = process.chunk()
#     print(f"Total Chunks: {len(chunks)}")
#     for chunk in chunks:
#         print(chunk.model_dump())

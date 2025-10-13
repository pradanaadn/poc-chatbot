
import pymupdf4llm
llama_reader = pymupdf4llm.LlamaMarkdownReader()
md_text = llama_reader.load_data("assets/example_documents.pdf", show_progress=True)
for text in md_text:
    # pathlib.Path("assets/extracted.md").write_text(text)
    print(text)

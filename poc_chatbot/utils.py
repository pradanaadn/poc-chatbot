from pathlib import Path

def read_markdown_file(file_path: str) -> str:
    """Reads a markdown file and returns its content as a string."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with path.open('r', encoding='utf-8') as file:
        content = file.read()
    
    return content


# if __name__ == "__main__":
#     # Example usage
#     try:
#         markdown_content = read_markdown_file('poc_chatbot/prompt/create_ticket.md')
#         print(markdown_content)
#     except FileNotFoundError as e:
#         print(e)
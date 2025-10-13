import uuid

def generate_id(text:str, page_number:int, document_title:str) -> str:
    id =  uuid.uuid5(uuid.NAMESPACE_OID, f"{page_number}-{text}-{document_title}")
    return str(id)


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
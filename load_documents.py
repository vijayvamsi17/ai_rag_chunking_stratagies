import os
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_community.document_loaders.generic import GenericLoader

def load_documents():
    # Load documents from a directory
    if not os.path.exists("./docs/"):
        print(
            "Directory './docs/' does not exist. Please create it and add your documents.")
        return []

    loader = GenericLoader(
        blob_loader=FileSystemBlobLoader(path="./docs/"),
        blob_parser=PyPDFParser()
    )
    documents = loader.load()

    # Print details of the first 2 documents
    for i, doc in enumerate(documents[:2]):
        print(f"Documents: {i + 1}:")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content: {doc.page_content}") # Print the first 100 characters of the content
        print(f"Metadata: {doc.metadata}")

    return documents
import os
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

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


def chunk_documents(documents):
    print("Chunking documents...")
    semantic_splitter = SemanticChunker(
        embeddings=OllamaEmbeddings(model="nomic-embed-text"),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=70
        )
    chunks = semantic_splitter.split_documents(documents)

    if chunks:
        print(f"Created {len(chunks)} chunks.")
        # Print details of the first 2 chunks
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i + 1}:")
            print(f"Source: {chunk.metadata.get('source')}")
            print(f"Content length: {len(chunk.page_content)} characters")
            print(f"Content: {chunk.page_content}.") # Print the first 100 characters of the chunk content
            print("-" * 50)
        if len(chunks) > 5:
            print(f"... and {len(chunks) - 5} more chunks.")
    return chunks


def main():
    print("Character Chunking Main Function")
    # Load documents from a directory
    documents = load_documents()
    # Here you can add your character chunking logic using the loaded documents
    chucks = chunk_documents(documents)


if __name__ == "__main__":
    main()

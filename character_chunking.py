import os
from langchain_community.document_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.parsers import PyPDFParser
from langchain_community.document_loaders.generic import GenericLoader
from langchain_text_splitters import CharacterTextSplitter


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
        print(f"Content: {doc.page_content[:100]}...") # Print the first 100 characters of the content
        print(f"Metadata: {doc.metadata}")

    return documents


def chunk_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Chunking documents...")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    if chunks:
        print(f"Created {len(chunks)} chunks.")
        # Print details of the first 2 chunks
        for i, chunk in enumerate(chunks[:5]):
            print(f"Chunk {i + 1}:")
            print(f"Source: {chunk.metadata.get('source')}")
            print(f"Content length: {len(chunk.page_content)} characters")
            print(f"Content: {chunk.page_content[:100]}...") # Print the first 100 characters of the chunk content
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

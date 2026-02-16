from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from get_embedding_function import get_embedding_function
from load_documents import load_documents

def chunk_documents(documents):
    print("Chunking documents...")
    semantic_splitter = SemanticChunker(
        embeddings=get_embedding_function(),
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
    print("Semantic Chunking Main Function")
    # Load documents from a directory
    documents = load_documents()
    # Here you can add your character chunking logic using the loaded documents
    chucks = chunk_documents(documents)


if __name__ == "__main__":
    main()

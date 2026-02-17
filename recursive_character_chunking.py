from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_documents import load_documents
from populate_database import create_vector_store

def chunk_documents(documents, chunk_size=800, chunk_overlap=0):
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
        )
    chunks = text_splitter.split_documents(documents)

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
    print("Recursive Character Chunking Main Function")
    # Load documents from a directory
    documents = load_documents()
    # Here you can add your character chunking logic using the loaded documents
    chucks = chunk_documents(documents)
    # Create a vector store from the chunks
    vector_store = create_vector_store(chucks, persist_directory="db/chroma_recursive_character_db")

if __name__ == "__main__":
    main()

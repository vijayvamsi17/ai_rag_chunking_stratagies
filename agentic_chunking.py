from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaLLM
from load_documents import load_documents

sample_text = """
The University began as a commuter school during the post-World War II boom when
returning soldiers were looking for an education financed by the G.I. Bill
®
. The movers and
shakers of Hartford recognized a need for a university and brought together three small
schools housed in buildings spread across the city, including the Wadsworth Atheneum
Museum of Art and the Hartford YMCA.
Unlike most private New England colleges, the University of Hartford did not begin as a
small liberal arts institution. From the outset, it has offered courses in electronics,
engineering, technology, and education along with strong programs in music, the visual arts,
and the arts and sciences. Today, it is known for excellence in the visual and performing arts,
engineering, and business; small classes; and its focus on mentoring all students.
The University has always been coed and open to all students, regardless of their
background. Designed initially to meet the needs of Hartford residents, it has stayed true to
the founders’ ideals but greatly surpassed its modest goals. Its mission today is to educate
students as citizens of the world, encouraging them to study abroad, get involved in
community service, and take responsibility for the planet and their futures.
"""

def chunk_documents():
    llm = OllamaLLM(model="llama3.2:latest")

    prompt = f"""
    You are a text chunking expert. Split this text into logical chunks.

    Rules:
    - Each chunk should be around 200 characters or less
    - Keep related information together in the same chunk
    - Put "<<<SPLIT>>>" between chunks

    Text:
    {sample_text}

    Return the text with "<<<SPLIT>>>" between chunks, and do not include any additional commentary or formatting.
    """

    print("Initiating AI chunking:")
    response = llm.invoke(prompt)
    print("AI chunking completed. Processing response... {response}")
    marked_chunks = response
    chunks = marked_chunks.split("<<<SPLIT>>>")

    clean_chunks = []
    for chunk in chunks:
        clean_chunk = chunk.strip()
        if clean_chunk:
            clean_chunks.append(clean_chunk)

    print(f"Generated chunks:")
    for i, chunk in enumerate(clean_chunks, 1):
        print(f"Chunk {i}: {chunk} (Length: {len(chunk)} characters)")
        print(f'"{chunk}"')
        print("-" * 50)

def main():
    print("Agentic Chunking Main Function")
 
    # Here you can add your character chunking logic using the loaded documents
    chucks = chunk_documents()


if __name__ == "__main__":
    main()

import chromadb
import requests
import json
import os
import sys
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b" 
MAX_DISTANCE = 1.0

client = chromadb.PersistentClient(path="./rag_db")
collection = client.get_or_create_collection("anime_docs")


# ---ingestion---
def load_documents(directory):
    """Load all files from a directory and chunk them by paragraph."""
    docs = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(('.txt', '.md')):
            with open(os.path.join(directory, filename), 'r') as f:
                content = f.read()
            paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
            for i, para in enumerate(paragraphs):
                docs.append({
                    "text": para,
                    "id": f"{filename}_{i}",
                    "metadata": {"source": filename, "chunk_index": str(i)}
                })
    return docs


def ingest(collection, docs_directory):
    """Load, chunk, and store documents in ChromaDB."""
    if not os.path.isdir(docs_directory):
        print(f"Dirctory '{docs_directory}' not found!")
        return 0
    
    chunks = load_documents(docs_directory)
    if not chunks:
        print("No documents found!")
        return 0
    
    collection.upsert(
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
        ids=[c["id"] for c in chunks]
    )
    return len(chunks)


#----retrieval----

def retrieve(collection, query, n_result=3, max_distance=MAX_DISTANCE):
    """Search ChromaDB for relevant document chunks and only keep chunks closer than max_distance."""
    if collection.count() == 0:
        return []
    
    results = collection.query(
        query_texts=[query],
        n_results=min(n_result, collection.count()),
    )

    chunks = []
    for i in range(len(results['documents'][0])):
        distance = results['distances'][0][i]
        if distance <= max_distance:
            chunks.append({
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })
    return chunks


def get_confidence(chunks):
    """Guardrails 2: confidence from the best (smallest) distance."""
    if not chunks:
        return "none"
    best = chunks[0]["distance"]
    if best < 0.5:
        return "high"
    elif best < 1.0:
        return "medium"
    else:
        return "low"
    

# --- generation ---

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions about anime and fiction.
Answer the user's question based ONLY on the context provided below.

Rules:
- Only use information from the CONTEXT section
- If the context doesn't contain the answer, say "I don't have enough information to answer that based on the available documents."
- NEVER make up information not present in context.
- Always cite your sources by mentioning the document name in parentheses.
- If the context is only loosely related to the question, acknowledge this.
- Keep your response under 200 words.
"""


def build_rag_prompt(question, retrieved_chunks):
    """Assemble the RAG prompt from retrieved context and user question"""
    context_parts = []
    for chunk in retrieved_chunks:
        source = chunk["metadata"]["source"]
        text = chunk["text"]
        context_parts.append(f"[Source: {source}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)


    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n\n"
            "Answer using only the context above. "
            "Cite the source file name in parentheses."
        )},
    ]


def generate(messages, stream=False):
    """Call Ollama to generate a response."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat", 
            json={
                "model": MODEL,
                "messages": messages,
                "stream": stream,
                "options": {"temperature": 0.1},
            }, 
            stream=stream,
            timeout=300,
        )
        response.raise_for_status()

        if stream:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        full_response += token
            print()
            return full_response
        return response.json()["message"]["content"]
        
    except requests.exceptions.ConnectionError:
        msg = "ERROR: Cannot connect to Ollama. Is it running? (ollama serve)"
        print(msg)
        return msg
    except requests.exceptions.HTTPError as e:
        msg = f"ERROR: Ollama returned an error ({e}). Did you run 'ollama pull {MODEL}'?"
        print(msg)
        return msg
    

def validate_response(answer, retrieved_sources):
    """Basic sanity checks on the generated response."""
    warnings = []
    lowered = answer.lower()

    if answer.startswith("ERROR"):
        warnings.append("Generation failed")
        return warnings

    if len(answer.split()) > 200:
        warnings.append("Response unusually long - may contain hallucinated details")

    refused =  "don't have enough information" in lowered or "i don't know" in lowered
    if retrieved_sources and not refused:
        if not any(src.lower() in lowered for src in retrieved_sources):
            warnings.append("No source document cited in the answer")
    return warnings


    
# ---full pipeline---

def rag_query(collection, question, n_results=3, stream=True):
    """Complete RAG pipeline: retrieve + build prompt + generate."""
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print(f"{'=' * 60}")

    # Retrieve
    chunks = retrieve(collection, question, n_results, max_distance=MAX_DISTANCE)

    if not chunks:
        print("\nNo chunks passed the distance threshold.")
        return {
            "answer": "I don't have enough relevant information in my knowledge base to answer that question.",
            "sources": [],
            "confidence": "low",
            "chunks_retrieved": 0,
        }
    
    confidence = get_confidence(chunks)
    
    print(f"\nRetrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  [{i+1}] (dist: {chunk['distance']:.4f}) {chunk['metadata']['source']}")
        print(f"       {chunk['text'][:100]}...")

    # Build prompt
    messages = build_rag_prompt(question, chunks)

    # Generate
    print(f"\nAnswer (streaming):\n")
    start = time.time()
    answer = generate(messages, stream=stream)
    elapsed = time.time() - start
    print(f"\n[Generated in {elapsed:.1f}s]")

    sources = list(dict.fromkeys(c["metadata"]["source"] for c in chunks))

    for warning in validate_response(answer, sources):
        print(f"WARNING: {warning}")


    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "chunks_retrieved": len(chunks),
    }



TEST_QUESTIONS = [
    ("Answerable from docs", "What is Equivalent Exchange in Fullmetal Alchemist?"),
    ("Related but not in docs", "Who composed the music for Spirited Away?"),
    ("Outside scope", "How do I bake sourdough bread?"),
]

def run_tests(collection):
    """Run one question of each type"""
    for label, question in TEST_QUESTIONS:
        print(f"\n##### TEST TYPE: {label} #####")
        rag_query(collection, question)




if __name__ == "__main__":

    if collection.count() == 0:
        count = ingest(collection, "docs")
        print(f"Ingested {count} document chunks.")
    else:
        print(f"Using existing collection: {collection.count()} chunks")

    if "--test" in sys.argv:
        run_tests(collection)
        sys.exit(0)

    print("\nRAG Pipeline Ready! Ask questions about your documents.")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ('quit', 'exit', 'q'):
            break
        if question:
            rag_query(collection, question)



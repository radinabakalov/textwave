import os
import glob

import numpy as np
import faiss
from flask import Flask, request, jsonify

from textwave.modules.extraction.preprocessing import DocumentProcessing
from textwave.modules.extraction.embedding import Embedding
from textwave.modules.generator.question_answering import QAGeneratorMistral


app = Flask(__name__)

# Path to the corpus relative to this file, so it works regardless of cwd
STORAGE_DIRECTORY = os.path.join(os.path.dirname(__file__), "storage")

CHUNKING_STRATEGY = 'sentence' # 'sentence' or 'fixed-length'
CHUNKING_PARAMETERS = {
    "num_sentences": 3,
    "overlap_size": 1,
}

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

INDEX_STRATEGY = "bruteforce" # uses FAISS IndexFlatL2
INDEX_PARAMETERS = {}

TOP_K = 5  # number of chunks to retrieve per query


# global index and chunk store -> built once at module load
_index = None
_chunks = []


def initialize_index():
    """
    1. Parse through all the documents contained in the storage directory.
    2. Chunk each document using the configured chunking strategy.
    3. Embed each chunk using the Embedding class.
    4. Store vector embeddings in a FAISS index (bruteforce = IndexFlatL2).
    5. Return the FAISS index.
    """
    global _chunks

    processor = DocumentProcessing()
    embedder = Embedding(EMBEDDING_MODEL)

    all_chunks = []

    # Find every .txt file in the storage directory (recursively)
    pattern = os.path.join(STORAGE_DIRECTORY, "**", "*.txt")
    txt_files = glob.glob(pattern, recursive=True)

    for filepath in txt_files:
        if CHUNKING_STRATEGY == 'sentence':
            chunks = processor.sentence_chunking(
                filepath,
                num_sentences=CHUNKING_PARAMETERS["num_sentences"],
                overlap_size=CHUNKING_PARAMETERS["overlap_size"],
            )
        else:
            chunks = processor.fixed_length_chunking(
                filepath,
                chunk_size=CHUNKING_PARAMETERS["chunk_size"],
                overlap_size=CHUNKING_PARAMETERS["overlap_size"],
            )
        all_chunks.extend(chunks)

    # If there are no documents, return a minimal empty index so the app still starts
    if not all_chunks:
        dim = embedder.encode("placeholder").shape[0]
        empty_index = faiss.IndexFlatL2(dim)
        _chunks = []
        return empty_index

    # Embed all chunks
    embeddings = np.array([embedder.encode(chunk) for chunk in all_chunks], dtype=np.float32)

    # Build the FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Store chunks so we can retrieve text by index position later
    _chunks = all_chunks

    return index


def _search_index(index: faiss.IndexFlatL2, query: str, k: int) -> list[str]:
    """Embed the query and return the top-k most relevant chunks from the index."""
    embedder = Embedding(EMBEDDING_MODEL)
    query_vec = np.array([embedder.encode(query)], dtype=np.float32)

    # Cap k at the number of indexed chunks to avoid FAISS errors
    k = min(k, index.ntotal)
    if k == 0:
        return []

    _, indices = index.search(query_vec, k)
    return [_chunks[i] for i in indices[0] if i < len(_chunks)]


@app.route("/generate", methods=["POST"])
def generate_answer():
    """
    Generate an answer to a given query by running the retrieval and generation pipeline.

    Accepts a POST request with a JSON body containing a "query" field.
    Retrieves the top-k relevant chunks from the FAISS index, then uses the
    Mistral API to generate an answer grounded in those chunks.

    Example:
        curl -X POST http://localhost:5000/generate \
             -H "Content-Type: application/json" \
             -d '{"query": "What is the role of antioxidants in green tea?"}'

    Returns:
        JSON with "query" and "answer" fields, or an error response.
    """
    # Must be JSON 
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request must include a JSON body."}), 400

    query = data.get("query", "")

    # Reject empty or whitespace-only queries
    if not query or not query.strip():
        return jsonify({"error": "Query must be a non-empty string."}), 422

    query = query.strip()

    # Retrieve the most relevant chunks from the index
    context = _search_index(_index, query, k=TOP_K)

    # Generate an answer using Mistral, grounded in the retrieved context
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    generator = QAGeneratorMistral(api_key=api_key)
    answer = generator.generate_answer(query=query, context=context)

    return jsonify({"query": query, "answer": answer})


# Build the index once when the module is loaded
_index = initialize_index()


if __name__ == "__main__":
    app.run(debug=True)

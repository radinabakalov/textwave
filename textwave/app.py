import os
import glob

import numpy as np
import faiss
from flask import Flask, request, jsonify

from textwave.modules.extraction.preprocessing import DocumentProcessing
from textwave.modules.extraction.embedding import Embedding
from textwave.modules.generator.question_answering import QAGeneratorMistral


app = Flask(__name__)

STORAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "storage")

CHUNKING_STRATEGY = 'sentence'
CHUNKING_PARAMETERS = {
    "num_sentences": 3,
    "overlap_size": 1,
}

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384 # dimension for all-MiniLM-L6-v2 -> avoids loading model just for dim

INDEX_STRATEGY = "bruteforce"
INDEX_PARAMETERS = {}

TOP_K = 5

# Global index and chunk store 
_index = None
_chunks = []


def initialize_index():
    """
    1. Parse through all documents in the storage directory.
    2. Chunk each document using the configured chunking strategy.
    3. Embed each chunk using the Embedding class.
    4. Store vector embeddings in a FAISS IndexFlatL2 (bruteforce).
    5. Return the FAISS index.
    """
    global _chunks

    processor = DocumentProcessing()
    embedder = Embedding(EMBEDDING_MODEL)

    all_chunks = []

    # Find every .txt file recursively under storage/
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

    # If storage is empty, return a valid but empty FAISS index
    # Use the known embedding dimension so we don't need to load the model just for this
    if not all_chunks:
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        _chunks = []
        return index

    # Embed all chunks and build the index
    embeddings = np.array([embedder.encode(chunk) for chunk in all_chunks], dtype=np.float32)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    _chunks = all_chunks

    return index


def _get_index():
    """Return the global index, building it on first call (lazy init)."""
    global _index
    if _index is None:
        _index = initialize_index()
    return _index


def _search_index(index, query: str, k: int) -> list:
    """Embed the query and return the top-k most relevant chunks."""
    embedder = Embedding(EMBEDDING_MODEL)
    query_vec = np.array([embedder.encode(query)], dtype=np.float32)

    # Cap k at the number of indexed chunks to avoid FAISS errors on empty index
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
    Retrieves top-k relevant chunks from the FAISS index, then uses the
    Mistral API to generate a grounded answer.

    Returns:
        JSON with "query" and "answer" fields, or an error response.
    """
    # Require a JSON body 
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Request must include a JSON body."}), 400

    query = data.get("query", "")

    # Reject empty or whitespace-only queries
    if not query or not query.strip():
        return jsonify({"error": "Query must be a non-empty string."}), 422

    query = query.strip()

    index = _get_index()
    context = _search_index(index, query, k=TOP_K)

    # Generate an answer via Mistral
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    generator = QAGeneratorMistral(api_key=api_key)
    answer = generator.generate_answer(query=query, context=context)

    return jsonify({"query": query, "answer": answer})


if __name__ == "__main__":
    app.run(debug=True)

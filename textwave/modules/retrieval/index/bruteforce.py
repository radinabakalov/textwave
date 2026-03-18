import numpy as np
import faiss


class BruteForceIndex:
    """
    Exact nearest-neighbor search using FAISS IndexFlatL2.
    Computes the true closest vectors.
    This is the accuracy ceiling against which HNSW and LSH are compared.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.chunks: list[str] = []

    def add(self, embeddings: np.ndarray, chunks: list[str]):
        """Add embeddings and their corresponding text chunks to the index."""
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int) -> list[str]:
        """Return the top-k most similar chunks for a given query embedding."""
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        query = np.array([query_embedding], dtype=np.float32)
        _, indices = self.index.search(query, k)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def search_with_indices(self, query_embedding: np.ndarray, k: int):
        """Return (chunks, indices, distances) for the top-k results."""
        k = min(k, self.index.ntotal)
        if k == 0:
            return [], [], []
        query = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query, k)
        chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return chunks, indices[0].tolist(), distances[0].tolist()

    @property
    def ntotal(self):
        return self.index.ntotal

import numpy as np
import faiss


class HNSWIndex:
    """
    Approximate nearest-neighbor search using FAISS IndexHNSWFlat.

    HNSW builds a multi-layer graph where each node connects to its nearest neighbors. 
    Search traverses this graph greedily, making it much faster than brute force 
    at the cost of occasionally missing the true nearest neighbor.

    Parameters:
        M: number of connections per node (higher -> better recall & more memory)
        ef_construction: search depth during index build (higher -> better quality & slower build)
        ef_search: search depth during query (higher -> better recall & slower query)
    """

    def __init__(self, dim: int, M: int = 32, ef_construction: int = 200, ef_search: int = 64):
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, M)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
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

import os
import pickle

from ..utils.bow import BagOfWords
from sympy import vectorize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import torch
import numpy as np

# TODO: You will need to implement: 
#  - Reranker.cross_encoder_rerank()
#  - Reranker.tfidf_rerank()
#  - Reranker.bow_rerank()
#  - Reranker.sequential_rerank()


class Reranker:
    """
    Perform reranking of documents based on their relevance to a given query.

    Supports multiple reranking strategies:
    - Cross-encoder: Uses a transformer model to compute pairwise relevance.
    - TF-IDF: Uses term frequency-inverse document frequency with similarity metrics.
    - BoW: Uses term Bag-of-Words with similarity metrics.
    - Hybrid: Combines TF-IDF and cross-encoder scores.
    - Sequential: Applies TF-IDF first, then cross-encoder for refined reranking.
    """

    def __init__(self, type, cross_encoder_model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2', corpus_directory=''):
        """
        Initialize the Reranker with a specified reranking strategy and optional model and corpus.

        :param type: Type of reranking ('cross_encoder', 'tfidf', 'bow', 'hybrid', or 'sequential').
        :param cross_encoder_model_name: HuggingFace model name for the cross-encoder (default: cross-encoder/ms-marco-TinyBERT-L-2-v2).
            - For more information on the default cross encoder, see https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2
            - For more information on general cross encoders, see https://huggingface.co/cross-encoder
        :param corpus_directory: Directory containing .txt files for TF-IDF corpus (optional).
        """
        self.type = type
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)


    def rerank(self, query, context, distance_metric="cosine", seq_k1=None, seq_k2=None):
        """
        Dispatch the reranking process based on the initialized strategy.

        :param query: Input query string to evaluate relevance against.
        :param context: List of document strings to rerank.
        :param distance_metric: Distance metric used for TF-IDF reranking (default: "cosine").
        :param seq_k1: Number of top documents to select in the first phase (TF-IDF) of sequential rerank.
        :param seq_k2: Number of top documents to return from the second phase (cross-encoder) of sequential rerank.
        :return: Tuple of (ranked documents, ranked indices, corresponding scores).
        """
        if self.type == "cross_encoder":
            return self.cross_encoder_rerank(query, context)
        elif self.type == "tfidf":
            return self.tfidf_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "bow":
            return self.bow_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "hybrid":
            return self.hybrid_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "sequential":
            return self.sequential_rerank(query, context, seq_k1, seq_k2, distance_metric=distance_metric)

    def cross_encoder_rerank(self, query, context):
        """
        Rerank documents using a cross-encoder transformer model.

        Computes relevance scores for each document-query pair, sorts them in
        descending order of relevance, and returns the ranked results.

        NOTE: See https://huggingface.co/cross-encoder for more information on 
        implementing cross-encoder

        :param query: Query string.
        :param context: List of candidate document strings.
        :return: Tuple of (ranked documents, ranked indices, relevance scores).
        """
        query_document_pairs = [(query, doc) for doc in context]
        inputs = self.tokenizer(query_document_pairs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.cross_encoder_model(**inputs).logits
            relevance_scores = logits.squeeze().tolist()

        # If there's only one doc, squeeze gives a scalar -> wrap it in a list
        if isinstance(relevance_scores, float):
            relevance_scores = [relevance_scores]

        # Sort by score descending (most relevant first)
        ranked_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)
        ranked_docs = [context[i] for i in ranked_indices]
        ranked_scores = [relevance_scores[i] for i in ranked_indices]

        return ranked_docs, ranked_indices, ranked_scores

    def tfidf_rerank(self, query, context, distance_metric="cosine"):
        """
        Rerank documents using TF-IDF vectorization and distance-based similarity.

        Creates a TF-IDF matrix from the query and context, computes pairwise distances,
        and sorts documents by similarity (lower distance implies higher relevance).

        :param query: Query string.
        :param context: List of document strings.
        :param distance_metric: Distance function to use (e.g., 'cosine', 'euclidean').
        :return: Tuple of (ranked documents, indices, similarity scores).
        """
        # Fit vectorizer on query & all docs together so they share a vocabulary
        vectorizer = TfidfVectorizer()
        all_texts = [query] + context
        tfidf_matrix = vectorizer.fit_transform(all_texts).toarray()

        # Row 0 is the query 
        # Rows 1: are the documents
        query_vec = tfidf_matrix[0:1]
        doc_vecs = tfidf_matrix[1:]

        # Compute cosine distance from query to each doc
        distances = pairwise_distances(query_vec, doc_vecs, metric=distance_metric)[0]

        # Sort ascending (lowest distance means most similar to the query)
        ranked_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        ranked_docs = [context[i] for i in ranked_indices]
        ranked_scores = [distances[i] for i in ranked_indices]

        return ranked_docs, ranked_indices, ranked_scores

    def bow_rerank(self, query, context, distance_metric="cosine"):
        """
        Rerank documents using BoW vectorization and distance-based similarity.

        Creates a BoW matrix from the query and context, computes pairwise distances,
        and sorts documents by similarity (lower distance implies higher relevance).

        :param query: Query string.
        :param context: List of document strings.
        :param distance_metric: Distance function to use (e.g., 'cosine', 'euclidean').
        :return: Tuple of (ranked documents, indices, similarity scores).
        """
        # Fit BagOfWords on query & all docs so they share a vocabulary
        all_texts = [query] + context
        bow = BagOfWords()
        bow.fit(all_texts)

        # Transform each text into a BoW vector
        query_vec = bow.transform(query).reshape(1, -1)
        doc_vecs = np.array([bow.transform(doc) for doc in context])

        # Compute cosine distance from query to each doc
        distances = pairwise_distances(query_vec, doc_vecs, metric=distance_metric)[0]

        # Sort ascending (lowest distance means most similar to the query)
        ranked_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        ranked_docs = [context[i] for i in ranked_indices]
        ranked_scores = [distances[i] for i in ranked_indices]

        return ranked_docs, ranked_indices, ranked_scores

    def hybrid_rerank(self, query, context, distance_metric="cosine", tfidf_weight=0.3):
        """
        Combine TF-IDF and cross-encoder scores to produce a hybrid reranking.

        This approach balances fast lexical matching (TF-IDF) with deeper semantic understanding
        (cross-encoder) by computing a weighted average of both scores.

        :param query: Query string.
        :param context: List of document strings.
        :param distance_metric: Distance metric for the TF-IDF portion.
        :param tfidf_weight: Weight (0-1) assigned to TF-IDF score in final ranking.
        :return: Tuple of (ranked documents, indices, combined scores).
        """
        # Get scores from both strategies
        _, tfidf_ranked_indices, tfidf_distances = self.tfidf_rerank(query, context, distance_metric=distance_metric)
        _, ce_ranked_indices, ce_scores = self.cross_encoder_rerank(query, context)

        # tfidf_rerank returns ascending distances
        # Convert to similarity scores and map back to original context order
        tfidf_sim_by_orig = [0.0] * len(context)
        for rank, orig_idx in enumerate(tfidf_ranked_indices):
            tfidf_sim_by_orig[orig_idx] = 1 - tfidf_distances[rank]

        # Cross_encoder_rerank returns descending scores -> map back to original order
        ce_by_orig = [0.0] * len(context)
        for rank, orig_idx in enumerate(ce_ranked_indices):
            ce_by_orig[orig_idx] = ce_scores[rank]

        # Normalize both to [0, 1] so they're on the same scale before combining
        def normalize(scores: list) -> list:
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [0.0] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        tfidf_norm = normalize(tfidf_sim_by_orig)
        ce_norm = normalize(ce_by_orig)

        # Weighted average
        combined = [
            tfidf_weight * t + (1 - tfidf_weight) * c
            for t, c in zip(tfidf_norm, ce_norm)
        ]

        # Sort descending by combined score
        ranked_indices = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)
        ranked_docs = [context[i] for i in ranked_indices]
        ranked_scores = [combined[i] for i in ranked_indices]

        return ranked_docs, ranked_indices, ranked_scores

    def sequential_rerank(self, query, context, seq_k1, seq_k2, distance_metric="cosine"):
        """
        Apply a two-stage reranking pipeline: TF-IDF followed by cross-encoder.

        This method narrows down the document pool using TF-IDF, then applies a
        cross-encoder to refine the top-k results for improved relevance accuracy.

        :param query: Query string.
        :param context: List of document strings.
        :param seq_k1: Top-k documents to retain after the first stage (TF-IDF).
        :param seq_k2: Final top-k documents to return after second stage (cross-encoder).
        :param distance_metric: Distance metric for TF-IDF.
        :return: Tuple of (ranked documents, indices, final relevance scores).
        """
        # Stage 1: Use TF-IDF to narrow down to the top seq_k1 docs
        _, tfidf_ranked_indices, _ = self.tfidf_rerank(query, context, distance_metric=distance_metric)
        top_k1_indices = tfidf_ranked_indices[:seq_k1]
        top_k1_docs = [context[i] for i in top_k1_indices]

        # Stage 2: Run cross-encoder on the smaller candidate set
        _, ce_local_indices, ce_scores = self.cross_encoder_rerank(query, top_k1_docs)

        # ce_local_indices index into top_k1_docs, so remap back to original context indices
        final_indices = [top_k1_indices[i] for i in ce_local_indices]
        final_docs = [context[i] for i in final_indices]

        # Trim to seq_k2 if needed
        if seq_k2 is not None:
            final_docs = final_docs[:seq_k2]
            final_indices = final_indices[:seq_k2]
            ce_scores = ce_scores[:seq_k2]

        return final_docs, final_indices, ce_scores


if __name__ == "__main__":
    query = "What are the health benefits of green tea?"
    documents = [
        "Green tea contains antioxidants that may help prevent cardiovascular disease.",
        "Coffee is also rich in antioxidants but can increase heart rate.",
        "Drinking water is essential for hydration.",
        "Green tea may also aid in weight loss and improve brain function."
    ]

    print("\nTF-IDF Reranking:")
    reranker = Reranker(type="tfidf")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nCross-Encoder Reranking:")
    reranker = Reranker(type="cross_encoder")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nHybrid Reranking:")
    reranker = Reranker(type="hybrid")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

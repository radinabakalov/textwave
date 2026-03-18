"""
eval_utils.py is the shared utilities file for TextWave analysis tasks.
All five task notebooks import from here.
"""

import os
import sys
import glob
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# Make sure the textwave package is importable from the analysis/ directory
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from textwave.modules.extraction.preprocessing import DocumentProcessing
from textwave.modules.extraction.embedding import Embedding

STORAGE_DIR = REPO_ROOT / "textwave" / "storage"
QA_TSV = REPO_ROOT / "textwave" / "qa_resources" / "question.tsv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Data loading
def load_questions(deduplicate: bool = True) -> pd.DataFrame:
    """
    Load the QA TSV, drop rows with missing Question or ArticleFile, and optionally deduplicate by (Question, ArticleFile).
    """
    df = pd.read_csv(QA_TSV, sep="\t")
    df = df.dropna(subset=["Question", "ArticleFile"]).reset_index(drop=True)
    if deduplicate:
        df = df.drop_duplicates(subset=["Question", "ArticleFile"]).reset_index(drop=True)
    return df


def get_corpus_files() -> list:
    """Return sorted list of all .txt.clean files in the storage directory."""
    return sorted(glob.glob(str(STORAGE_DIR / "*.txt.clean")))


# Chunking & embedding
def build_chunks(
    strategy: str = "sentence",
    num_sentences: int = 3,
    overlap_size: int = 1,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> Tuple[list, list]:
    """
    Chunk all corpus files using the given strategy.

    Returns:
        chunks: list of text strings
        source_files: parallel list mapping each chunk to its source filename (stem only)
    """
    processor = DocumentProcessing()
    chunks = []
    source_files = []

    for filepath in get_corpus_files():
        stem = Path(filepath).name.replace(".txt.clean", "")
        if strategy == "sentence":
            file_chunks = processor.sentence_chunking(
                filepath, num_sentences=num_sentences, overlap_size=overlap_size
            )
        else:
            file_chunks = processor.fixed_length_chunking(
                filepath, chunk_size=chunk_size, overlap_size=chunk_overlap
            )
        chunks.extend(file_chunks)
        source_files.extend([stem] * len(file_chunks))

    return chunks, source_files


def embed_chunks(chunks: list, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed all chunks and return as a float32 numpy array."""
    embedder = Embedding(model_name)
    return np.array([embedder.encode(c) for c in chunks], dtype=np.float32)


def precompute_query_embeddings(
    questions_df: pd.DataFrame,
    save_path: str = "query_embeddings.npy",
    model_name: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Embed all questions and save to disk.
    """
    embedder = Embedding(model_name)
    vecs = np.array(
        [embedder.encode(q) for q in questions_df["Question"].tolist()],
        dtype=np.float32,
    )
    np.save(save_path, vecs)
    print(f"Saved {len(vecs)} query embeddings to {save_path}!")
    return vecs


# Retrieval evaluation
def retrieval_metrics(
    retrieved_source_files: list,
    relevant_source_file: str,
    k: int,
) -> dict:
    """
    Compute precision@k, recall@k, F1@k, and hit@k.
    A chunk is relevant if its source file matches the question's ArticleFile.
    Since each question has exactly one relevant document, recall is 0 or 1.
    """
    relevant = relevant_source_file.strip()
    retrieved = [s.strip() for s in retrieved_source_files[:k]]

    hits = sum(1 for s in retrieved if s == relevant)
    precision_at_k = hits / k if k > 0 else 0.0
    recall_at_k = 1.0 if hits > 0 else 0.0
    f1 = (
        2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k)
        if (precision_at_k + recall_at_k) > 0
        else 0.0
    )
    return {
        "precision@k": precision_at_k,
        "recall@k": recall_at_k,
        "f1@k": f1,
        "hit": recall_at_k,
    }


def evaluate_retrieval(
    questions_df: pd.DataFrame,
    index,
    source_files: list,
    embedder_or_vecs,
    k: int = 5,
) -> pd.DataFrame:
    """
    Run retrieval for every question and return a results dataframe.

    embedder_or_vecs can be either:
    - an Embedding instance (will encode each question on the fly)
    - a numpy array of precomputed query vectors (shape: n_questions x dim)
    """
    use_precomputed = isinstance(embedder_or_vecs, np.ndarray)
    rows = []

    for i, (_, row) in enumerate(questions_df.iterrows()):
        if use_precomputed:
            query_vec = embedder_or_vecs[i]
        else:
            query_vec = embedder_or_vecs.encode(row["Question"])

        retrieved_chunks, indices, _ = index.search_with_indices(query_vec, k)
        retrieved_sources = [source_files[j] for j in indices if j < len(source_files)]
        metrics = retrieval_metrics(retrieved_sources, row["ArticleFile"], k)
        metrics["Question"] = row["Question"]
        metrics["ArticleFile"] = row["ArticleFile"]
        metrics["difficulty"] = row.get("DifficultyFromAnswerer", "unknown")
        rows.append(metrics)

    return pd.DataFrame(rows)


# Generation evaluation
def run_generation_experiment(
    questions_df: pd.DataFrame,
    generator,
    context_fn=None,
    sleep_sec: float = 1.0,
) -> pd.DataFrame:
    """
    Run generation for every question and return scored results.

    Args:
        questions_df: must have Question, Answer, ArticleFile, DifficultyFromAnswerer
        generator: object with generate_answer(query, context) method
        context_fn: callable(question) -> list[str] of context chunks; None for no-RAG
        sleep_sec: delay between calls to avoid rate limiting
    """
    from textwave.modules.utils.metrics import Matching
    matcher = Matching()
    rows = []

    grouped = questions_df.groupby("Question")

    for question, group in grouped:
        references = group["Answer"].tolist()
        difficulty = group["DifficultyFromAnswerer"].iloc[0]
        article_file = group["ArticleFile"].iloc[0]

        context = context_fn(question) if context_fn else []

        try:
            answer = generator.generate_answer(query=question, context=context)
            time.sleep(sleep_sec)

            em = matcher.exact_match(answer, references[0])
            scores, tm = matcher.transformer_match(answer, references[0], question)
            # scores can come back as a dict or something unexpected -- handle safely
            try:
                t_score = float(max(scores.values())) if scores else 0.0
            except (TypeError, ValueError, AttributeError):
                t_score = 0.0
        except Exception as e:
            answer = ""
            em, tm, t_score = 0, 0, 0.0
            print(f"Error on: {question[:60]}... -> {e}")

        rows.append({
            "question": question,
            "generated_answer": answer,
            "references": references,
            "difficulty": difficulty,
            "article_file": article_file,
            "exact_match": int(em),
            "transformer_match": int(tm),
            "transformer_score": t_score if isinstance(t_score, float) else 0.0,
        })

    return pd.DataFrame(rows)


# Reporting helpers
def summarize_results(df: pd.DataFrame, score_col: str = "transformer_match") -> pd.DataFrame:
    """Return overall & difficulty-stratified summary, filtering to easy/medium/hard."""
    valid = df[df["difficulty"].isin(["easy", "medium", "hard"])]
    overall = df[score_col].mean()
    by_diff = valid.groupby("difficulty")[score_col].mean()
    summary = pd.DataFrame({
        "difficulty": ["overall"] + by_diff.index.tolist(),
        score_col: [overall] + by_diff.tolist(),
    })
    return summary


def print_summary(label: str, df: pd.DataFrame, score_col: str = "transformer_match"):
    """Pretty-print a results summary."""
    summary = summarize_results(df, score_col)
    print(summary.to_string(index=False))
    return summary

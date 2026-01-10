"""assignment-8-test/unit-test.py

Unit tests for Assignment 8

Run with:
    cd assignment-8-test/
    PYTHONPATH="../textwave" python -m unittest unit-test.py
"""

from __future__ import annotations

import math
import unittest
from copy import deepcopy
from collections import Counter

from textwave.modules.utils.bow import BagOfWords 
from textwave.modules.utils.tfidf import TF_IDF
from textwave.modules.extraction.preprocessing import DocumentProcessing

import numpy as np

class _MockedDocumentProcessing(DocumentProcessing):
    """Subclass that injects raw text — bypassing file I/O."""

    def __init__(self, text: str):
        super().__init__()
        self._mock_text = text

    # Monkey-patch the private reader to return in-memory text
    def _DocumentProcessing__read_text_file(self, _path: str) -> str:  # type: ignore
        return self._mock_text


class _SpyDocumentProcessing(_MockedDocumentProcessing):
    """Extends the mock to record whether ``trim_white_space`` gets called."""

    def __init__(self, text: str):
        super().__init__(text)
        self.trim_called: bool = False

    def trim_white_space(self, text: str) -> str:  # type: ignore[override]
        self.trim_called = True
        return super().trim_white_space(text)


# ---------------------------------------------------------------------------
# Test Case
# ---------------------------------------------------------------------------

class TestFixedLengthChunking(unittest.TestCase):
    """Edge-case and functional tests for fixed-length chunking."""

    # ---------------------------------------------
    # Helpers
    # ---------------------------------------------

    def _run_chunker(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Utility to call the chunker without touching the filesystem."""
        dp = _MockedDocumentProcessing(text)
        return dp.fixed_length_chunking("dummy.txt", chunk_size, overlap)

    # ---------------------------------------------
    # Core Behaviour
    # ---------------------------------------------

    def test_basic_fixed_length(self) -> None:
        """Splits text into equal-sized chunks with correct overlap."""
        text = "abcdefghijklmnopqrstuvwxyz"  # len = 26
        chunk_size, overlap = 10, 2
        chunks = self._run_chunker(text, chunk_size, overlap)

        # Each chunk size should be <= chunk_size and > 0
        for ch in chunks:
            self.assertLessEqual(len(ch), chunk_size)
            self.assertGreater(len(ch), 0)

        # Check step consistency: next chunk starts with previous tail of `overlap` chars
        for i in range(1, len(chunks)):
            prev, cur = chunks[i - 1], chunks[i]
            self.assertEqual(prev[-overlap:], cur[:overlap])

        # Reconstruct the original text by stitching chunks minus overlaps
        reconstructed = chunks[0]
        for nxt in chunks[1:]:
            reconstructed += nxt[overlap:]
        self.assertEqual(reconstructed, text)

    def test_chunk_size_larger_than_text(self) -> None:
        """Returns a single chunk when chunk_size exceeds text length."""
        text = "short"
        chunk_size, overlap = 100, 2
        chunks = self._run_chunker(text, chunk_size, overlap)
        self.assertEqual(chunks, [text])

    def test_overlap_zero(self) -> None:
        """Overlap of zero should produce non-overlapping consecutive chunks."""
        text = "0123456789" * 3  # len = 30
        chunk_size, overlap = 7, 0
        chunks = self._run_chunker(text, chunk_size, overlap)

        # Concatenating all chunks should reproduce original text
        self.assertEqual("".join(chunks), text)

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty input yields an empty list of chunks."""
        chunks = self._run_chunker("", chunk_size=5, overlap=2)
        self.assertEqual(chunks, [])

    def test_invalid_overlap_raises(self) -> None:
        """Overlap >= chunk_size should raise ValueError (implementation expectation)."""
        text = "1234567890"
        with self.assertRaises(ValueError):
            _ = self._run_chunker(text, chunk_size=5, overlap=5)
        with self.assertRaises(ValueError):
            _ = self._run_chunker(text, chunk_size=5, overlap=6)

    def test_trim_white_space_invoked(self) -> None:
        """`fixed_length_chunking` must call `trim_white_space` internally."""
        raw_text = "   spaced   out   text   "
        spy = _SpyDocumentProcessing(raw_text)
        _ = spy.fixed_length_chunking("dummy.txt", chunk_size=5, overlap_size=2)  # type: ignore[arg-type]
        self.assertTrue(spy.trim_called, "trim_white_space() was not invoked inside fixed_length_chunking.")

    # ---------------------------------------------
    # Accuracy example
    # ---------------------------------------------

    def test_fixed_length_accuracy_expected_output(self) -> None:
        """Concrete example to verify exact chunk boundaries and overlap."""
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # len = 26
        chunk_size, overlap = 10, 2
        chunks = self._run_chunker(text, chunk_size, overlap)
        expected = [
            "ABCDEFGHIJ",  # first 10 chars
            "IJKLMNOPQR",  # overlaps IJ
            "QRSTUVWXYZ",  # last 8 chars
        ]
        self.assertEqual(chunks, expected)



class TestBagOfWords(unittest.TestCase):
    """Edge-case and integration tests for BagOfWords."""

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _build_simple_model(self) -> BagOfWords:
        """Returns a small model trained on a simple corpus."""
        corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "Never jump over the lazy dog quickly in the evening under bright stars.",
            "A quick movement of the enemy will jeopardize six gunboats and confuse the guards.",
            "All that glitters is not gold but sometimes it fools many unsuspecting travelers.",
            "To be or not to be, that remains the eternal question pondered by countless sages across ages.",
        ]
        return BagOfWords().fit(corpus)

    # ------------------------------------------------------------------
    # Tokenization Tests
    # ------------------------------------------------------------------

    def test_tokenize_basic(self) -> None:
        """Basic lowercase extraction; order of tokens is irrelevant."""
        bow = BagOfWords()
        tokens = bow._tokenize("Hello, HELLO World!")
        self.assertCountEqual(tokens, ["hello", "hello", "world"])

    def test_tokenize_edge_cases(self) -> None:
        """Handles punctuation, numbers, and Unicode accents."""
        bow = BagOfWords()
        text = "C3PO's café — déjà vu? #2025"
        tokens = bow._tokenize(text)
        expected = ["c3po", "café", "déjà", "vu", "2025"] 
        self.assertCountEqual(tokens, expected)

    # ------------------------------------------------------------------
    # Vocabulary / Fit Tests
    # ------------------------------------------------------------------

    def test_fit_builds_unique_vocabulary(self) -> None:
        """Vocabulary should contain unique tokens with unique indices."""
        corpus = [
            "alpha beta beta gamma",
            "gamma delta",
        ]
        bow = BagOfWords().fit(corpus)
        # Tokens should be unique
        self.assertEqual(len(bow.vocabulary_), len(set(bow.vocabulary_.keys())))
        # Indices should be unique
        self.assertEqual(len(bow.vocabulary_), len(set(bow.vocabulary_.values())))

    # ------------------------------------------------------------------
    # Transform Tests (Simple & Complex)
    # ------------------------------------------------------------------

    def test_transform_counts_correctly(self) -> None:
        """Term counts must match expected values for a simple doc."""
        corpus = ["foo bar bar"]
        bow = BagOfWords().fit(corpus)
        vec = bow.transform("foo bar bar bar baz")
        
        self.assertAlmostEqual(vec[0], 0.948, places=2)
        self.assertAlmostEqual(vec[1], 0.316, places=2)
        # baz is OOV and should not appear
        self.assertNotIn("baz", bow.vocabulary_)

    def test_transform_complex_document(self) -> None:
        """Checks counts after fitting on a richer five-sentence corpus."""
        bow = self._build_simple_model()
        doc = "The quick dog jumps high over the lazy fox near gold."
        vec = bow.transform(doc)

        self.assertAlmostEqual(vec[6], 0.40824, places=2)
        self.assertAlmostEqual(vec[11], 0.40824, places=2)
        self.assertAlmostEqual(vec[13], 0.40824, places=2)
        self.assertAlmostEqual(vec[17], 0.40824, places=2)
        self.assertAlmostEqual(vec[18], 0.40824, places=2)
        self.assertAlmostEqual(vec.sum(), 2.44948, places=3)
        # 'mine' is OOV and should not appear
        self.assertNotIn("mine", vec)

    def test_transform_empty_document(self) -> None:
        """Empty string yields empty vector."""
        bow = BagOfWords().fit(["some content here"])
        self.assertEqual(bow.transform("").tolist(), [0,])

    def test_transform_only_oov_tokens(self) -> None:
        """Document with only OOV tokens returns empty vector."""
        bow = BagOfWords().fit(["known words only"])
        self.assertEqual(bow.transform("unknown tokens elsewhere").tolist(), [0, 0])

    def test_round_trip_corpus_counts(self) -> None:
        """Transforming training docs should recover original term counts."""
        corpus = [
            "alpha alpha beta",
            "beta gamma",
        ]
        bow = BagOfWords().fit(corpus)
        expected_doc0 = [0.894, 0.4472, 0.0]
        expected_doc1 = [0.0, 0.707, 0.707]

        for a, b in zip(bow.transform(corpus[0]).tolist(), expected_doc0):
            self.assertAlmostEqual(a, b, places=1)

        for a, b in zip(bow.transform(corpus[1]).tolist(), expected_doc1):    
            self.assertAlmostEqual(a, b, places=1)


    # ------------------------------------------------------------------
    # Robustness / Refit Tests
    # ------------------------------------------------------------------

    def test_refit_resets_vocabulary(self) -> None:
        """Calling ``fit`` a second time should overwrite previous vocab."""
        bow = BagOfWords().fit(["one two"])
        first_vocab = bow.vocabulary_.copy()
        bow.fit(["three four"])
        self.assertNotEqual(first_vocab, bow.vocabulary_)
        self.assertIn("three", bow.vocabulary_)
        self.assertNotIn("one", bow.vocabulary_)

    def test_transform_before_fit_raises(self) -> None:
        """Ensure transform raises AttributeError if called before fit."""
        bow = BagOfWords()
        with self.assertRaises(AttributeError):
            _ = bow.transform("some text")


    


class TestTFIDF(unittest.TestCase):
    """Edge-case and integration tests for TF_IDF."""

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _build_simple_model(self) -> TF_IDF:
        """Returns a small model trained on a richer, 5-sentence corpus."""
        corpus = [
            "The quick brown fox jumps over the lazy dog",
            "The lazy dog likes to sleep all day",
            "The brown fox prefers to eat cheese",
            "The red fox jumps over the brown fox",
        ]
        return TF_IDF().fit(corpus)

    # ------------------------------------------------------------------
    # Tokenization Tests
    # ------------------------------------------------------------------

    def test_tokenize_basic_2(self) -> None:
        """Basic lowercase extraction; order is irrelevant."""
        tfidf = TF_IDF()
        tokens = tfidf._tokenize("Hello, HELLO World!")
        self.assertCountEqual(tokens, ["hello", "hello", "world"])

    def test_tokenize_edge_cases_2(self) -> None:
        """Handles punctuation, numbers, and Unicode accents."""
        tfidf = TF_IDF()
        text = "C3PO's café — déjà vu? #2025"
        tokens = tfidf._tokenize(text)
        expected = ["c3po", "café", "déjà", "vu", "2025"]
        self.assertCountEqual(tokens, expected)

    # ------------------------------------------------------------------
    # Fit / Vocabulary / IDF Tests
    # ------------------------------------------------------------------

    def test_fit_builds_unique_vocab_and_idf(self) -> None:
        """Vocabulary is unique; IDF vector aligns with it and rewards rarer terms."""
        corpus = [
            "alpha beta beta gamma",
            "gamma delta",
        ]
        tfidf = TF_IDF().fit(corpus)

        # ---- vocabulary uniqueness -------------------------------------------------
        self.assertEqual(len(tfidf.vocabulary_), len(set(tfidf.vocabulary_.keys())))
        self.assertEqual(len(tfidf.vocabulary_), len(set(tfidf.vocabulary_.values())))

        # ---- IDF vector length matches vocab size ----------------------------------
        self.assertEqual(len(tfidf.idf_), len(tfidf.vocabulary_))

        # ---- rarer term gets higher IDF --------------------------------------------
        delta_idx = tfidf.vocabulary_["delta"]
        gamma_idx = tfidf.vocabulary_["gamma"]
        self.assertGreater(tfidf.idf_[delta_idx], tfidf.idf_[gamma_idx])

    # ------------------------------------------------------------------
    # Transform Tests (Simple & Complex)
    # ------------------------------------------------------------------

    def test_transform_tfidf_values_small_corpus(self) -> None:
        """
        Checks TF-IDF values on a toy corpus with closed-form expectations.
        Expectations are computed from the model’s IDF vector so they remain
        correct even if the smoothing formula changes.
        """
        tfidf = self._build_simple_model()          # builds + fits a TF_IDF()
        doc = "The brown dog chases the fox"
        vec = tfidf.transform(doc)                  # ndarray (|V|,)

        # Helper to compute the expected weight for any token in *doc*
        def expected_weight(token: str, tf: int = .33) -> float:
            idx = tfidf.vocabulary_[token]
            return tf * tfidf.idf_[idx]

        expected = {
            "brown": expected_weight("brown"),
            "dog":   expected_weight("dog"),
            "fox": expected_weight("fox"),
        }

        for token, exp_val in expected.items():
            # Token must be in the learned vocabulary
            self.assertIn(token, tfidf.vocabulary_)

            # Same value at its vector position
            idx = tfidf.vocabulary_[token]
            self.assertAlmostEqual(vec[idx], exp_val, places=2)


    def test_transform_complex_document_2(self) -> None:
        """
        Checks that TF-IDF vector has positive scores for in-vocabulary terms
        and zero for an OOV term.
        """
        tfidf = self._build_simple_model()   # helper builds & fits a TF_IDF instance
        doc = (
            "The quick dog jumps high over the lazy fox near gold "
            "and history repeats itself."
        )

        vec = tfidf.transform(doc)           # ndarray

        # In-vocabulary words should get a positive score
        for token in ["quick", "dog", "lazy", "fox"]:
            self.assertIn(token, tfidf.vocabulary_)
            idx = tfidf.vocabulary_[token]
            self.assertGreater(vec[idx], 0.0)

        # Out-of-vocabulary token should have zero weight
        self.assertNotIn("spaceship", tfidf.vocabulary_)

    def test_transform_empty_document_2(self) -> None:
        """Empty string yields empty vector."""
        tfidf = TF_IDF().fit(["some content here"])
        self.assertEqual(tfidf.transform("").tolist(), [0,])

    def test_transform_only_oov_tokens_2(self) -> None:
        """Document with only OOV tokens returns empty vector."""
        tfidf = TF_IDF().fit(["known words only"])
        self.assertEqual(tfidf.transform("unknown tokens elsewhere").tolist(), [0., 0.])

    def test_round_trip_corpus_vectors_nonzero(self) -> None:
        corpus = ["a a b", "b c"]
        tfidf = TF_IDF().fit(corpus)

        for doc in corpus:
            vec = tfidf.transform(doc)     # vec is an np.ndarray
            self.assertEqual(vec.ndim, 1)  # sanity: 1-D vector
            self.assertGreater(vec.size, 0)

            # At least one element should be > 0 (otherwise the vector is “empty”)
            self.assertGreater(vec[vec > 0].size, 0)

            # Every *non-zero* entry must be strictly positive
            for score in vec[vec > 0]:
                self.assertGreater(score, 0.0)

    # ------------------------------------------------------------------
    # Robustness / Refit Tests
    # ------------------------------------------------------------------

    def test_refit_resets_vocabulary_and_idf(self) -> None:
        """Calling ``fit`` a second time should overwrite previous vocab and IDF."""
        tfidf = TF_IDF().fit(["one two"])
        first_vocab = tfidf.vocabulary_.copy()
        tfidf.fit(["three four five"])
        self.assertNotEqual(first_vocab, tfidf.vocabulary_)
        self.assertIn("three", tfidf.vocabulary_)
        self.assertNotIn("one", tfidf.vocabulary_)
        # IDF mapping should be updated accordingly
        self.assertEqual(len(tfidf.idf_.tolist()), 3)

    def test_transform_before_fit_raises_2(self) -> None:
        """Ensure transform raises AttributeError if called before fit."""
        tfidf = TF_IDF()
        with self.assertRaises(AttributeError):
            _ = tfidf.transform("some text")

if __name__ == "__main__":
    unittest.main(verbosity=2)

import math
import re
from collections import Counter, defaultdict
from typing import List

import nltk
import numpy as np

nltk.download('stopwords', quiet=True)


class TF_IDF:
    """
    A TF-IDF transformer that learns a vocabulary from a corpus and transforms
    documents into their TF-IDF representation.
    
    The TF-IDF (Term Frequency - Inverse Document Frequency) model is a numerical
    statistic intended to reflect how important a word is to a document in a corpus.
    This transformer builds a vocabulary of unique tokens from the provided corpus and
    computes an IDF score for each token. New documents can then be transformed into
    a dictionary where each token is associated with its TF-IDF score.
    
    Attributes:
        vocabulary_ (dict): A mapping of tokens (str) to unique indices (int), created
                            during the fitting process.
        idf_ (dict): A mapping of tokens to their computed inverse document frequency values.
    """

    def __init__(self):
        """
        Initializes the TF_IDF transformer with an empty vocabulary and IDF mapping.
        
        This constructor sets up the transformer without any pre-loaded vocabulary.
        The vocabulary and IDF values will be computed when the 'fit' method is called.
        """
        # Intentionally not initializing vocabulary or idf here so that
        # calling transform() before fit() raises AttributeError
        pass

    def _tokenize(self, documents: List[str]):
        """
        Tokenizes the input text by converting it to lowercase and extracting words.
        
        The function uses a regular expression to match word boundaries and extract
        alphanumeric sequences as tokens. This is a basic tokenization approach that
        may be extended for more complex use cases.

        NOTE: We should exclude stop words!
        
        Parameters:
            text (str): The text to tokenize.
            
        Returns:
            list: A list of word tokens (str) extracted from the input text.
            
        Example:
            >>> tokens = TF_IDF()._tokenize("Hello World!")
            >>> print(tokens)
            ['hello', 'world']
        """
        try:
            stop_words = set(nltk.corpus.stopwords.words('english'))
        except LookupError:
            # Fallback if nltk stopwords aren't downloaded
            stop_words = {'i','me','my','we','our','you','your','he','him','his','she','her',
                          'it','its','they','them','their','what','which','who','this','that',
                          'these','those','am','is','are','was','were','be','been','have','has',
                          'had','do','does','did','a','an','the','and','but','if','or','of',
                          'at','by','for','with','to','from','in','out','on','off','over',
                          'then','so','no','not','only','some','such','too','very','here','there',
                          'all','both','each','few','more','most','other','same','than','s','t'}

        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter stop words and single-character tokens
        return [t for t in tokens if t not in stop_words and len(t) > 1]

    def fit(self, document: str):
        """
        Learns the vocabulary and computes the inverse document frequency (IDF) from the corpus.
        
        The 'fit' method processes each document in the provided corpus, tokenizes them,
        and constructs a set of unique tokens. It then calculates the document frequency
        for each token (i.e., the number of documents that contain the token). The IDF for
        each token is computed using the formula:
        
            IDF(token) = log(total_documents / (document_frequency + 1)) + 1
        
        The vocabulary is stored as a mapping from token to index, and the IDF values
        are stored in a separate dictionary.
        
        Parameters:
            documents (list of str): A list of documents (each document is a string)
                                     that forms the training corpus.
                                     
        Returns:
            TF_IDF (object): The instance of the TF_IDF transformer with the learned vocabulary and IDF values.
            In other words, your function should end in: `return self`.
            
        Example:
            >>> corpus = ["The quick brown fox.", "Lazy dog."]
            >>> transformer = TF_IDF().fit(corpus)
        """
        documents = document
        n = len(documents)

        # Collect all unique tokens across the corpus
        all_tokens = set()
        for doc in documents:
            all_tokens.update(self._tokenize(doc))

        # Sort for consistent ordering, then assign each token an index
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(all_tokens))}

        # Count how many documents contain each token (document frequency)
        doc_freq = Counter()
        for doc in documents:
            # Use a set so each token counts once per document, not once per occurrence
            unique_tokens = set(self._tokenize(doc))
            for token in unique_tokens:
                if token in self.vocabulary_:
                    doc_freq[token] += 1

        # Store idf as a numpy array indexed by vocabulary position
        # Formula: log(N / (df + 1)) + 1 -> the +1 smooths zero-frequency terms
        idf_array = np.zeros(len(self.vocabulary_))
        for token, idx in self.vocabulary_.items():
            idf_array[idx] = math.log(n / (doc_freq[token] + 1)) + 1
        self.idf_ = idf_array

        return self

    def transform(self, document):
        """
        Transforms a document into its TF-IDF representation.
        
        This method tokenizes the input document and computes the term frequency (TF) for each token.
        The TF is normalized by dividing the token count by the total number of tokens in the document.
        Each token's TF value is then multiplied by its corresponding IDF value (learned during 'fit')
        to obtain the TF-IDF score. Only tokens present in the learned vocabulary are included.
        
        Parameters:
            document (str): A single document (string) to be transformed.
            
        Returns:
            numpy.array: A numpy array indexing each term (from the learned vocabulary) with its TF-IDF scores in the document.
                  Only tokens present in the vocabulary are included.
            
        Example:
            >>> transformer = TF_IDF().fit(["The quick brown fox.", "Lazy dog."])
            >>> tfidf_vector = transformer.transform("The quick fox.")
        """
        tokens = self._tokenize(document)

        # Only count tokens that are in the vocabulary
        in_vocab = [t for t in tokens if t in self.vocabulary_]
        total = len(in_vocab)

        # Return zero vector
        if total == 0:
            return np.zeros(len(self.vocabulary_))

        counts = Counter(in_vocab)
        vector = np.zeros(len(self.vocabulary_))

        for token, count in counts.items():
            # TF normalized by in-vocab token count
            tf = count / total
            idx = self.vocabulary_[token]
            vector[idx] = tf * self.idf_[idx]

        return vector



if __name__ == "__main__":
    # Example corpus of 9 documents to fit the TF-IDF transformer.
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Never jump over the lazy dog quickly.",
        "A quick movement of the enemy will jeopardize six gunboats.",
        "All that glitters is not gold.",
        "To be or not to be, that is the question.",
        "I think, therefore I am.",
        "The only thing we have to fear is fear itself.",
        "Ask not what your country can do for you; ask what you can do for your country.",
        "That's one small step for man, one giant leap for mankind.",
    ]

    # Fit the transformer on the corpus.
    transformer = TF_IDF()
    transformer.fit(corpus)
    
    # Test document to transform after fitting the corpus.
    test_document = "The quick dog jumps high over the lazy fox."
    tfidf_test = transformer.transform(test_document)
    
    print(tfidf_test)

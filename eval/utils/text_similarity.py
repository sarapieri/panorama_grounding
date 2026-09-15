# Ours: phrase-level text similarity (exact match, WordNet synonyms, SBERT cosine) used by the
# PanoCaps metrics to match predicted and ground-truth phrases.
import re
from functools import lru_cache
from typing import List, Union, Iterable, Optional

import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

Label = Union[str, List[str]]


class TextSimilarityMetric:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.sbert_model = SentenceTransformer(model_name)
        self.lemmatizer = WordNetLemmatizer()

    # normalization & helpers
    def remove_leading_articles(self, text: str) -> str:
        text = re.sub(r'^(a|an|the)\s+', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'-\d+$', '', text)
        return text.strip()

    def _as_list(self, x: Optional[Label]) -> List[str]:
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        # assume iterable of strings
        return [s for s in x if isinstance(s, str)]

    @lru_cache(maxsize=10000)
    def _norm_str(self, s: str) -> str:
        return self.remove_leading_articles(s.lower().strip())

    # WordNet
    @lru_cache(maxsize=10000)
    def get_synonyms(self, word: str) -> frozenset:
        """Retrieve synonyms of a single token using WordNet with lemmatization.

        Falls back to the word itself if the WordNet data is missing.
        """
        try:
            word = self.lemmatizer.lemmatize(word.lower().strip())
            synsets = wordnet.synsets(word)
            synonyms = set()
            for synset in synsets:
                for lemma in synset.lemmas():
                    synonyms.add(self.lemmatizer.lemmatize(lemma.name().replace('_', ' ')))
            synonyms.add(word)
            return frozenset(synonyms)
        except LookupError:
            return frozenset({word.lower().strip()})

    # SBERT embeddings
    @lru_cache(maxsize=10000)
    def _embed(self, normalized_text: str):
        """
        Cached embedding for a normalized string.
        We pass in *normalized* text to ensure cache hits.
        """
        return self.sbert_model.encode(normalized_text, normalize_embeddings=True)

    # mention-level similarity (string vs string)
    def compute_similarity(self, label1: str, label2: str) -> float:
        """
        Similarity between two single strings (mentions).
        Uses exact match -> 1.0, WordNet synonym match (if single-token) -> 1.0,
        fallback to SBERT cosine similarity.
        """
        label1 = self._norm_str(label1)
        label2 = self._norm_str(label2)

        if label1 == label2:
            return 1.0

        # WordNet
        if " " not in label1 and " " not in label2:
            if self.get_synonyms(label1).intersection(self.get_synonyms(label2)):
                return 1.0

        e1 = self._embed(label1)
        e2 = self._embed(label2)
        return float(cos_sim(e1, e2).item())

    # entity-level similarity
    def compute_entity_similarity(self, mentions_a: Label, mentions_b: Label) -> float:
        """
        Count-invariant: returns the maximum similarity between any mention pair.
        If either side is empty, returns 0.0.
        """
        A = [self._norm_str(x) for x in self._as_list(mentions_a) if x and x.strip()]
        B = [self._norm_str(x) for x in self._as_list(mentions_b) if x and x.strip()]

        if not A or not B:
            return 0.0

        # any exact normalized string match -> 1.0
        if any(a == b for a in A for b in B):
            return 1.0

        best = 0.0
        for a in A:
            for b in B:
                sim = self.compute_similarity(a, b)  # mention-level
                if sim > best:
                    best = sim
                    if best >= 1.0:
                        return 1.0
        return float(best)

    # matrices & pretty print
    def compute_similarity_matrix(self, gt_labels: Iterable[Label], dt_labels: Iterable[Label]) -> np.ndarray:
        """
        Accepts lists where each element is either a string (single mention)
        or a list of strings (multiple mentions for that entity).
        Returns entity-level similarity matrix using max pairwise aggregation.
        """
        gt_labels = list(gt_labels)
        dt_labels = list(dt_labels)

        text_sims = np.zeros((len(gt_labels), len(dt_labels)), dtype=float)
        for i, gt in enumerate(gt_labels):
            for j, dt in enumerate(dt_labels):
                text_sims[i, j] = self.compute_entity_similarity(gt, dt)
        return text_sims

    def pretty_print_similarity_matrix(
        self,
        sim_matrix: np.ndarray,
        gt_labels: Iterable[Label],
        dt_labels: Iterable[Label],
        ) -> None:
        """
        Pretty print; join multi-mentions with ' | ' for readability.
        """
        def disp(x: Label) -> str:
            parts = self._as_list(x)
            parts = [self.remove_leading_articles(str(s)) for s in parts]
            if not parts:
                return ""
            return " | ".join(parts)

        gt_disp = [disp(x) for x in gt_labels]
        dt_disp = [disp(x) for x in dt_labels]

        print("\n[SIM] Similarity matrix:")
        header = "\t" + "\t".join(dt_disp)
        print("[SIM] " + header)
        print("[SIM] " + "-" * max(8, len(header)))
        for name, row in zip(gt_disp, sim_matrix):
            row_str = "\t".join(f"{sim:.2f}" for sim in row)
            print("[SIM] " + f"{name}\t{row_str}")

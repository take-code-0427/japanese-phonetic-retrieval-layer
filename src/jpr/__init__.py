"""Japanese Phonetic Retrieval Layer.

LLM が弱い「音韻空間」を外部の検索システムとして提供する。意味的類似性
(semantic embedding が担う) とは独立した軸として、発音の近さで日本語の語を引く。

    from jpr import PhoneticSearcher, PhoneticStore

    searcher = PhoneticSearcher(PhoneticStore(default_store_path()))
    pronunciation, results = searcher.search("乳首")
"""

from .distance import phoneme_distance, phonetic_similarity, weighted_edit_distance
from .index import Category, IndexEntry
from .phonology import Mora, Pronunciation, analyze_reading
from .reading import ReadingExtractor
from .search import (
    PRESETS,
    ComparisonResult,
    PhoneticSearcher,
    ScoreWeights,
    SearchResult,
    compare_pronunciations,
)
from .store import PhoneticStore, default_store_path

__all__ = [
    "PRESETS",
    "Category",
    "ComparisonResult",
    "IndexEntry",
    "Mora",
    "PhoneticSearcher",
    "PhoneticStore",
    "Pronunciation",
    "ReadingExtractor",
    "ScoreWeights",
    "SearchResult",
    "analyze_reading",
    "compare_pronunciations",
    "default_store_path",
    "phoneme_distance",
    "phonetic_similarity",
    "weighted_edit_distance",
]

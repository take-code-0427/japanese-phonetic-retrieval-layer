"""音韻検索エンジン (Phonetic RAG のオンライン側)。

2 段構成をとる。

    入力語 -> 読み -> 音素列 -> embedding
        -> ANN (HNSW) で候補 Top-K を高速取得        … 速度をここで稼ぐ
        -> weighted phonetic edit distance で rerank … 精度をここで稼ぐ
        -> 最終 Top-N

embedding だけに任せないのは、「なぜ近いのか」が曖昧になるため。ダジャレ用途
ではモーラ数が同じ・語尾が一致・子音だけ違う・母音列が似ている といった
局所的な構造が効くので、最終スコアはそれらを明示的に重み付けして決める。

用途によって近さの定義が変わるので、重みはプリセットで切り替える。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from .distance import (
    WORST_SUBSTITUTION_COST,
    phonetic_similarity,
    similarity_normalizer,
    weighted_edit_distance,
)
from .embedding import embed
from .index import DEFAULT_CATEGORIES, Category, IndexEntry, familiarity_of
from .phonology import Pronunciation, analyze_reading
from .reading import ReadingExtractor
from .store import PhoneticStore


@dataclass(frozen=True)
class ScoreWeights:
    """最終スコアの重み。

    `phoneme` は精密な編集距離、`embedding` は ANN 空間での近さ、
    `mora` はモーラ数の一致、`coda` は語尾の一致、`vowel` は母音列 (韻) の一致。
    `familiarity` は語の一般性で、音韻的に同等な候補の順序を決めるのに使う。
    """

    phoneme: float = 0.55
    embedding: float = 0.20
    mora: float = 0.05
    coda: float = 0.10
    vowel: float = 0.05
    familiarity: float = 0.05

    def normalized(self) -> ScoreWeights:
        total = (
            self.phoneme
            + self.embedding
            + self.mora
            + self.coda
            + self.vowel
            + self.familiarity
        )
        if total <= 0:
            raise ValueError("重みの合計が 0 です")
        return ScoreWeights(
            phoneme=self.phoneme / total,
            embedding=self.embedding / total,
            mora=self.mora / total,
            coda=self.coda / total,
            vowel=self.vowel / total,
            familiarity=self.familiarity / total,
        )


#: 用途別のプリセット。
#:
#: pun        ダジャレ・なぞなぞ。音韻全体の近さと、知られた語であることを重視。
#: rhyme      韻を踏む語。語尾と母音列を重視し、語頭の一致は求めない。
#: mishearing 聞き間違い・ASR 補正。全体の音韻とリズムの一致を最重視。
PRESETS: dict[str, ScoreWeights] = {
    "pun": ScoreWeights(
        phoneme=0.50, embedding=0.15, mora=0.05, coda=0.10, vowel=0.05, familiarity=0.15
    ),
    "rhyme": ScoreWeights(
        phoneme=0.20, embedding=0.10, mora=0.10, coda=0.35, vowel=0.20, familiarity=0.05
    ),
    "mishearing": ScoreWeights(
        phoneme=0.65, embedding=0.20, mora=0.10, coda=0.03, vowel=0.02, familiarity=0.00
    ),
}

DEFAULT_PRESET = "pun"

#: ANN から取る候補数の既定値。rerank 対象なので Top-N より十分大きく取る。
DEFAULT_CANDIDATES = 400


@dataclass(frozen=True)
class SearchResult:
    """検索結果 1 件。

    最終順位は `score` で決まるが、内訳も返すので「なぜ近いと判断したか」を
    呼び出し側 (LLM を含む) が検証できる。
    """

    surface: str
    reading: str
    #: 重み付き最終スコア。
    score: float
    #: 精密な音韻編集距離ベースの類似度。
    phonetic_similarity: float
    #: ANN 空間でのコサイン類似度。
    embedding_similarity: float
    #: 語尾の一致度。
    coda_similarity: float
    #: 母音列 (韻) の一致度。
    vowel_similarity: float
    mora_count: int
    pos: str
    category: Category
    phonemes: tuple[str, ...]
    familiarity: float

    @property
    def phoneme_string(self) -> str:
        return " ".join(self.phonemes)


@dataclass(frozen=True)
class ComparisonResult:
    """2 語の音韻比較結果。"""

    a_text: str
    a_reading: str
    a_phonemes: tuple[str, ...]
    b_text: str
    b_reading: str
    b_phonemes: tuple[str, ...]
    similarity: float
    distance: float
    #: 空間ごとのコサイン類似度。どの側面が似ているのかがわかる。
    spaces: dict[str, float] = field(default_factory=dict)


class PhoneticSearcher:
    """ANN + rerank の 2 段検索。"""

    #: rerank 時にモーラ数がこれ以上離れた候補は落とす。
    _MAX_MORA_GAP = 3

    def __init__(
        self,
        store: PhoneticStore,
        extractor: ReadingExtractor | None = None,
        *,
        ann_space: str = "phonetic",
    ) -> None:
        self.store = store
        self.ann_space = ann_space
        self._extractor = extractor
        self._category_ids = store.category_ids
        self._mora_counts = store.mora_counts

    @property
    def extractor(self) -> ReadingExtractor:
        """読み取得器。Sudachi のロードが重いので初回参照まで遅延させる。"""
        if self._extractor is None:
            self._extractor = ReadingExtractor(dict_type=self.store.meta.dict_type)
        return self._extractor

    def pronounce(self, text: str) -> Pronunciation:
        """テキストを音韻表現に変換する。"""
        return analyze_reading(self.extractor.reading_of(text))

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        preset: str = DEFAULT_PRESET,
        weights: ScoreWeights | None = None,
        candidates: int = DEFAULT_CANDIDATES,
        min_score: float = 0.0,
        categories: Iterable[Category] | None = None,
        exclude_same_reading: bool = True,
        dedupe_by_reading: bool = True,
        candidate_filter: Callable[[IndexEntry], bool] | None = None,
    ) -> tuple[Pronunciation, list[SearchResult]]:
        """`query` と音が近い語を返す。

        戻り値は (クエリの音韻表現, 結果リスト)。

        `preset` は "pun" / "rhyme" / "mishearing"。`weights` を渡すと
        プリセットより優先される。`candidates` は ANN から取る候補数で、
        大きくすると再現率が上がり遅くなる。

        `candidate_filter` は候補を意味的に絞るためのフック。音韻空間だけでは
        「乳首」に音が近い語は数百あり、そのうちどれが答えかは意味の制約
        (「お菓子である」など) が決める。呼び出し側が語彙のリストなどで
        絞り込めるようにしている。
        """
        pronunciation = self.pronounce(query)
        if not pronunciation.phonemes:
            return pronunciation, []

        active_weights = (weights or self._preset(preset)).normalized()
        query_vectors = embed(pronunciation)

        rows, embedding_scores = self._ann_candidates(query_vectors, candidates)
        if rows.size == 0:
            return pronunciation, []

        rows, embedding_scores = self._apply_cheap_filters(
            rows,
            embedding_scores,
            pronunciation,
            categories,
        )
        if rows.size == 0:
            return pronunciation, []

        results = self._rerank(
            rows=rows,
            embedding_scores=embedding_scores,
            query=query,
            pronunciation=pronunciation,
            query_vectors=query_vectors,
            weights=active_weights,
            exclude_same_reading=exclude_same_reading,
            candidate_filter=candidate_filter,
            min_score=min_score,
        )

        if dedupe_by_reading:
            results = _dedupe_by_reading(results)

        results.sort(key=lambda r: (-r.score, r.mora_count, r.surface))
        return pronunciation, results[:limit]

    # --- 段 1: ANN 候補生成 ------------------------------------------------

    def _ann_candidates(
        self,
        query_vectors: dict[str, np.ndarray],
        candidates: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """HNSW で候補行とコサイン類似度を得る。"""
        wanted = min(candidates, len(self.store))
        index = self.store.ann(self.ann_space, ef=max(wanted, 64))
        query = query_vectors[self.ann_space].reshape(1, -1)
        labels, distances = index.knn_query(query, k=wanted)
        # hnswlib の 'ip' 空間は 1 - 内積を返す。正規化済みベクトルなので
        # これを戻すとコサイン類似度になる。
        return labels[0], 1.0 - distances[0]

    def _apply_cheap_filters(
        self,
        rows: np.ndarray,
        embedding_scores: np.ndarray,
        pronunciation: Pronunciation,
        categories: Iterable[Category] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """編集距離の前に、配列演算で済む条件で候補を削る。"""
        wanted = tuple(categories) if categories is not None else tuple(DEFAULT_CATEGORIES)
        wanted_ids = [self.store.category_id(c) for c in wanted]
        wanted_ids = [i for i in wanted_ids if i >= 0]
        if not wanted_ids:
            return rows[:0], embedding_scores[:0]
        keep = np.isin(self._category_ids[rows], np.array(wanted_ids, dtype=np.int8))

        # モーラ数が大きく離れた語は音韻的な近さとして意味がないので落とす。
        gap = np.abs(self._mora_counts[rows].astype(np.int32) - pronunciation.mora_count)
        keep &= gap <= self._MAX_MORA_GAP

        return rows[keep], embedding_scores[keep]

    # --- 段 2: 精密な音韻距離で rerank ------------------------------------

    def _rerank(
        self,
        *,
        rows: np.ndarray,
        embedding_scores: np.ndarray,
        query: str,
        pronunciation: Pronunciation,
        query_vectors: dict[str, np.ndarray],
        weights: ScoreWeights,
        exclude_same_reading: bool,
        candidate_filter: Callable[[IndexEntry], bool] | None,
        min_score: float,
    ) -> list[SearchResult]:
        query_phonemes = pronunciation.phonemes
        query_moras = pronunciation.mora_count
        normalized_query = self.extractor.normalize(query)

        coda_matrix = self.store.vectors("coda")
        vowel_matrix = self.store.vectors("vowel")
        query_coda = query_vectors["coda"]
        query_vowel = query_vectors["vowel"]

        results: list[SearchResult] = []
        for row, embedding_score in zip(rows.tolist(), embedding_scores.tolist(), strict=True):
            entry = self.store.entry(row)

            if exclude_same_reading and entry.reading == pronunciation.reading:
                continue
            if entry.surface == query or entry.surface == normalized_query:
                continue
            if candidate_filter is not None and not candidate_filter(entry):
                continue

            phonetic = _edit_similarity(query_phonemes, entry.phonemes)
            coda = float(query_coda @ coda_matrix[row])
            vowel = float(query_vowel @ vowel_matrix[row])
            mora = _mora_similarity(query_moras, entry.mora_count)
            familiarity = familiarity_of(entry.cost)

            score = (
                weights.phoneme * phonetic
                + weights.embedding * max(0.0, embedding_score)
                + weights.mora * mora
                + weights.coda * max(0.0, coda)
                + weights.vowel * max(0.0, vowel)
                + weights.familiarity * familiarity
            )
            if score < min_score:
                continue

            results.append(
                SearchResult(
                    surface=entry.surface,
                    reading=entry.reading,
                    score=round(score, 4),
                    phonetic_similarity=round(phonetic, 4),
                    embedding_similarity=round(max(0.0, embedding_score), 4),
                    coda_similarity=round(max(0.0, coda), 4),
                    vowel_similarity=round(max(0.0, vowel), 4),
                    mora_count=entry.mora_count,
                    pos=entry.pos,
                    category=entry.category,
                    phonemes=entry.phonemes,
                    familiarity=round(familiarity, 3),
                )
            )

        return results

    @staticmethod
    def _preset(name: str) -> ScoreWeights:
        try:
            return PRESETS[name]
        except KeyError:
            raise ValueError(
                f"未知のプリセット: {name} (利用可能: {', '.join(sorted(PRESETS))})"
            ) from None

    # --- 比較 -------------------------------------------------------------

    def compare(self, a: str, b: str) -> ComparisonResult:
        """2 つのテキストの音韻類似度を計算する。"""
        pa = self.pronounce(a)
        pb = self.pronounce(b)
        va, vb = embed(pa), embed(pb)

        spaces: dict[str, float] = {}
        for name in va:
            if name == "rhythm":
                # rhythm は正規化していないので距離で見る。
                gap = float(np.linalg.norm(va[name] - vb[name]))
                spaces[name] = round(max(0.0, 1.0 - gap), 4)
            else:
                spaces[name] = round(float(va[name] @ vb[name]), 4)

        return ComparisonResult(
            a_text=a,
            a_reading=pa.reading,
            a_phonemes=pa.phonemes,
            b_text=b,
            b_reading=pb.reading,
            b_phonemes=pb.phonemes,
            similarity=round(phonetic_similarity(pa, pb), 4),
            distance=round(weighted_edit_distance(pa.phonemes, pb.phonemes), 4),
            spaces=spaces,
        )


def _edit_similarity(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """精密な音韻編集距離ベースの類似度。"""
    if not a or not b:
        return 0.0
    distance = weighted_edit_distance(a, b)
    denominator = similarity_normalizer(len(a), len(b), WORST_SUBSTITUTION_COST)
    if denominator <= 0.0:
        return 1.0
    return max(0.0, 1.0 - distance / denominator)


def _mora_similarity(a: int, b: int) -> float:
    """モーラ数の近さ。同数なら 1.0、離れるにつれ線形に下がる。"""
    if a == b:
        return 1.0
    return max(0.0, 1.0 - abs(a - b) / max(a, b, 1))


def _dedupe_by_reading(results: list[SearchResult]) -> list[SearchResult]:
    """同音の異表記 (「仕組」「仕組み」「し組み」…) を 1 件に畳む。

    音韻検索では同じ音を何度返しても情報量が増えないため、読みごとに
    最もスコアが高く、同点なら一般的な表記を残す。
    """
    best: dict[str, SearchResult] = {}
    for result in results:
        current = best.get(result.reading)
        if current is None or (result.score, result.familiarity) > (
            current.score,
            current.familiarity,
        ):
            best[result.reading] = result
    return list(best.values())


__all__ = [
    "DEFAULT_CANDIDATES",
    "DEFAULT_PRESET",
    "PRESETS",
    "ComparisonResult",
    "PhoneticSearcher",
    "ScoreWeights",
    "SearchResult",
]

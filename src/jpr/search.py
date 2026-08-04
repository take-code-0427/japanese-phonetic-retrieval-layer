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
    INDEL_COST,
    WORST_SUBSTITUTION_COST,
    edit_distance_batch,
    phoneme_ids,
    phonetic_similarity,
    weighted_edit_distance,
)
from .embedding import embed
from .index import (
    COST_FAMILIAR,
    COST_RARE,
    DEFAULT_CATEGORIES,
    Category,
    IndexEntry,
)
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
            self.phoneme + self.embedding + self.mora + self.coda + self.vowel + self.familiarity
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

#: ANN から取る候補数の既定値。
#:
#: 202 万語では 3 モーラ語のクエリに対しコサイン 0.91 以上の候補が 400 件を超える。
#: k=400 では「乳首」に対する「手首」のような明らかな近傍が候補にすら入らず、
#: 稀語ばかりが並んだ。実測では k=5000 で「手首」が 1 位に来る。
#: 候補数を増やすと rerank のコストが線形に増えるので、品質が飽和する手前で止める。
DEFAULT_CANDIDATES = 5000

#: 打ち切り線を決めるために確定させる件数の、limit に対する倍率。
#: 同音の異表記が後段で 1 件に畳まれるため、limit ぴったりで打ち切ると
#: 畳んだ後に件数が足りなくなる。
_RERANK_MARGIN = 8


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


@dataclass(frozen=True)
class _ScoredCandidates:
    """rerank で求めた候補全件のスコアと内訳。

    スコア計算と、上位を `SearchResult` に起こす処理を分けている。結果が
    limit に届かないときは選抜幅だけ広げて起こし直せばよく、配列の計算を
    やり直す必要がない。
    """

    rows: np.ndarray
    scores: np.ndarray
    phonetic: np.ndarray
    embedding: np.ndarray
    coda: np.ndarray
    vowel: np.ndarray
    familiarity: np.ndarray

    @property
    def count(self) -> int:
        return int(self.rows.size)


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

        scored = self._score_candidates(
            rows=rows,
            embedding_scores=embedding_scores,
            pronunciation=pronunciation,
            query_vectors=query_vectors,
            weights=active_weights,
        )

        # 同音異表記が畳まれる分と、除外・フィルタで落ちる分だけ結果が目減りする。
        # どれだけ減るかは走らせるまでわからないので、limit に届かなければ
        # 選抜幅を広げて作り直す。スコアは全候補ぶん計算済みなので、
        # やり直しのコストは選抜と文字列の復号だけ。
        keep = max(limit * _RERANK_MARGIN, limit)
        while True:
            results = self._materialize(
                scored=scored,
                query=query,
                pronunciation=pronunciation,
                exclude_same_reading=exclude_same_reading,
                candidate_filter=candidate_filter,
                min_score=min_score,
                keep=keep,
            )
            if dedupe_by_reading:
                results = _dedupe_by_reading(results)
            if len(results) >= limit or keep >= scored.count:
                break
            keep = min(keep * 4, scored.count)

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

    def _score_candidates(
        self,
        *,
        rows: np.ndarray,
        embedding_scores: np.ndarray,
        pronunciation: Pronunciation,
        query_vectors: dict[str, np.ndarray],
        weights: ScoreWeights,
    ) -> _ScoredCandidates:
        """候補全件のスコアと内訳を配列で求める。"""
        store = self.store
        query_ids = phoneme_ids(pronunciation.phonemes)

        # ベクトルの内積とモーラ数・一般性は配列演算でまとめて出す。
        coda = np.clip(query_vectors["coda"] @ store.vectors("coda")[rows].T, 0.0, None)
        vowel = np.clip(query_vectors["vowel"] @ store.vectors("vowel")[rows].T, 0.0, None)
        embedding = np.clip(embedding_scores, 0.0, None)
        candidate_moras = store.mora_counts[rows].astype(np.int32)
        mora = _mora_similarity_array(pronunciation.mora_count, candidate_moras)
        familiarity = _familiarity_array(store.costs[rows])

        # 編集距離も候補全体を 1 度の DP で計算する。以前は編集距離を除いた
        # 上限 (partial + weights.phoneme) の高い順に見て、確定済み上位の
        # 最低スコアを打ち切り線にし、見込みのない候補には距離を掛けなかった。
        # 1 件ずつの距離計算が支配的だった頃はこれが効いたが、バッチ化すると
        # 1 件あたり 80μs から 4μs に落ちる。打ち切りで削れるのは実測で候補の
        # 15% 程度しかなく、全件まとめて計算するほうが速い (2000 件で
        # 190ms -> 9ms)。上限計算が不要になったので、スコアの重み構成を
        # 変えても距離計算側の妥当性を気にする必要はなくなった。
        matrix, lengths = store.phoneme_id_matrix(rows)
        distances = edit_distance_batch(query_ids, matrix, lengths)
        phonetic = _edit_similarity_array(distances, query_ids.size, lengths)

        scores = (
            weights.embedding * embedding
            + weights.mora * mora
            + weights.coda * coda
            + weights.vowel * vowel
            + weights.familiarity * familiarity
            + weights.phoneme * phonetic
        )

        return _ScoredCandidates(
            rows=rows,
            scores=scores,
            phonetic=phonetic,
            embedding=embedding,
            coda=coda,
            vowel=vowel,
            familiarity=familiarity,
        )

    def _materialize(
        self,
        *,
        scored: _ScoredCandidates,
        query: str,
        pronunciation: Pronunciation,
        exclude_same_reading: bool,
        candidate_filter: Callable[[IndexEntry], bool] | None,
        min_score: float,
        keep: int,
    ) -> list[SearchResult]:
        """スコア上位 `keep` 件を `SearchResult` に起こす。

        文字列の復号と `IndexEntry` の生成はここだけで行う。候補全件で
        `entry()` を作ると 2000 件で 40ms かかるが、必要なのは同音異表記を
        畳んだ後に limit 件残る分だけ。
        """
        store = self.store
        scores = scored.scores
        eligible = np.flatnonzero(scores >= min_score) if min_score > 0.0 else None
        selected = _top_positions(scores, keep, eligible)

        results: list[SearchResult] = []
        normalized_query = self.extractor.normalize(query)
        for position in selected:
            row = int(scored.rows[position])
            reading = store.reading(row)
            if exclude_same_reading and reading == pronunciation.reading:
                continue
            surface = store.surface(row)
            if surface in (query, normalized_query):
                continue

            entry = store.entry(row)
            if candidate_filter is not None and not candidate_filter(entry):
                continue

            results.append(
                SearchResult(
                    surface=entry.surface,
                    reading=entry.reading,
                    score=round(float(scores[position]), 4),
                    phonetic_similarity=round(float(scored.phonetic[position]), 4),
                    embedding_similarity=round(float(scored.embedding[position]), 4),
                    coda_similarity=round(float(scored.coda[position]), 4),
                    vowel_similarity=round(float(scored.vowel[position]), 4),
                    mora_count=entry.mora_count,
                    pos=entry.pos,
                    category=entry.category,
                    phonemes=entry.phonemes,
                    familiarity=round(float(scored.familiarity[position]), 3),
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
        return compare_pronunciations(a, b, self.pronounce(a), self.pronounce(b))


def compare_pronunciations(
    a_text: str,
    b_text: str,
    a: Pronunciation,
    b: Pronunciation,
) -> ComparisonResult:
    """2 つの音韻表現を比較する。索引を必要としない。"""
    va, vb = embed(a), embed(b)

    spaces: dict[str, float] = {}
    for name in va:
        if name == "rhythm":
            # rhythm は正規化していないので距離で見る。
            gap = float(np.linalg.norm(va[name] - vb[name]))
            spaces[name] = round(max(0.0, 1.0 - gap), 4)
        else:
            spaces[name] = round(float(va[name] @ vb[name]), 4)

    return ComparisonResult(
        a_text=a_text,
        a_reading=a.reading,
        a_phonemes=a.phonemes,
        b_text=b_text,
        b_reading=b.reading,
        b_phonemes=b.phonemes,
        similarity=round(phonetic_similarity(a, b), 4),
        distance=round(weighted_edit_distance(a.phonemes, b.phonemes), 4),
        spaces=spaces,
    )


def _edit_similarity_array(
    distances: np.ndarray,
    query_length: int,
    candidate_lengths: np.ndarray,
) -> np.ndarray:
    """編集距離を候補ごとの分母で正規化して類似度にする。

    分母は `distance.similarity_normalizer` と同じ式を候補全体に対して
    配列で解いたもの。**片方だけ変えると検索スコアと `compare` の類似度が
    静かに食い違う** (`tests/test_search.py` が両者の一致を検証する)。
    """
    if distances.size == 0:
        return distances
    shorter = np.minimum(candidate_lengths, query_length).astype(np.float64)
    longer = np.maximum(candidate_lengths, query_length).astype(np.float64)
    denominator = shorter * WORST_SUBSTITUTION_COST + (longer - shorter) * INDEL_COST
    # 長さが 0 の候補は分母も 0 になる。距離で割れないので類似度 0 とする。
    similarity = np.divide(
        distances,
        denominator,
        out=np.ones_like(distances),
        where=denominator > 0.0,
    )
    return np.clip(1.0 - similarity, 0.0, None)


def _top_positions(
    scores: np.ndarray,
    keep: int,
    eligible: np.ndarray | None,
) -> list[int]:
    """スコア上位 `keep` 件の位置を降順で返す。

    全体を argsort すると候補数に比例したコストを払うが、必要なのは上位
    数十件だけなので argpartition で切ってから並べ替える。
    """
    positions = np.arange(scores.size) if eligible is None else eligible
    if positions.size == 0:
        return []
    values = scores if eligible is None else scores[eligible]
    if positions.size > keep:
        cut = np.argpartition(-values, keep - 1)[:keep]
        positions, values = positions[cut], values[cut]
    return positions[np.argsort(-values)].tolist()


def _mora_similarity(a: int, b: int) -> float:
    """モーラ数の近さ。同数なら 1.0、離れるにつれ線形に下がる。"""
    if a == b:
        return 1.0
    return max(0.0, 1.0 - abs(a - b) / max(a, b, 1))


def _mora_similarity_array(query_moras: int, candidates: np.ndarray) -> np.ndarray:
    """`_mora_similarity` を候補全体にまとめて適用する。"""
    gap = np.abs(candidates - query_moras).astype(np.float64)
    scale = np.maximum(np.maximum(candidates, query_moras), 1).astype(np.float64)
    return np.clip(1.0 - gap / scale, 0.0, 1.0)


def _familiarity_array(costs: np.ndarray) -> np.ndarray:
    """`familiarity_of` を候補全体にまとめて適用する。"""
    span = COST_RARE - COST_FAMILIAR
    return np.clip(1.0 - (costs.astype(np.float64) - COST_FAMILIAR) / span, 0.0, 1.0)


def _representative_rank(result: SearchResult) -> tuple[float, float, float]:
    """同音異表記の中で代表として残す優先度。大きいほど優先。

    SudachiDict では読みをそのまま書いたカタカナ見出し (「カカク」) が
    漢字表記 (「価格」) より低コストなことがある。コストだけで選ぶと
    「科学」の近傍が「カカク」になり、語として何を指すのかが読み取れない。
    見出しの情報量を優先し、コストは同程度の表記を選ぶときの手がかりに使う。

    同じ読みの候補は音韻類似度が等しいので、スコアの差は一般性の重みぶんしか
    生じない。だから情報量をスコアより先に見る。
    """
    return (_surface_informativeness(result), result.score, result.familiarity)


def _surface_informativeness(result: SearchResult) -> float:
    """表層が語を特定できる度合い。

    読みと同じカタカナ列をそのまま並べた見出しは、音は合っていても
    どの語なのかを伝えないので最も低く見る。
    """
    if result.surface == result.reading:
        return 0.0
    if any("一" <= ch <= "鿿" for ch in result.surface):
        return 1.0
    return 0.5


def _dedupe_by_reading(results: list[SearchResult]) -> list[SearchResult]:
    """同音の異表記 (「仕組」「仕組み」「し組み」…) を 1 件に畳む。

    音韻検索では同じ音を何度返しても情報量が増えないため、読みごとに
    代表を 1 つだけ残す。
    """
    best: dict[str, SearchResult] = {}
    for result in results:
        current = best.get(result.reading)
        if current is None or _representative_rank(result) > _representative_rank(current):
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
    "compare_pronunciations",
]

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
from functools import partial

import numpy as np

from .distance import (
    INDEL_COST,
    WORST_SUBSTITUTION_COST,
    edit_distance_csr,
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

#: 結果が `limit` に届かないときに選抜幅を広げる上限 (初回の何倍まで)。
#:
#: 枝刈り (`PhoneticSearcher._survivors`) が順位を保証する範囲を決めるので、
#: 再試行のループと `needed` の見積もりで同じ値を使わなければならない。
#: 実測では 32 クエリ x limit=10/50 で初回の 4 倍 (keep=320) までしか広がらない。
#:
#: 上げるほど枝刈りが効かなくなる。`needed` が増えると打ち切り線が下がって
#: 生存率が上がるためで、16 倍にすると 4〜6 モーラの検索が 326ms から 457ms へ
#: 悪化した (枝刈りなしより遅い — 標本の距離計算が丸ごと無駄になる)。
_RETRY_LIMIT = 4

#: 編集距離の打ち切り線を求めるために、先に本計算する標本の件数
#: (`PhoneticSearcher._survivors`)。
#:
#: 標本を増やしても打ち切り線は動かない。上位数十件は 32768 件の標本にすでに
#: 入っているので、線を決めるのに必要な情報が出揃っている (4 倍の 131072 件に
#: しても 53 万候補での生存率は pun 18.7% / mishearing 97.0% で変わらず、
#: 標本の計算コストだけが増えて 212ms から 283ms に悪化した)。
#:
#: 下限としても働く。候補がこれ以下なら距離計算が数ミリ秒で済むので、
#: 打ち切りの判定を挟まず全件計算する (ANN 経路の 2000 件では距離が 0.6ms)。
_PROBE_CANDIDATES = 32768

#: 編集距離の打ち切りを試みる `weights.phoneme` の上限。
#:
#: 打ち切り線は「編集距離が最良だった場合のスコア」を上限として引くので、
#: `weights.phoneme` が大きいほど上限が緩く、候補が落ちなくなる。4〜6 モーラ
#: (53 万候補) での生存率の実測:
#:
#:     w.phoneme  0.20   0.30   0.40   0.50   0.60   0.65
#:     生存率     1.8%   7.2%  23.7%  62.7%  96.9%  99.9%
#:
#: 生存率が高いと標本 (`_PROBE_CANDIDATES` 件) の距離計算がそのまま無駄な
#: 上乗せになる。mishearing (0.65) では 3〜8 モーラの検索が 237ms から 348ms へ
#: 悪化した。**枝刈りは効くときだけ試みる** — 0.45 は生存率が 3 割を超える
#: 手前に置いた線で、rhyme (0.20) は通し mishearing (0.65) は通さない。
#: pun (0.50) も外れるが、あれは境界付近で削減と標本コストが釣り合う
#: (実測 -20%〜+14% とばらつき、平均すると得がない)。
_MAX_PHONEME_WEIGHT_FOR_PRUNING = 0.45


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

    #: 候補が索引全体のこの割合を超えたら、行を選んで引くのをやめて全行走査する
    #: (`_space_scores` 参照)。実測の分岐点は 1〜2% で、そこを少し上に取っている。
    _FULL_SCAN_RATIO = 0.02

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
        limit: int | None = 10,
        preset: str = DEFAULT_PRESET,
        weights: ScoreWeights | None = None,
        candidates: int = DEFAULT_CANDIDATES,
        min_score: float = 0.0,
        categories: Iterable[Category] | None = None,
        min_mora: int | None = None,
        max_mora: int | None = None,
        exclude_same_reading: bool = True,
        dedupe_by_reading: bool = True,
        candidate_filter: Callable[[IndexEntry], bool] | None = None,
    ) -> tuple[Pronunciation, list[SearchResult]]:
        """`query` と音が近い語を返す。

        戻り値は (クエリの音韻表現, 結果リスト)。

        `preset` は "pun" / "rhyme" / "mishearing"。`weights` を渡すと
        プリセットより優先される。`candidates` は ANN から取る候補数で、
        大きくすると再現率が上がり遅くなる。

        `min_mora` / `max_mora` はモーラ数の範囲。**どちらかを指定すると ANN を
        使わず、その範囲の行を全走査する。** ANN の候補生成は phonetic 空間の
        Top-K なので、モーラ数の違う語はそもそも候補に入らない。「乳首」
        (3 モーラ) に対する「筑前煮」(5 モーラ) は phonetic 空間で全体
        154148 位、5 モーラ内に限っても 23823/301551 位で、candidates を
        20000 に増やしても拾えなかった。候補を後段で絞る `candidate_filter`
        方式でも ANN を通った分しか見ないので同じ取りこぼしが起きる。だから
        範囲指定時は候補生成そのものを差し替える。代償は速度で、1 モーラ長
        あたり 1.0〜1.6 秒 (実測: 4 モーラ 36 万語 1374ms / 5 モーラ 30 万語
        1020ms / 6 モーラ 30 万語 1575ms)。

        `limit=None` は「上限なし」。スコア閾値 (`min_score`) で母集団を切って
        その全件を得る用途を想定している。`min_score` なしで併用すると閾値を
        通った全候補を `SearchResult` に起こすので、モーラ範囲の全走査と
        組み合わせると数十万件になる。

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

        # 候補生成には 2 つの経路がある。モーラ範囲を指定したときだけ
        # ANN を迂回して母集団を全走査する (`_scan_candidates` 参照)。
        selection = self._selection_mask(min_mora=min_mora, max_mora=max_mora)
        if selection is None:
            rows, embedding_scores = self._ann_candidates(query_vectors, candidates)
        else:
            rows, embedding_scores = self._scan_candidates(query_vectors, selection)
        if rows.size == 0:
            return pronunciation, []

        rows, embedding_scores = self._apply_cheap_filters(
            rows,
            embedding_scores,
            pronunciation,
            categories,
            apply_mora_gap=selection is None,
        )
        if rows.size == 0:
            return pronunciation, []

        # 編集距離を計算する候補を削るために、後段が必要とする件数を見積もる
        # (`_survivors`)。上位が何件あれば足りるかが確定しない経路では削らない:
        #
        # - `limit=None` は全候補を起こす。
        # - `min_score` は閾値を通った候補を全件返すので、上位 N 件の保証では
        #   足りない (枝刈りで落ちた候補も閾値を超えている可能性がある)。
        # - `candidate_filter` は呼び出し側の意味的な絞り込みで、どれだけ落ちるか
        #   わからない。下の再試行が選抜幅を広げても、削った候補は戻せない。
        #
        # 削る場合も下の再試行が広げ得る上限 (`_RETRY_LIMIT` 倍) まで保証して
        # おく。枝刈りは「上位 `needed` 件の順位が全件計算と一致する」ことだけを
        # 約束するので、それを超えて選抜幅を広げると順位の保証の外に出る。
        if limit is not None and min_score <= 0.0 and candidate_filter is None:
            needed = max(limit * _RERANK_MARGIN, limit) * _RETRY_LIMIT
        else:
            needed = rows.size

        scored = self._score_candidates(
            rows=rows,
            embedding_scores=embedding_scores,
            pronunciation=pronunciation,
            query_vectors=query_vectors,
            weights=active_weights,
            needed=needed,
        )

        materialize = partial(
            self._materialize,
            scored=scored,
            query=query,
            pronunciation=pronunciation,
            exclude_same_reading=exclude_same_reading,
            candidate_filter=candidate_filter,
            min_score=min_score,
        )

        if limit is None:
            # 上限が無いので選抜幅を広げる再試行が意味を持たない。最初から
            # 全候補を起こす。実際に起きるのは `min_score` を通った分だけ。
            results = materialize(keep=scored.count)
            if dedupe_by_reading:
                results = _dedupe_by_reading(results)
        else:
            # 同音異表記が畳まれる分と、除外・フィルタで落ちる分だけ結果が
            # 目減りする。どれだけ減るかは走らせるまでわからないので、limit に
            # 届かなければ選抜幅を広げて作り直す。スコアは計算済みなので、
            # やり直しのコストは選抜と文字列の復号だけ。
            #
            # 枝刈りした場合だけ `_RETRY_LIMIT` 倍で打ち止める。枝刈りが順位を
            # 保証するのはそこまでなので (上の `needed` 参照)、超えて広げると
            # 順位が全件計算と一致しない候補を混ぜることになる。削っていない
            # 経路 (`candidate_filter` など) では候補を全部使い切る — フィルタが
            # 候補の大半を落とすので、ここを絞ると結果が空になる。
            initial = max(limit * _RERANK_MARGIN, limit)
            ceiling = min(needed, scored.count)
            keep = min(initial, ceiling)
            while True:
                results = materialize(keep=keep)
                if dedupe_by_reading:
                    results = _dedupe_by_reading(results)
                if len(results) >= limit or keep >= ceiling:
                    break
                keep = min(keep * 4, ceiling)

        results.sort(key=lambda r: (-r.score, r.mora_count, r.surface))
        return pronunciation, results if limit is None else results[:limit]

    def mora_range_size(self, min_mora: int | None, max_mora: int | None) -> int:
        """モーラ範囲に該当する索引上の語数。全走査のコストの目安になる。

        警告を出すかどうかは窓ごとに判断が違う (CLI は標準エラーに 1 行、
        Web はレスポンスに含める、MCP は stdio を壊せない) ので、
        `search` から副作用として出さずに問い合わせられるようにしている。
        """
        mask = self._selection_mask(min_mora=min_mora, max_mora=max_mora)
        return len(self.store) if mask is None else int(mask.sum())

    # --- 段 1: 候補生成 ----------------------------------------------------

    def _selection_mask(
        self,
        *,
        min_mora: int | None,
        max_mora: int | None,
    ) -> np.ndarray | None:
        """索引全体から候補生成の母集団を切り出す真偽マスク。

        `None` は「絞り込みが指定されていない」= ANN 経路でよい、を表す。

        ここに積める条件は store が列配列で持っている軸に限る
        (`mora_counts` / `costs` / `category_ids`)。行を Python オブジェクト
        に起こさずにマスクを立てられるので、ANN を通さずに母集団を決められる。
        モーラ範囲以外の軸を足すときも、同じように `&=` で条件を積む。
        カテゴリだけは絞り込みの有無に関わらず常に効くので
        `_apply_cheap_filters` 側に置いたままにしてある。
        """
        if min_mora is None and max_mora is None:
            return None
        if min_mora is not None and max_mora is not None and min_mora > max_mora:
            raise ValueError(f"モーラ範囲が逆転しています: {min_mora} > {max_mora}")

        moras = self._mora_counts
        mask = np.ones(moras.size, dtype=bool)
        if min_mora is not None:
            mask &= moras >= min_mora
        if max_mora is not None:
            mask &= moras <= max_mora
        return mask

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
        return labels[0].astype(np.int64), 1.0 - distances[0]

    def _scan_candidates(
        self,
        query_vectors: dict[str, np.ndarray],
        selection: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """マスクが立った行を全走査し、内積を直接求める。

        ANN を使わない経路。HNSW は phonetic 空間の Top-K しか返さないので、
        モーラ数の違う語は近傍の外に沈んで候補にすら入らない (`search` の
        docstring 参照)。ここでは母集団の全行に対して内積を取るので、
        近傍順位に関わらず rerank の土俵に載る。

        **選んだ行だけを引くのではなく、全行と内積を取ってからマスクする。**
        直感に反するが実測でこちらが 14 倍速い (97 万行の選抜で 228ms -> 16ms)。
        `vectors[rows]` の内訳は fancy indexing が 191ms、内積が 7ms で、
        コストは演算ではなく mmap から飛び飛びに読んだ行を新しい配列へ
        実体化する部分にある。全行走査は連続読みなので、対象が 2 倍の
        202 万行あっても順次アクセスの帯域で押し切れる。

        (以前はここで `vectors[rows]` を実体化していた。「全行と内積を取る案は
        5 倍遅い」と判断していたが、それは編集距離が支配的で内積の差が埋もれて
        いた頃の測定。編集距離を Rust に移して 26 倍速くなった結果、内積が
        全走査の中で最大の項目になり、力関係が入れ替わった。)
        """
        rows = np.flatnonzero(selection).astype(np.int64)
        if rows.size == 0:
            return rows, np.zeros(0, dtype=np.float32)
        vectors = self.store.vectors(self.ann_space)
        # 正規化済みベクトルなので内積がそのままコサイン類似度になり、
        # ANN 経路が返すスコアと同じ尺度で揃う。
        scores = (vectors @ query_vectors[self.ann_space])[rows]
        return rows, scores

    def _apply_cheap_filters(
        self,
        rows: np.ndarray,
        embedding_scores: np.ndarray,
        pronunciation: Pronunciation,
        categories: Iterable[Category] | None,
        *,
        apply_mora_gap: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """編集距離の前に、配列演算で済む条件で候補を削る。"""
        wanted = tuple(categories) if categories is not None else tuple(DEFAULT_CATEGORIES)
        wanted_ids = [self.store.category_id(c) for c in wanted]
        wanted_ids = [i for i in wanted_ids if i >= 0]
        if not wanted_ids:
            return rows[:0], embedding_scores[:0]
        keep = np.isin(self._category_ids[rows], np.array(wanted_ids, dtype=np.int8))

        # モーラ数が大きく離れた語は音韻的な近さとして意味がないので落とす。
        # ただしこれは ANN の候補生成が粗いことを補う安全網なので、呼び出し側が
        # モーラ範囲を明示したときは適用しない。3 モーラのクエリに 7 モーラを
        # 要求すると、ギャップ 4 で全件落ちてしまう。
        if apply_mora_gap:
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
        needed: int,
    ) -> _ScoredCandidates:
        """候補のスコアと内訳を配列で求める。

        `needed` は後段が必要とする件数。これを使って編集距離を計算する候補を
        削るが (`_survivors`)、**上位 `needed` 件の順位とスコアは全件計算した
        場合と一致する**。
        """
        store = self.store
        query_ids = phoneme_ids(pronunciation.phonemes)

        # ベクトルの内積とモーラ数・一般性は配列演算でまとめて出す。
        # 編集距離より 2 桁安いので、絞り込む前に全候補ぶん出しておく。
        coda = np.clip(self._space_scores("coda", rows, query_vectors), 0.0, None)
        vowel = np.clip(self._space_scores("vowel", rows, query_vectors), 0.0, None)
        embedding = np.clip(embedding_scores, 0.0, None)
        candidate_moras = store.mora_counts[rows].astype(np.int32)
        mora = _mora_similarity_array(pronunciation.mora_count, candidate_moras)
        familiarity = _familiarity_array(store.costs[rows])

        partial = (
            weights.embedding * embedding
            + weights.mora * mora
            + weights.coda * coda
            + weights.vowel * vowel
            + weights.familiarity * familiarity
        )

        # 編集距離だけは候補数に対して重い (53 万件で 48ms、他の成分の合計の
        # 1.4 倍)。順位に入り得ない候補には計算しない。
        survivors, phonetic = self._survivors(rows, partial, query_ids, weights, needed)
        if survivors is not None:
            rows = rows[survivors]
            partial = partial[survivors]
            coda, vowel = coda[survivors], vowel[survivors]
            embedding, familiarity = embedding[survivors], familiarity[survivors]

        # 編集距離は残った候補を 1 度の DP でまとめて計算する。索引の CSR を
        # Rust にそのまま渡すのでパディング行列を組む処理が不要になり、
        # 53 万候補で 1097ms (行列 102ms + DP 990ms) が 41ms になる。
        # 打ち切り線を出すのに使った標本ぶんは計算済みなので、埋まっていない
        # ところだけを計算する。
        if phonetic is None:
            phonetic = self._phonetic_scores(rows, query_ids)
        else:
            missing = np.flatnonzero(np.isnan(phonetic))
            if missing.size:
                phonetic[missing] = self._phonetic_scores(rows[missing], query_ids)

        scores = partial + weights.phoneme * phonetic

        return _ScoredCandidates(
            rows=rows,
            scores=scores,
            phonetic=phonetic,
            embedding=embedding,
            coda=coda,
            vowel=vowel,
            familiarity=familiarity,
        )

    def _survivors(
        self,
        rows: np.ndarray,
        partial: np.ndarray,
        query_ids: np.ndarray,
        weights: ScoreWeights,
        needed: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """編集距離を計算する必要がある候補の位置と、計算済みの距離。

        戻り値は (生存位置, 生存分の類似度)。前者が `None` なら「全件必要」で
        枝刈りをしていない。後者は打ち切り線を出す標本で計算した分が埋まった
        配列で、未計算のところは `NaN`。呼び出し側はそこだけ計算すればよい。
        標本は生存集合に必ず含まれる (標本のスコアが打ち切り線を決めるので、
        線を下回るのは標本の下位だけ) が、再計算すると枝刈りが効かないときに
        標本ぶんが丸ごと無駄になる。

        編集距離以外の成分 (`partial`) は既に確定している。編集距離による加算は
        `weights.phoneme` が上限なので、`partial + weights.phoneme` はその候補が
        取り得る**最大**スコアになる。上位 `needed` 件が確定していれば、その最低
        スコアを超えられない候補は順位に入れないので距離を計算しなくてよい。
        打ち切り線は上限の高い `_PROBE_CANDIDATES` 件だけ先に本計算して求める。

        **上位 `needed` 件の順位とスコアは全件計算と一致する** (厳密な打ち切り)。
        比率で固定的に削る案 (上限上位 5% だけ計算) は 30 クエリの実測で pun の
        top10 が 27/29 しか一致せず、真に上位の候補を落とした。上限は
        `weights.phoneme` ぶん緩いので、比率では「まだ線を超え得る候補」を
        切ってしまう。

        効き方は `weights.phoneme` に強く依存するので、割に合わない条件では
        丸ごと諦める (`_MAX_PHONEME_WEIGHT_FOR_PRUNING`)。重みが大きいと上限が
        緩くて候補が落ちず、標本の距離計算がただの上乗せになる。
        `_PROBE_CANDIDATES` を増やしても線は動かない — 上位数十件は 32768 件の
        標本にすでに入っているので、それ以上探しても打ち切り線が上がらない。

        標本の距離は捨てずに呼び出し側へ返す。生存集合には標本が必ず含まれる
        (線を決めたのが標本自身なので、落ちるのはその下位だけ) ため、
        計算し直すと標本ぶんを二度払うことになる。

        ANN 経路 (2000 件) では距離計算が 0.6ms しかないので、打ち切りの判定
        コストのほうが高くなる。`_PROBE_CANDIDATES` に届かない候補数では
        丸ごと省いて全件計算する。
        """
        count = partial.size
        if count <= _PROBE_CANDIDATES or needed >= count:
            return None, None
        if weights.phoneme > _MAX_PHONEME_WEIGHT_FOR_PRUNING:
            # 上限が緩すぎて候補が落ちない。標本の計算ぶんだけ損になる。
            return None, None

        upper = partial + weights.phoneme

        # 上限の高い順に標本を取り、その中で needed 位のスコアを打ち切り線にする。
        probe = np.argpartition(upper, count - _PROBE_CANDIDATES)[count - _PROBE_CANDIDATES :]
        probe_phonetic = self._phonetic_scores(rows[probe], query_ids)
        probe_scores = partial[probe] + weights.phoneme * probe_phonetic
        index = max(probe_scores.size - needed, 0)
        cutline = np.partition(probe_scores, index)[index]

        keep = upper >= cutline
        survivors = np.flatnonzero(keep)

        # 標本で計算した距離を生存集合の中での位置に移す。残りは NaN のままに
        # して呼び出し側に計算させる。生存位置は候補全体でのインデックスなので、
        # 累積和で「生存集合の中で何番目か」に写す。
        offsets = np.cumsum(keep) - 1
        probe_kept = keep[probe]
        phonetic = np.full(survivors.size, np.nan, dtype=probe_phonetic.dtype)
        phonetic[offsets[probe[probe_kept]]] = probe_phonetic[probe_kept]

        return survivors, phonetic

    def _phonetic_scores(self, rows: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
        """候補行の編集距離ベースの類似度。索引の CSR をそのまま Rust に渡す。"""
        blob, bounds, distance_ids = self.store.phoneme_csr
        distances = edit_distance_csr(query_ids, rows, blob, bounds, distance_ids)
        return _edit_similarity_array(distances, query_ids.size, self.store.phoneme_lengths(rows))

    def _space_scores(
        self,
        space: str,
        rows: np.ndarray,
        query_vectors: dict[str, np.ndarray],
    ) -> np.ndarray:
        """指定した空間で候補行のコサイン類似度を出す。

        候補が多いときは **全行と内積を取ってからマスクする**。mmap から行を
        飛び飛びに引く実体化 (`vectors[rows]`) はコストが行数に対して急に伸びる
        一方、全行走査は連続読みなので一定で済む。実測の分岐点は候補が全体の
        1〜2% あたり (coda 48 次元で全行 10ms、5% の 10 万行を引くと 20ms)。

        ANN 経路は候補 2000 件 (0.1%) なので分岐点の下、モーラ範囲の全走査は
        数十万件で上。どちらも通る経路なので件数で切り替える。
        """
        vectors = self.store.vectors(space)
        query = query_vectors[space]
        if rows.size >= self._FULL_SCAN_RATIO * len(self.store):
            return (vectors @ query)[rows]
        return vectors[rows] @ query

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


def _representative_rank(result: SearchResult) -> tuple[float, float, float, str]:
    """同音異表記の中で代表として残す優先度。大きいほど優先。

    SudachiDict では読みをそのまま書いたカタカナ見出し (「カカク」) が
    漢字表記 (「価格」) より低コストなことがある。コストだけで選ぶと
    「科学」の近傍が「カカク」になり、語として何を指すのかが読み取れない。
    見出しの情報量を優先し、コストは同程度の表記を選ぶときの手がかりに使う。

    同じ読みの候補は音韻類似度が等しいので、スコアの差は一般性の重みぶんしか
    生じない。だから情報量をスコアより先に見る。

    末尾の表層は決定性のためにある。「焦立とう」と「苛だとう」、「仕組」と
    「仕組み」のように 3 者がすべて同じ候補が実在し、これがないと代表が候補の
    到着順で決まる。候補の並びは候補生成の経路や編集距離の打ち切りで変わるので、
    同じクエリが実行のたびに違う表記を返してしまう
    (`tests/test_acceptance.py::test_edit_distance_pruning_does_not_change_ranking`
    がこの安定性を検証する)。

    `familiarity` は同点になりやすい。`COST_FAMILIAR` (5000) 以下のコストは
    すべて 1.0 に飽和するので、「仕組み」(4500) と「仕組」(5000) は一般性では
    分けられない。表層の符号順という決め方自体に意味はなく、**再現性だけを
    与えている**。
    """
    return (
        _surface_informativeness(result),
        result.score,
        result.familiarity,
        result.surface,
    )


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

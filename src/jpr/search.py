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
    containment_scan,
    edit_distance_csr,
    phoneme_ids,
    phonetic_similarity,
    vowel_skeleton_similarity,
    weighted_edit_distance,
)
from .embedding import embed
from .index import DEFAULT_CATEGORIES, Category, IndexEntry
from .phonology import Pronunciation, analyze_reading
from .phrase import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_CHUNK_CANDIDATES,
    DEFAULT_MAX_CHUNK_MORAS,
    DEFAULT_MAX_NODES_PER_SPAN,
    DEFAULT_MIN_CHUNK_SCORE,
    DEFAULT_NODE_BUDGET,
    PhraseCandidate,
    PhraseComposer,
    PhraseLattice,
)
from .reading import ReadingExtractor
from .store import INDEXED_SPACES, PhoneticStore


@dataclass(frozen=True)
class ScoreWeights:
    """最終スコアの重み。

    `phoneme` は精密な編集距離、`embedding` は ANN 空間での近さ、
    `mora` はモーラ数の一致、`coda` は語尾の一致、`vowel` は母音列 (韻) の一致。
    `familiarity` は語の一般性で、音韻的に同等な候補の順序を決めるのに使う。
    `containment` はクエリの音が候補に完全な形で入っていること
    (`distance.containment_ratio`)。
    """

    phoneme: float = 0.55
    embedding: float = 0.20
    mora: float = 0.05
    coda: float = 0.10
    vowel: float = 0.05
    familiarity: float = 0.05
    containment: float = 0.00

    def normalized(self) -> ScoreWeights:
        total = (
            self.phoneme
            + self.embedding
            + self.mora
            + self.coda
            + self.vowel
            + self.familiarity
            + self.containment
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
            containment=self.containment / total,
        )


#: 用途別のプリセット。
#:
#: pun        ダジャレ・なぞなぞ。音韻全体の近さと、知られた語であることを重視。
#: rhyme      韻を踏む語。語尾と母音列を重視し、語頭の一致は求めない。
#: mishearing 聞き間違い・ASR 補正。全体の音韻とリズムの一致を最重視。
#:
#: **`pun` の `familiarity` は 0.25 が上限** (v9 で 0.15 から上げた)。ダジャレは
#: 相手が知っている語でなければ成立しないので効かせたいが、上げすぎると音が
#: 離れる。実測で 0.30 では「パソコン」の 1 位が「家族」、「りんご」の 1 位が
#: 「前後」になった。0.35 では「乳首」に「一部」「記述」が入る。
#:
#: **旧指標 (Sudachi の連接コスト) では上げると悪化した。** あれは頻度ではなく
#: 知名度としては逆転していたので (`frequency` の項)、0.50 まで上げると「電話」
#: の上位が「デンダ」「レンガ」「ダンガ」で埋まった。**同じ重みでも指標が違えば
#: 結論が変わる。**
#:
#: `containment` は「クエリの音が完全な形で入っている」度合い。pun と
#: mishearing に入れて rhyme には入れない — **韻は語尾の一致を見るので、
#: 語頭に余分が付いているかどうかが関係しない**。「りんご」に対する
#: 「ラリンゴ」は韻としては「リンゴ」と同じ扱いでよく、包含を足すと
#: `coda` と競合して語尾の弱い包含語が混ざる。
#:
#: **0.12 は「上位に入るが埋め尽くさない」線。** 包含候補は編集距離を 1.0 と
#: 見るので (`_score_candidates`)、この重みは丸ごと上乗せになる。
#: `familiarity` 0.25 での上位 8 件に含まれる包含語の件数:
#:
#:     containment  0.06  0.09  0.12  0.15  0.18
#:     りんご         2     4     5     5     5
#:     電話          0     0     2     3     5
#:     パソコン        3     5     6     7     8
#:     眼鏡          0     0     3     5     6
#:
#: 0.18 では「パソコン」が 8/8 と埋め尽くされ、0.06 では「電話」「眼鏡」の
#: 包含語が上位 8 件に 1 つも入らない。
#:
#: **この値は `familiarity` と釣り合っている。** 頻度表は UniDic 短単位なので
#: 複合語 (「エア電話」「ノートパソコン」) は未収録で `UNKNOWN_FAMILIARITY`
#: に落ちる。一般性の重みを上げると包含語がそのぶん押されるので、両方を
#: 一緒に動かさなければならない。旧指標 (連接コスト) の頃は 0.06 だった。
PRESETS: dict[str, ScoreWeights] = {
    "pun": ScoreWeights(
        phoneme=0.50,
        embedding=0.15,
        mora=0.05,
        coda=0.10,
        vowel=0.05,
        familiarity=0.25,
        containment=0.12,
    ),
    # rhyme の一般性は 0.10 まで。上げると語尾の一致が崩れる — 実測で 0.15 の
    # 「眼鏡」に「姓」「ながら」、0.20 で「回」が入った。韻は語尾が合っている
    # ことが条件なので、一般性で押し込むと条件を満たさない語が混ざる。
    "rhyme": ScoreWeights(
        phoneme=0.20, embedding=0.10, mora=0.10, coda=0.35, vowel=0.20, familiarity=0.10
    ),
    "mishearing": ScoreWeights(
        phoneme=0.65,
        embedding=0.20,
        mora=0.10,
        coda=0.03,
        vowel=0.02,
        familiarity=0.00,
        containment=0.06,
    ),
}

DEFAULT_PRESET = "pun"

#: 候補生成から rerank に渡す件数の既定値。
#:
#: 202 万語では 3 モーラ語のクエリに対しコサイン 0.91 以上の候補が 400 件を超える。
#: k=400 では「乳首」に対する「手首」のような明らかな近傍が候補にすら入らず、
#: 稀語ばかりが並んだ。実測では k=5000 で「手首」が 1 位に来る。
#: 候補数を増やすと rerank のコストが線形に増えるので、品質が飽和する手前で止める。
#:
#: **候補生成は索引全体との内積なので、ここを増やしても生成側は重くならない**
#: (`PhoneticSearcher._top_candidates`)。伸びるのは rerank だけ。ANN を使って
#: いた頃は k を上げると探索そのものが線形以上に遅くなり、かつ Top-K に入らない
#: 語を rerank が拾えないという取りこぼしがあったが (k=5000 で総当たりとの
#: top10 一致率 0.73)、いまは内積の順位が正確なので k の意味は
#: 「rerank に何件見せるか」だけになった。
#:
#: **5000 -> 8000 は量子化 (int8) の誤差を吸収するため。** 内積の誤差は最大
#: 0.012 で、Top-K 境界のスコア差 (0.0002 程度) より大きいので順位が入れ替わる。
#: float32 の Top-5000 を正解としたときの recall は k=5000 で 0.919、
#: k=6000 で 0.977、**k=8000 で 0.997**。int8 の候補生成は k を増やしても
#: 重くならないので (内積は全行に対して取る)、広げて取りこぼしを消すほうが得。
DEFAULT_CANDIDATES = 8000

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

#: 同音異表記の編集距離を代表 1 件に畳む処理 (`_group_representatives`) を
#: 試みる最小候補数。
#:
#: 畳み込み自体のコストは `group_ids` の gather と累積和で候補数に線形。
#: ANN 経路 (数千件) では距離計算が 1ms 未満しかなく判定のほうが高くつくので、
#: 全走査経路の規模でだけ効かせる。値そのものに敏感さはない — 損益分岐は
#: 数千件のどこかで、全走査の候補 (数十万件) とは 2 桁離れている。
_GROUP_DEDUPE_MIN_CANDIDATES = 8192


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
    #: クエリの音が完全な形で入っている度合い (`distance.containment_ratio`)。
    #: 含まないなら 0.0、含むならクエリが候補の音素列に占める割合。
    containment: float
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
    containment: np.ndarray

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
        candidate_space: str = "phonetic",
    ) -> None:
        self.store = store
        self.candidate_space = candidate_space
        self._extractor = extractor
        self._category_ids = store.category_ids
        self._mora_counts = store.mora_counts
        self._composer: PhraseComposer | None = None

    @property
    def extractor(self) -> ReadingExtractor:
        """読み取得器。Sudachi のロードが重いので初回参照まで遅延させる。"""
        if self._extractor is None:
            self._extractor = ReadingExtractor()
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
        # 索引に載っている空間だけを作る。`consonant` と `rhythm` は索引が
        # 持たないので (`store.INDEXED_SPACES`)、作っても引く相手がいない。
        query_vectors = embed(pronunciation, INDEXED_SPACES)

        # 候補生成には 2 つの経路がある。モーラ範囲を指定したときだけ
        # ANN を迂回して母集団を全走査する (`_scan_candidates` 参照)。
        bounds = self._mora_bounds(min_mora=min_mora, max_mora=max_mora)
        if bounds is None:
            rows, embedding_scores = self._top_candidates(query_vectors, candidates, pronunciation)
        else:
            rows, embedding_scores = self._scan_candidates(query_vectors, bounds)
        if rows.size == 0:
            return pronunciation, []

        # 包含候補を拾う。既定経路では**候補生成に合流させる** — 包含は
        # phonetic 空間の近さと相関しないので Top-K に入らない
        # (`_containment` 参照)。範囲指定の経路では母集団が既に区間の全行なので
        # 合流させる先がなく、占有率を配るだけでよい。
        containment_rows, containment_ratios = (
            self._containment(pronunciation)
            if active_weights.containment > 0.0
            else (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32))
        )

        if bounds is not None and containment_rows.size:
            # **区間の外へは出さない。** 呼び出し側がモーラ範囲を明示している
            # 経路なので、包含だからといって範囲外の語を返してよい理由がない。
            # 下流も `rows` が区間の `arange` であることを前提にしている
            # (`_apply_cheap_filters` と `_space_scores` のスライス読み)。
            inside = (containment_rows >= bounds[0]) & (containment_rows < bounds[1])
            containment_rows = containment_rows[inside]
            containment_ratios = containment_ratios[inside]

        rows, embedding_scores = self._apply_cheap_filters(
            rows,
            embedding_scores,
            pronunciation,
            categories,
            bounds=bounds,
        )

        if bounds is None and containment_rows.size:
            # 合流させる行にも同じ条件を掛ける。合流をカテゴリフィルタの後に
            # 置くのは、前だと全走査経路のスライス読みが崩れるため。
            containment_rows, containment_ratios = self._apply_cheap_filters(
                containment_rows,
                containment_ratios,
                pronunciation,
                categories,
                # モーラ帯を適用しない。包含は余分の多さが本質なので、帯で切ると
                # 「りんご」に対する「リンゴジュース」(6 モーラ) が落ちる。歯止めは
                # 占有率がスコア側で掛ける減点のほう。
                apply_mora_gap=False,
            )

        rows, embedding_scores, containment = self._merge_containment(
            rows, embedding_scores, containment_rows, containment_ratios
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
            containment=containment,
            pronunciation=pronunciation,
            query_vectors=query_vectors,
            weights=active_weights,
            needed=needed,
            bounds=bounds,
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
        bounds = self._mora_bounds(min_mora=min_mora, max_mora=max_mora)
        return len(self.store) if bounds is None else bounds[1] - bounds[0]

    # --- 段 1: 候補生成 ----------------------------------------------------

    def _mora_bounds(
        self,
        *,
        min_mora: int | None,
        max_mora: int | None,
    ) -> tuple[int, int] | None:
        """候補生成の母集団になる行の連続区間 [start, end)。

        `None` は「絞り込みが指定されていない」= ANN 経路でよい、を表す。

        索引の行はモーラ数の昇順に並んでいるので (`store._locality_order`)、
        範囲は必ず連続区間になり二分探索で切り出せる (`store.mora_range`)。
        以前はここで全行の真偽マスクを立てていたが、マスク経由だと下流の
        すべてが fancy indexing になり、97 万行の選抜で連続読みの 8 倍
        (48ms 対 6ms) を払っていた。

        モーラ以外の軸で母集団を切りたくなったら、その軸が連続区間に
        できるか (格納順に含められるか) をまず考える。できない軸は
        `_apply_cheap_filters` 側でマスクとして積む。
        """
        if min_mora is None and max_mora is None:
            return None
        if min_mora is not None and max_mora is not None and min_mora > max_mora:
            raise ValueError(f"モーラ範囲が逆転しています: {min_mora} > {max_mora}")
        return self.store.mora_range(min_mora, max_mora)

    def _top_candidates(
        self,
        query_vectors: dict[str, np.ndarray],
        candidates: int,
        pronunciation: Pronunciation,
    ) -> tuple[np.ndarray, np.ndarray]:
        """索引全体と内積を取り、上位 `candidates` 件の行と類似度を得る。

        **HNSW を使わない。** hnswlib の `load_index` はグラフをヒープに
        実体化するので (mmap のオプションが無い)、匿名メモリを 652MB 使う
        (core / 111 万語)。内訳はグラフ 346MB と**ベクトルの複製 306MB** で、
        後者は `vectors-phonetic.npy` と同じデータの二重持ち。`M` を下げても
        複製は消えないので 408MB (M=4) が下限で、そこでは recall が 0.758 まで
        落ちる。2GB の本番が OOM killed になった主因がこれ。

        総当たりは mmap をそのまま読むので匿名メモリが増えない (+4MB)。
        速度も core では負けない — 実測で内積 8〜9ms + Top-K 抽出 3〜7ms の
        11〜16ms に対し、HNSW (M=24) は 16ms。**recall は 1.0 になるので
        品質は上がる** (M=24 で 0.986)。

        かつて「総当たりは ANN より遅い (21〜30ms 対 15ms)」と記録していたが、
        あれは full (202 万語) での測定。core (111 万語) では逆転している。
        **語数が変われば測り直す。**

        内積と Top-K は Rust が担う (`store.top_rows`)。実測 (full 202 万語、
        プロセスを分けて 20 回の最小値):

        | | 内積 | Top-K |
        |---|---|---|
        | NumPy float32 (BLAS + argpartition) | 15.4〜26.0ms | 9.0〜15.5ms |
        | Rust int8 | 16.8〜28.8ms | 2.6〜10.0ms |

        **内積は BLAS と互角** — int8 で読むバイト数が 1/4 になっても、
        走査は DRAM 帯域で決まり、int8 の積和が BLAS の SIMD された f32 積和
        よりスループットで劣るぶんと相殺される。速度で得しているのは
        Top-K だけで、**量子化の主目的はサイズ** (索引 1.64GB -> 508MB)。

        **内積を取るのはグループ行列** (v5)。同音異表記はベクトルが同一なので、
        行単位で Top-K を取ると上位が同じ音の異表記で埋まる。グループで取って
        から行へ展開すれば、`candidates` 件が「異なる音素列 `candidates` 個」を
        意味するようになり、rerank に渡る音の多様性が上がる。

        **内積を取るのはモーラ帯だけ** (v6)。この経路の候補は直後に
        `_apply_cheap_filters` が `_MAX_MORA_GAP` で削るので、帯の外の行は
        内積を取っても必ず捨てられる。行はモーラ数順に並んでいるので
        (`store._locality_order`) 帯は連続区間で、スライス 1 本で切れる。
        実測 (full 202 万語) で 3 モーラのクエリは行が 49% になり、候補生成が
        **8.0ms -> 4.0ms**。7 モーラでは帯が広く 86% にしか減らないので
        7.2ms -> 6.2ms に留まる — **効きはクエリのモーラ数に依存する**。

        **近似ではない。** 落とすのはどのみち後段が捨てる行なので、上位の
        順位とスコアは全行に内積を取った場合と一致する (実測で Top-200 が
        完全一致、`tests/test_acceptance.py::test_mora_band_does_not_change_ranking`
        が検証する)。だから `_apply_cheap_filters` 側のギャップ判定と
        **同じ `_MAX_MORA_GAP` を使わなければならない** — ここを緩めると
        後段が捨てる行に内積を取り、狭めると順位が変わる。
        """
        store = self.store
        wanted = min(candidates, store.group_count)
        query = query_vectors[self.candidate_space]

        # 後段のギャップ判定が残す帯だけをグループの連続区間に落とす。
        row_start, row_end = store.mora_range(
            pronunciation.mora_count - self._MAX_MORA_GAP,
            pronunciation.mora_count + self._MAX_MORA_GAP,
        )
        if row_start >= row_end:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        group_start, group_end = store.group_mora_range_of_rows(row_start, row_end)

        groups, scores = store.top_groups(
            self.candidate_space, query, wanted, group_start, group_end
        )
        return self._expand_groups(groups, scores)

    def _containment(self, pronunciation: Pronunciation) -> tuple[np.ndarray, np.ndarray]:
        """クエリを完全な形で含むグループを索引全体から拾い、行に展開する。

        戻り値は (行, 占有率)。行は昇順。

        **候補生成に合流させる経路**であって、後段のフィルタではない。包含は
        phonetic 空間の近さと相関しないので、Top-K に入った候補を絞るだけでは
        拾えない — 実測で「りんご」を含む 204 グループのうち Top-8000 に入るのは
        48 件、モーラ帯に限っても 123 件のうち 48 件しかなかった。だから
        `_scan_candidates` と同じく候補生成そのものを足す。

        **モーラ帯 (`_MAX_MORA_GAP`) の外も拾う。** 包含は「余分がどれだけ
        付いているか」が本質なので、帯で切ると長い複合語が落ちる (「りんご」
        3 モーラに対する「リンゴジュース」6 モーラ)。代わりに占有率が余分の
        多さを減点するので、帯の代わりの歯止めはスコア側にある。

        走査は索引全体の音素 CSR (full で 17MB / 146 万グループ) に対して行い、
        **実測 5.4〜7.6ms** (load 6、15 回の最小値)。既定経路の内積 (4ms) と
        同程度だが、**検索全体では埋もれる** — 8 クエリの中央値が包含あり
        20.3ms・なし 20.3ms で差が出なかった。rayon が走査を並列化するので、
        rerank と結果の実体化に隠れる。

        **1 グループを 1 タスクにしてはいけない。** 最初そう書いたら同じ走査が
        19〜37ms かかった (rayon のスケジューリングが支配的)。1 件の判定は音素
        12 個の比較で数十ナノ秒しかないので、8192 件ずつの塊にする
        (`rust/src/lib.rs` の `containment_scan`)。
        """
        blob, bounds, distance_ids = self.store.phoneme_csr
        groups, ratios = containment_scan(
            phoneme_ids(pronunciation.phonemes),
            blob,
            bounds,
            distance_ids,
            0,
            self.store.group_count,
        )
        if groups.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        return self._expand_groups(groups, ratios)

    def _expand_groups(
        self, groups: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """グループとそのスコアを、所属する行とスコアに展開する。

        グループの行は連続区間なので (`store.group_starts`)、区間長を数えて
        `repeat` でスコアを配り、区間を連結して行を作る。

        **行は昇順に並べ直す。** 同音異表記の畳み込み (`_group_representatives`)
        と全走査経路のスライス読み (`_space_scores`) が昇順を前提にしている。
        グループはスコア降順で返るので、そのまま繋ぐと行が飛び飛びになる。
        """
        if groups.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=scores.dtype)
        starts = self.store.group_starts
        begins = starts[groups]
        lengths = starts[groups + 1] - begins
        # 各グループの先頭行を、そのグループの行数だけ繰り返してから
        # グループ内の連番を足す (連続区間の連結)。
        repeated = np.repeat(begins, lengths)
        offsets = np.arange(repeated.size, dtype=np.int64) - np.repeat(
            np.cumsum(lengths) - lengths, lengths
        )
        rows = repeated + offsets
        row_scores = np.repeat(scores, lengths)
        order = np.argsort(rows, kind="stable")
        return rows[order], row_scores[order]

    def _scan_candidates(
        self,
        query_vectors: dict[str, np.ndarray],
        bounds: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """モーラ範囲の連続区間を全走査し、内積を直接求める。

        ANN を使わない経路。HNSW は phonetic 空間の Top-K しか返さないので、
        モーラ数の違う語は近傍の外に沈んで候補にすら入らない (`search` の
        docstring 参照)。ここでは母集団の全行に対して内積を取るので、
        近傍順位に関わらず rerank の土俵に載る。

        行がモーラ数順に並んでいるので、母集団はスライス 1 本で読める。
        並べ替える前は「散らばった行の fancy indexing (97 万行で 191ms) を
        避けるために全 202 万行と内積を取ってからマスクする」という迂回を
        していたが (48ms)、連続区間なら範囲の行だけの連続読みで済む (6ms)。

        **内積を取るのは範囲のグループぶんだけ** (v5)。グループもモーラ数順に
        並ぶので (行がそう並んでおり、グループは行の連続区間なので) 範囲は
        やはりスライス 1 本で、走査する行数が 28% 減る。スコアは所属する行へ
        配る — rerank は行ごとの一般性やカテゴリを見るので、候補は行のまま
        渡す必要がある。
        """
        start, end = bounds
        if start >= end:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        group_start, group_end = self.store.group_mora_range_of_rows(start, end)
        # 量子化スケールを戻したコサイン類似度が返るので、Top-K 経路と
        # 同じ尺度で揃う。
        group_scores = self.store.dot_groups(
            self.candidate_space, query_vectors[self.candidate_space], group_start, group_end
        )
        rows = np.arange(start, end, dtype=np.int64)
        # 行 -> グループ -> スコア。グループ番号は区間の先頭を 0 に寄せる。
        scores = group_scores[self.store.group_ids[start:end] - group_start]
        return rows, scores

    def _merge_containment(
        self,
        rows: np.ndarray,
        embedding_scores: np.ndarray,
        containment_rows: np.ndarray,
        containment_ratios: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """既定経路の候補と包含候補の和集合を取り、行ごとの占有率を並べる。

        戻り値は (行, 内積スコア, 占有率) で、行は昇順。**昇順は後段の前提**
        (`_group_representatives` と `_space_scores` が隣接比較で重複を畳む)。

        包含だけで拾われた行には内積スコアが無い。改めて引くと 1 行ごとの
        fancy indexing が増えるので、**0.0 を入れて `embedding` 成分を捨てる**。
        包含候補は占有率と編集距離で評価されるべきもので、内積の値は
        `_top_candidates` が既に「上位ではない」と判定している。
        """
        if containment_rows.size == 0:
            return rows, embedding_scores, np.zeros(rows.size, dtype=np.float32)
        if rows.size == 0:
            return (
                containment_rows,
                np.zeros(containment_rows.size, dtype=embedding_scores.dtype),
                containment_ratios,
            )

        merged = np.union1d(rows, containment_rows)
        # 既定経路の内積スコアを新しい並びに配る。`rows` は昇順なので
        # `searchsorted` が使える。
        positions = np.searchsorted(merged, rows)
        scores = np.zeros(merged.size, dtype=embedding_scores.dtype)
        scores[positions] = embedding_scores
        containment = np.zeros(merged.size, dtype=np.float32)
        containment[np.searchsorted(merged, containment_rows)] = containment_ratios
        return merged, scores, containment

    def _apply_cheap_filters(
        self,
        rows: np.ndarray,
        embedding_scores: np.ndarray,
        pronunciation: Pronunciation,
        categories: Iterable[Category] | None,
        *,
        bounds: tuple[int, int] | None = None,
        apply_mora_gap: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """編集距離の前に、配列演算で済む条件で候補を削る。

        `bounds` は全走査経路の連続区間。渡されたときは `rows` がその区間の
        `arange` そのものなので、列配列をスライスで読める (97 万行の
        fancy indexing を 1 回省く)。

        `apply_mora_gap=False` はモーラ差の安全網を外す。包含候補
        (`_containment`) だけが通る — あちらは余分の多さを占有率で減点するので、
        帯で切る必要がない。
        """
        wanted = tuple(categories) if categories is not None else tuple(DEFAULT_CATEGORIES)
        wanted_ids = [self.store.category_id(c) for c in wanted]
        wanted_ids = [i for i in wanted_ids if i >= 0]
        if not wanted_ids:
            return rows[:0], embedding_scores[:0]

        if bounds is not None:
            candidate_categories = self._category_ids[bounds[0] : bounds[1]]
        else:
            candidate_categories = self._category_ids[rows]
        # ID の種類は数個しかないので、`isin` (ソート + 二分探索) ではなく
        # 真偽表を引く。97 万行で 6.3ms -> 1ms 未満。
        lookup = np.zeros(max(candidate_categories.max(initial=0), max(wanted_ids)) + 1, dtype=bool)
        lookup[wanted_ids] = True
        keep = lookup[candidate_categories]

        # モーラ数が大きく離れた語は音韻的な近さとして意味がないので落とす。
        # ただしこれは ANN の候補生成が粗いことを補う安全網なので、呼び出し側が
        # モーラ範囲を明示したとき (`bounds` あり) は適用しない。3 モーラの
        # クエリに 7 モーラを要求すると、ギャップ 4 で全件落ちてしまう。
        if bounds is None and apply_mora_gap:
            gap = np.abs(self._mora_counts[rows].astype(np.int32) - pronunciation.mora_count)
            keep &= gap <= self._MAX_MORA_GAP

        return rows[keep], embedding_scores[keep]

    # --- 段 2: 精密な音韻距離で rerank ------------------------------------

    def _score_candidates(
        self,
        *,
        rows: np.ndarray,
        embedding_scores: np.ndarray,
        containment: np.ndarray,
        pronunciation: Pronunciation,
        query_vectors: dict[str, np.ndarray],
        weights: ScoreWeights,
        needed: int,
        bounds: tuple[int, int] | None = None,
    ) -> _ScoredCandidates:
        """候補のスコアと内訳を配列で求める。

        `needed` は後段が必要とする件数。これを使って編集距離を計算する候補を
        削るが (`_survivors`)、**上位 `needed` 件の順位とスコアは全件計算した
        場合と一致する**。`bounds` は全走査経路の母集団の連続区間で、
        rerank 用空間の内積をスライス読みにするために回す (`_space_scores`)。
        """
        store = self.store
        query_ids = phoneme_ids(pronunciation.phonemes)

        # ベクトルの内積とモーラ数・一般性は配列演算でまとめて出す。
        # 編集距離より 2 桁安いので、絞り込む前に全候補ぶん出しておく。
        coda = np.clip(self._space_scores("coda", rows, query_vectors, bounds), 0.0, None)
        # 母音軸はベクトル空間ではなく母音骨格の編集距離 (v8)。母音の類似は
        # 列の照合であって、プーリングした内積では長さの違いが消える
        # (`distance.vowel_skeleton_similarity` の項を参照)。骨格は音素列の
        # 半分程度の長さしかないので、DP でも全候補に払える。
        vowel = self._vowel_scores(rows, phoneme_ids(pronunciation.vowel_skeleton))
        embedding = np.clip(embedding_scores, 0.0, None)
        candidate_moras = store.mora_counts[rows].astype(np.int32)
        mora = _mora_similarity_array(pronunciation.mora_count, candidate_moras)
        familiarity = store.familiarities[rows].astype(np.float64)

        partial = (
            weights.embedding * embedding
            + weights.mora * mora
            + weights.coda * coda
            + weights.vowel * vowel
            + weights.familiarity * familiarity
            + weights.containment * containment
        )

        # 編集距離だけは候補数に対して重い (53 万件で 48ms、他の成分の合計の
        # 1.4 倍)。順位に入り得ない候補には計算しない。
        survivors, phonetic = self._survivors(rows, partial, query_ids, weights, needed)
        if survivors is not None:
            rows = rows[survivors]
            partial = partial[survivors]
            coda, vowel = coda[survivors], vowel[survivors]
            embedding, familiarity = embedding[survivors], familiarity[survivors]
            containment = containment[survivors]

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

        # 包含が成立した候補は編集距離を 1.0 と見る。**同じ「余分」を 2 つの
        # 成分が二重に数えるのを避けるため。** クエリが完全な形で入っている以上
        # 一致部分の距離は 0 で、距離が減っているのは余分ぶんだけ。その余分は
        # `containment` の占有率がすでに減点している。
        #
        # 二重に数えると加点が打ち消される。実測で「りんご」->「ラリンゴ」は
        # 占有率 0.714 で +0.093 得るのに、編集距離が 0.735 と非包含語
        # (0.93〜0.98) より低くて -0.087 失い、重みを 0.25 まで上げないと
        # 順位が動かなかった。**重みで押すのではなく役割を分ける** —
        # 一致したかどうかは `phonetic`、余分の多寡は `containment` が持つ。
        #
        # `_survivors` の枝刈りは `weights.phoneme` を距離の上限として使うので、
        # 1.0 に固定してもその上限をちょうど達成するだけで、順位の保証は崩れない。
        if weights.containment > 0.0:
            phonetic = np.where(containment > 0.0, 1.0, phonetic)

        scores = partial + weights.phoneme * phonetic

        return _ScoredCandidates(
            rows=rows,
            scores=scores,
            phonetic=phonetic,
            embedding=embedding,
            coda=coda,
            vowel=vowel,
            familiarity=familiarity,
            containment=containment,
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
        # 打ち切り線は標本のスコア分布だけから決まるので標本の並び順に意味はない。
        # 昇順に直しておくと標本の距離計算でも同音異表記の畳み込みが効く
        # (`_group_representatives` は行の昇順を要求する)。
        probe.sort()
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
        return self._csr_similarity(
            rows, query_ids, self.store.phoneme_csr, self.store.phoneme_lengths(rows)
        )

    def _vowel_scores(self, rows: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
        """候補行の母音骨格の類似度。`_phonetic_scores` の母音版。

        DP・コスト表・正規化は音素列と共有し、読む CSR だけが違う
        (`store.vowel_csr`)。記号版の参照実装は
        `distance.vowel_skeleton_similarity` で、**2 者は同じ値を返さなければ
        ならない** (`tests/test_search.py` が一致を検証する)。
        """
        return self._csr_similarity(
            rows, query_ids, self.store.vowel_csr, self.store.vowel_lengths(rows)
        )

    def _csr_similarity(
        self,
        rows: np.ndarray,
        query_ids: np.ndarray,
        csr: tuple[np.ndarray, np.ndarray, np.ndarray],
        lengths: np.ndarray,
    ) -> np.ndarray:
        """候補行の CSR 列に対する編集距離を類似度に直す。

        音素列と母音骨格で共通の経路。**Rust には母音専用のコードが無い** —
        骨格の記号はすべて音素なので、コスト表・正規化・グループ代表の
        畳み込みをそのまま共有する (`store.vowel_csr` の項)。読む CSR と
        正規化に使う長さだけが軸ごとに違う。

        同じ音素列の行 (同音異表記) には同じ距離しか出ないので、代表 1 行だけ
        計算して残りへ配る (`_group_representatives`)。4〜6 モーラ帯の候補では
        ユニークな音素列が 58.8% しかなく、距離計算が 4 割減る。
        """
        blob, bounds, distance_ids = csr
        groups = self.store.group_ids[rows].astype(np.int64)
        representatives = self._group_representatives(groups)
        if representatives is None:
            distances = edit_distance_csr(query_ids, groups, blob, bounds, distance_ids)
        else:
            leaders, inverse = representatives
            distances = edit_distance_csr(query_ids, leaders, blob, bounds, distance_ids)[inverse]
        return _edit_similarity_array(distances, query_ids.size, lengths)

    def _group_representatives(self, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """候補のグループ列を重複のない代表に畳む。

        戻り値は (代表グループ, 各候補の代表位置)。代表の距離を `[inverse]` で
        gather すれば全候補の距離になる。畳めない・畳む価値がないときは `None`。

        索引の行は同じ音素列が隣接するよう並んでいるので (`store.group_ids`
        は単調非減少)、候補行が昇順ならグループ列も昇順になり、代表判定は
        隣接比較 1 回で済む。昇順でないのは `_survivors` の標本抽出だけで、
        そこは抽出後にソートして渡してくる。ここで昇順を確かめるのは、
        その前提が崩れたときに黙って違う候補の距離を配らないため
        (確認は候補数に線形で軽い)。
        """
        if groups.size < max(_GROUP_DEDUPE_MIN_CANDIDATES, 2):
            return None
        return _collapse_sorted_runs(groups)

    def _space_scores(
        self,
        space: str,
        rows: np.ndarray,
        query_vectors: dict[str, np.ndarray],
        bounds: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """指定した空間で候補行のコサイン類似度を出す。

        候補が多いときは **母集団の連続区間ぜんぶと内積を取ってから選ぶ**。
        mmap から行を飛び飛びに引く実体化 (`vectors[rows]`) はコストが行数に
        対して急に伸びる一方、連続読みは帯域で押し切れる。実測の分岐点は
        候補が母集団の 1〜2% あたり (coda 48 次元で全行 10ms、5% の 10 万行を
        引くと 20ms)。

        `bounds` は全走査経路の母集団 (モーラ範囲の連続区間)。カテゴリフィルタ
        後の候補は区間の 5 割強を占めるので分岐点のはるか上、ANN 経路
        (`bounds=None`, 候補 2000 件 = 全体の 0.1%) は下。どちらも通るので
        件数で切り替える。

        **引くのはグループのベクトル** (v5)。候補行をグループに写してから
        内積を取り、結果を行へ配り直す。選抜経路 (`dot_selected_groups`) では
        重複するグループを畳めるので、実体化する行数がそのぶん減る。
        """
        store = self.store
        query = query_vectors[space]
        groups = store.group_ids[rows]
        if bounds is not None:
            group_start, group_end = store.group_mora_range_of_rows(*bounds)
        else:
            group_start, group_end = 0, store.group_count

        # 全走査の判定は候補行数で行う。**候補のグループ数を数えるために
        # `np.unique` を呼んではいけない** — ハッシュ化が候補数に効き、
        # 97 万候補の 2 空間で 378ms かかって全走査経路が 2 倍に伸びた。
        # 行数はグループ数の上界なので、判定にはこれで足りる。
        if rows.size >= self._FULL_SCAN_RATIO * (group_end - group_start):
            scores = store.dot_groups(space, query, group_start, group_end)
            return scores[groups - group_start]

        # 選抜経路。候補が少ないときだけ通るので、ここで重複を畳む。
        #
        # 候補行は昇順で渡る (`_expand_groups` は並べ替えて返し、`_scan_candidates`
        # は `arange` を返す)。昇順ならグループ列も昇順なので、隣接比較だけで
        # 代表が取れる。**その前提が崩れたら畳まない** — 黙って別のグループの
        # スコアを配るより、重複ぶん引き直すほうがいい。
        collapsed = _collapse_sorted_runs(groups)
        if collapsed is None:
            return store.dot_selected_groups(space, query, groups)
        leaders, inverse = collapsed
        return store.dot_selected_groups(space, query, leaders)[inverse]

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
                    containment=round(float(scored.containment[position]), 4),
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

    # --- 分割合成 ---------------------------------------------------------

    @property
    def composer(self) -> PhraseComposer:
        """分割合成器 (`phrase.PhraseComposer`)。

        モーラ数ごとの行の選抜を抱えるので (実測 150〜330ms)、窓ごとに作り直さず
        searcher に持たせて使い回す。常駐する窓 (Web / MCP) ではこれが体感を
        分ける。カテゴリを変えて合成したいときは `PhraseComposer` を直接作る。
        """
        if self._composer is None:
            self._composer = PhraseComposer(self.store)
        return self._composer

    def compose(
        self,
        text: str,
        *,
        limit: int = 10,
        max_chunk_moras: int = DEFAULT_MAX_CHUNK_MORAS,
        chunk_candidates: int = DEFAULT_CHUNK_CANDIDATES,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        min_chunk_score: float = DEFAULT_MIN_CHUNK_SCORE,
        allow_particles: bool = True,
    ) -> tuple[Pronunciation, list[PhraseCandidate]]:
        """`text` の音を「複数の語 + 助詞」の連なりで組み立てる。

        長い入力に音が近い**単一の語**は辞書に無いので、通常の `search` では
        答えが返らない。こちらは入力をモーラ境界で区間に切り、区間ごとに語を
        当てて繋ぐ (`phrase.py` 参照)。空耳・替え歌の経路。

        読みの解析はここで行うので、漢字を含むテキストをそのまま渡せる
        (`PhraseComposer.compose` 自体は Sudachi を持たない)。
        """
        return self.composer.compose(
            text,
            pronunciation=self.pronounce(text),
            limit=limit,
            max_chunk_moras=max_chunk_moras,
            chunk_candidates=chunk_candidates,
            beam_width=beam_width,
            min_chunk_score=min_chunk_score,
            allow_particles=allow_particles,
        )

    def lattice(
        self,
        text: str,
        *,
        node_budget: int = DEFAULT_NODE_BUDGET,
        max_nodes_per_span: int = DEFAULT_MAX_NODES_PER_SPAN,
        max_chunk_moras: int = DEFAULT_MAX_CHUNK_MORAS,
        chunk_candidates: int = DEFAULT_CHUNK_CANDIDATES,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        min_chunk_score: float = DEFAULT_MIN_CHUNK_SCORE,
        allow_particles: bool = True,
    ) -> tuple[Pronunciation, PhraseLattice]:
        """`compose` と同じ経路集合を 1 枚の DAG に畳んで返す。

        候補を並べると同じ語が何度も出る (実測で区間の 65〜77% が重複)。
        ノードに畳むと 1 度しか現れず、分岐だけが見える。`node_budget` に
        届くまでビーム幅を広げる (`phrase.PhraseComposer.lattice`)。
        """
        return self.composer.lattice(
            text,
            pronunciation=self.pronounce(text),
            node_budget=node_budget,
            max_nodes_per_span=max_nodes_per_span,
            max_chunk_moras=max_chunk_moras,
            chunk_candidates=chunk_candidates,
            beam_width=beam_width,
            min_chunk_score=min_chunk_score,
            allow_particles=allow_particles,
        )


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
    # 母音は埋め込み空間ではなく骨格列の照合 (v8)。検索の rerank と同じ定義を
    # 出す — 名前が同じ軸が窓によって違う値を返すと、内訳から順位を検算できない。
    spaces["vowel"] = round(vowel_skeleton_similarity(a, b), 4)

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


def _collapse_sorted_runs(values: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """昇順の配列を重複のない代表に畳む。

    戻り値は (代表, 各要素の代表位置)。代表側で求めた値を `[inverse]` で
    gather すれば元の並びに戻る。**昇順でない、または畳める重複が無いときは
    `None`** — 呼び出し側は元の配列をそのまま使えばよい。

    `np.unique` を使わないのは、ハッシュ化 (実際にはソート) が要素数に効く
    ため。97 万候補 x 2 空間で 378ms かかり、全走査経路が 2 倍に伸びた。
    昇順なら同じ値は隣接するので、隣接比較 1 回で代表が取れる。

    昇順の確認をここで行うのは、前提が崩れたときに黙って別の要素の値を
    配らないため (候補数に線形で軽い)。
    """
    if values.size == 0:
        return None
    deltas = np.diff(values)
    if np.any(deltas < 0):
        return None
    first = np.empty(values.size, dtype=bool)
    first[0] = True
    np.not_equal(deltas, 0, out=first[1:])
    leaders = values[first]
    if leaders.size == values.size:
        return None
    return leaders, np.cumsum(first) - 1


def _mora_similarity_array(query_moras: int, candidates: np.ndarray) -> np.ndarray:
    """モーラ数の近さ。同数なら 1.0、離れるにつれ線形に下がる。"""
    gap = np.abs(candidates - query_moras).astype(np.float64)
    scale = np.maximum(np.maximum(candidates, query_moras), 1).astype(np.float64)
    return np.clip(1.0 - gap / scale, 0.0, 1.0)


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

    `familiarity` は同点になりやすい。頻度表に無い語はすべて
    `frequency.UNKNOWN_FAMILIARITY` で並ぶので、同音異表記が揃って未収録だと
    一般性では分けられない。表層の符号順という決め方自体に意味はなく、
    **再現性だけを与えている**。
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

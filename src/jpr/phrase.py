"""長い入力を「複数の語 + 助詞」の連なりに置き換える (空耳・替え歌の経路)。

既存の検索 (`search.py`) は 1 語を 1 語に写す。だが「ワタシノナマエハ」のような
長い入力に音が近い**単一の語**は辞書に存在しない — 音韻的に近い長さの語はあっても
意味を持つ列にならないので、返せる答えが無い。人が空耳を作るときにやっているのは
入力を区間に切り、区間ごとに別の語を当てて、繋ぎに助詞を挟むという操作なので、
ここではそれをそのまま検索問題として解く。

    ワタシ | ノ | ナマエ | ハ
      ↓     ↓     ↓      ↓
     渡し   の   名前     は

3 つの部品でできている。

1. **区間の切り出し** — 入力のモーラ列から連続する区間 (i, j) を列挙する。
   モーラ境界でしか切らない。音素の途中 (子音と母音の間) で切ると、どちらの
   側も発音できない断片になり、辞書の語と照合しても意味のある距離が出ない。
2. **区間ごとの照合** — 区間の音素列と、**同じモーラ数の語**を weighted
   phonetic edit distance で突き合わせる (`search.py` の rerank と同じ距離)。
3. **連結** — 区間の並びを DP + ビームで繋ぎ、全体のスコアが高い列を選ぶ。

## なぜ区間の母集団を「同じモーラ数」に限るのか

区間長 ±1 のプールまで見ると照合が 3 倍を超える。13 モーラの入力 (46 区間) の
実測で、±1 まで見ると延べ 1936 万行・3559ms かかるのに対し、同じモーラ数だけに
限れば延べ 303 万行・523ms で済む。

**長さの伸縮は別の場所が担っているので、ここで払う必要がない。** 区間の切り方
そのものが可変長なので、「3 モーラの区間に 4 モーラの語を当てる」ことは
「4 モーラの区間を切って 4 モーラの語を当てる」ことでほぼ代替できる。区間内の
細かい伸縮 (促音の有無など) は編集距離が同じモーラ数の中で吸収する。

## なぜ助詞を内蔵の表で持つのか

**索引に助詞が無い。** `index.EXCLUDED_POS` が助詞・助動詞を落としているので、
索引を引いても 0 件しか返らない (実測で「の」「を」「に」「が」「は」「で」
「と」すべて 0 件)。これは索引側の正しい判断で、単語検索の結果に助詞が出ても
ダジャレの候補にならない。

さらに **索引には 1 モーラの語が 1 件も無い** (実測: 2 モーラ 38210 件に対し
1 モーラ 0 件)。「ノ」「ハ」のような 1 モーラの区間を埋められるのは内蔵の表
だけなので、この表は利便のためではなく**経路が成立するための必須部品**。

助詞は繋ぎなので、語と同じ重みでスコアに入れない (`PARTICLE_BONUS`)。同じ音を
語で埋められるならそのほうが情報量があるが、助詞しか合わないところに無理に
語を当てると「意味のある列」から遠ざかる。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .distance import (
    INDEL_COST,
    WORST_SUBSTITUTION_COST,
    edit_distance_csr,
    phoneme_ids,
    weighted_edit_distance,
)
from .index import DEFAULT_CATEGORIES, Category
from .phonology import Mora, Pronunciation, analyze_reading
from .store import PhoneticStore

#: 繋ぎに使える機能語。**索引から引けないので内蔵する** (モジュール docstring 参照)。
#:
#: 表層と読みの対応だけを持つ。助詞は表記の揺れがほぼ無いので、辞書を引く
#: 必要がない。
#:
#: **「は」「へ」「を」は読みを 2 つ持たせる。** 助詞としての発音は表記と違う
#: (ハ→[wa]、ヘ→[e]、ヲ→[o]) が、入力がどちらで書かれてくるかは決まらない。
#: 「ワタシノナマエハ」のようにかなを並べた入力は表記どおりの ハ で来るのに対し、
#: 実際の発音を写した入力は ワ で来る。片方しか持たないと、もう片方の入力で
#: その位置に助詞が当たらない — 実測で読みを ワ だけにしていたとき、
#: 「ワタシノナマエハ」の末尾の ハ に「は」(sim 0.717) が負けて「さ」(サ、0.867)
#: が採られ、「私の名前は」が候補に出てこなかった。
_PARTICLES: tuple[tuple[str, str], ...] = (
    # 格助詞。
    ("の", "ノ"),
    ("を", "オ"),
    ("を", "ヲ"),
    ("に", "ニ"),
    ("が", "ガ"),
    ("は", "ワ"),  # 係助詞としての発音 [wa]
    ("は", "ハ"),  # 表記どおりに書かれた入力のため
    ("へ", "エ"),  # 方向の助詞としての発音 [e]
    ("へ", "ヘ"),
    ("で", "デ"),
    ("と", "ト"),
    ("も", "モ"),
    ("や", "ヤ"),
    ("か", "カ"),
    ("ね", "ネ"),
    ("よ", "ヨ"),
    ("さ", "サ"),
    ("な", "ナ"),
    ("ば", "バ"),
    ("し", "シ"),
    ("ぞ", "ゾ"),
    ("ぜ", "ゼ"),
    ("わ", "ワ"),
    # 2 モーラ以上の助詞・助動詞。長い繋ぎが要るところで効く。
    ("から", "カラ"),
    ("まで", "マデ"),
    ("より", "ヨリ"),
    ("など", "ナド"),
    ("だけ", "ダケ"),
    ("こそ", "コソ"),
    ("しか", "シカ"),
    ("ほど", "ホド"),
    ("でも", "デモ"),
    ("には", "ニワ"),
    ("には", "ニハ"),
    ("では", "デワ"),
    ("では", "デハ"),
    ("とは", "トワ"),
    ("とは", "トハ"),
    ("のに", "ノニ"),
    ("ので", "ノデ"),
    ("けど", "ケド"),
    ("たり", "タリ"),
    ("ながら", "ナガラ"),
    ("だった", "ダッタ"),
    ("です", "デス"),
    ("ます", "マス"),
    ("した", "シタ"),
    ("する", "スル"),
    ("いる", "イル"),
    ("ある", "アル"),
    ("なる", "ナル"),
    ("だ", "ダ"),
)

#: 1 区間に許すモーラ数の上限。
#:
#: 区間の数は入力長 x この値で増え、区間ごとに 1 回ずつ編集距離のバッチが走る
#: ので、そのまま実行時間に乗る。4 にすると 13 モーラで 46 区間・約 0.5 秒。
#: 上げても得が薄い — 5 モーラ以上の区間に当たる語は、それ自体が既存の
#: `search` で引ける長さなので、分割合成の役割ではない。
DEFAULT_MAX_CHUNK_MORAS = 4

#: 1 区間あたり保持する語の数。
#:
#: 連結の探索はこの候補の組み合わせを見るので、ビーム幅と掛け合わさって
#: 探索空間になる。区間内の上位だけ見れば足りるのは、区間のスコアが
#: 全体スコアに線形に入るため — 区間で 20 位の語が全体で 1 位になるには
#: 他の区間が極端に悪い必要があり、その組み合わせはビームが別に持つ。
DEFAULT_CHUNK_CANDIDATES = 12

#: 連結の探索で各位置に保持する経路の数。
#:
#: モーラ位置ごとに「そこまでの最良経路」を上位 N 本だけ残す。厳密な DP
#: (各位置 1 本) では駄目で、**そこまでのスコアが最良の経路が、続きで詰まる**
#: ことがある。位置 5 までを 2 語で綺麗に埋めた経路より、少し悪い 3 語の経路の
#: ほうが残り 8 モーラに合う語を持つ場合がある。ビームはその分岐を残す。
DEFAULT_BEAM_WIDTH = 24

#: 区間のスコアがこれを下回る語は候補に採らない。
#:
#: 音が合っていない語を並べても空耳として読めないので、足切りする。
#: 0.6 は「モーラ数が同じで子音 1 つが違う」程度が通る線。
DEFAULT_MIN_CHUNK_SCORE = 0.55

#: 助詞を当てたときに区間スコアへ掛ける係数。
#:
#: 助詞に掛ける係数。
#:
#: 助詞は音が完全に一致していても「繋ぎ」でしかないので、同じスコアの語より
#: わずかに低く見る。ただし**下げすぎてはいけない** — 1 モーラの区間を
#: 埋められるのは助詞だけなので (索引に 1 モーラの語が無い)、係数が低いと
#: ビームが助詞を通る経路を刈ってしまい、「ワタシ|ノ|ナマエ|ハ」のような
#: 本来の切り方が出てこない。実測では 0.9 で「ワタ|シノ|ナマ|エサ」に負け、
#: 0.98 で助詞を使う経路が上位に来た。
PARTICLE_BONUS = 0.98

#: 区間スコアに占める音韻類似度の重み。残りを一般性と表層の情報量で分ける。
#:
#: **加算ではなく配分にする。** 加算 (`similarity + 0.25 * familiarity`) だと
#: 区間スコアが 1.25 まで伸びて、合成結果のスコアが 1.0 を超える。通常検索の
#: スコアが 0〜1 なのと食い違い、2 つの経路の数値を並べて読めなくなる。
WEIGHT_PHONETIC = 0.72

#: 区間スコアに占める語の一般性の重み。
#:
#: 空耳は「知っている語」で構成されないと読めないので、`search` の pun
#: プリセット (0.15) より強く効かせる。稀語で音だけ合わせた列は、音韻的に
#: 正しくても空耳として機能しない。
WEIGHT_FAMILIARITY = 0.16

#: 区間スコアに占める表層の情報量の重み。
#:
#: 読みをそのままカタカナで書いた見出し (「ワタシ」「エサ」) は音が合っていても
#: 何を指すのか読めない。`search._surface_informativeness` が同音異表記の代表を
#: 選ぶのに使っているのと同じ判断を、ここでは**候補の採否**に効かせる。
#: これが無いと「ワタシノナマエサ」のようなカタカナだけの列が上位を埋める
#: (実測: 上位 5 件中 4 件がカタカナ見出しだった)。
#:
#: **一般性より軽くしなければならない。** 情報量は「漢字かどうか」しか見ないので、
#: 重くすると稀な漢字語 (「撰り」「梻」「炮り」) が既知の語に勝つ。実測で 0.12 と
#: 一般性 0.16 が逆転していた頃は「ワタシノナマエハ」が「分か死の七異派」になった。
WEIGHT_INFORMATIVENESS = 0.08

#: 語の数に対する減点。
#:
#: 細かく刻めばどんな入力でも音は合う (1 モーラずつ助詞で埋めれば距離 0 に
#: 近づく) が、それは空耳ではない。語数が増えるほど減点して、**少ない語で
#: 長く合わせた列**を上位に出す。区間スコアの平均だけで順位を付けると、
#: 常に最も細かい分割が勝ってしまう。
SEGMENT_PENALTY = 0.06

#: ラティス表示で描くノード数の上限。
#:
#: 上位数件を畳んだだけでは構造が薄く (5 件で 8 ノード)、全候補を載せると画面が
#: 埋まる。**上限を決めてそこまで経路を足す**という中間を取る。
#:
#: 40 は 8 モーラの入力でビーム幅 96 前後に相当する (実測: beam 48 で 32 ノード、
#: 96 で 45 ノード)。モーラ位置ごとに数ノードが並ぶ密度で、分岐が読める。
DEFAULT_NODE_BUDGET = 40

#: 1 つの区間に並べるノード数の上限。
#:
#: 同じ区間のノードは互いに排他 (どれか 1 つを選ぶ) なので、図では同じ縦位置に
#: 積み上がる。短い入力では 1 区間に候補が集中し、実測で「チクビ」(3 モーラ) は
#: [0:3] に 10 個・[1:3] に 12 個が並んで 24 行・高さ 1138px の縦長になった。
#:
#: 予算 (`DEFAULT_NODE_BUDGET`) は全体の数を抑えるだけで、1 か所への集中は
#: 抑えない。**両方要る** — 予算だけでは細長い図になり、区間の上限だけでは
#: 総数が膨らむ。
DEFAULT_MAX_NODES_PER_SPAN = 6

#: ノード上限に届くまでビーム幅を広げるときの倍率と上限。
#:
#: **候補数を増やしてもノードは増えない。** 経路の数はビーム幅で決まるので
#: (実測: limit=500 でも beam=24 なら候補 24 件で飽和する)、ラティスを育てる
#: 唯一の手はビームを広げること。
#:
#: 広げるのは安い。8 モーラの入力で beam 24 → 400 が 428ms → 293ms
#: (最初の 1 回が暖機を払っただけで、実質は横ばい)。区間ごとの照合が支配的で、
#: ビームの刈り込みはその後の話なので幅にほとんど比例しない。
_LATTICE_BEAM_GROWTH = 2
_LATTICE_MAX_BEAM = 512


@dataclass(frozen=True)
class PhraseSegment:
    """合成結果を構成する 1 区間。

    「入力のどこが」「何に」なったかを持つ。空耳は対応が読めないと検証
    できないので、区間の位置と入力側の読みまで返す。
    """

    surface: str
    reading: str
    #: 入力のモーラ列における区間 [start, end)。
    start: int
    end: int
    #: 入力側のこの区間の読み (カタカナ)。
    source_reading: str
    #: この区間の音韻類似度 (0.0〜1.0)。
    similarity: float
    #: 助詞・助動詞として埋めたか。
    is_particle: bool
    phonemes: tuple[str, ...]

    @property
    def mora_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class PhraseCandidate:
    """合成結果 1 件。"""

    #: 連結した表層 (「渡しの名前は」)。
    text: str
    #: 連結した読み (カタカナ)。
    reading: str
    #: 全体のスコア。
    score: float
    #: 区間スコアの (助詞・一般性を含む) 平均。語数の減点を含まない。
    phonetic_similarity: float
    segments: tuple[PhraseSegment, ...]

    @property
    def segment_count(self) -> int:
        return len(self.segments)


@dataclass(frozen=True)
class LatticeNode:
    """ラティスの 1 ノード = ある区間に当てた 1 つの語。

    候補リストでは同じ語が候補ごとに何度も現れる。実測では表示している区間の
    65〜77% が重複で、「名前」「は」「納衣」「に」は上位 10 件の**全部**に出て
    いた (40 区間が 11 ノードに畳める)。ノードにすると 1 度しか描かれない。
    """

    #: ノードの識別子。`{start}:{end}:{surface}` — 同じ区間に同じ表層を当てた
    #: ものは 1 つのノードに畳む。読みではなく表層で畳むのは、画面に出るのが
    #: 表層で、違う表層が同じノードに見えると読めなくなるため。
    id: str
    surface: str
    reading: str
    start: int
    end: int
    source_reading: str
    similarity: float
    is_particle: bool
    #: このノードを通る経路の数。太さや濃さに写して「よく使われる語」を示す。
    path_count: int
    #: このノードを通る経路の最良スコア。
    best_score: float

    @property
    def mora_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class LatticeEdge:
    """ラティスの辺 = 2 つのノードが経路上で隣接したこと。

    `source` / `target` が `None` のときは入力の始端・終端との接続。
    どのノードが文の先頭・末尾になり得るかが図に要る。
    """

    source: str | None
    target: str | None
    #: この辺を通る経路の数。
    path_count: int


@dataclass(frozen=True)
class PhraseLattice:
    """合成の候補群を 1 枚の DAG に畳んだもの。

    候補リストと**同じ経路集合**を表す。候補を N 件並べる代わりに、区間ごとの
    ノードと隣接関係で持つので同じ語が 1 度しか出ない。

    ノード数は `node_budget` で抑える。上限に届くまでビーム幅を広げて経路を
    足していくので (`PhraseComposer.lattice`)、上位数件だけでは薄く、全候補
    では埋まるという両極を避けられる。
    """

    nodes: tuple[LatticeNode, ...]
    edges: tuple[LatticeEdge, ...]
    #: 畳む前の経路。図から一覧に戻れるようにするため、また `_trim_lattice` が
    #: 経路単位で削るために持つ。
    paths: tuple[PhraseCandidate, ...]
    #: 経路の数 (`len(paths)`)。
    path_count: int
    #: 実際に使ったビーム幅。どこまで広げたかを呼び出し側が読める。
    beam_width: int
    #: 予算のために経路を削ったか。**「もっと候補がある」ことを意味する** —
    #: ビーム幅を広げ切った・選択肢を使い切った場合は立たない。
    truncated: bool

    @property
    def node_count(self) -> int:
        return len(self.nodes)


def _mora_phonemes(moras: tuple[Mora, ...]) -> tuple[str, ...]:
    return tuple(p for m in moras for p in m.phonemes)


def _informativeness(surface: str, reading: str) -> float:
    """表層が語を特定できる度合い。

    `search._surface_informativeness` と同じ判断。あちらは同音異表記の
    代表選びに使うが、こちらは**候補の採否**に効かせる (`WEIGHT_INFORMATIVENESS`)。
    分割合成では区間ごとに表記を選ぶので、読みをそのまま書いた見出しを許すと
    列全体がカタカナで埋まって何を指すのか読めなくなる。
    """
    if surface == reading:
        return 0.0
    if any("一" <= ch <= "鿿" for ch in surface):
        return 1.0
    return 0.5


def _is_readable_surface(surface: str) -> bool:
    """表層が合成結果の一部として読めるか。

    `index.is_searchable_surface` は「かなか漢字が 1 文字でもあれば通す」ので、
    「2 士」「8 耐」のような数字混じりの見出しが索引に入っている。1 語だけ返す
    通常検索ではさほど目立たないが、**分割合成では列に混ざると全体が読めなく
    なる** (実測で「コンニチハセカイ」が「婚 2 士 8 耐」になった)。
    数字とラテン文字を含む表層を落とす。

    全角の英数字 (「Ｂ級」) も同じ理由で落とす。半角だけを見ていると
    ASCII 判定を素通りする。
    """
    return not any(_is_alnum_latin(ch) for ch in surface)


def _is_alnum_latin(ch: str) -> bool:
    """数字・ラテン文字か (半角と全角の両方)。"""
    if ch.isascii():
        return ch.isalnum()
    return "０" <= ch <= "９" or "Ａ" <= ch <= "Ｚ" or "ａ" <= ch <= "ｚ"


def _chunk_similarity(
    distances: np.ndarray,
    query_length: int,
    candidate_lengths: np.ndarray,
) -> np.ndarray:
    """編集距離を類似度に写す。

    `search._edit_similarity_array` と同じ式。**片方だけ変えると分割合成と
    通常検索でスコアの尺度が食い違う** ので、意図的に同じ正規化を使う
    (`distance.similarity_normalizer` の配列版)。
    """
    if distances.size == 0:
        return distances
    shorter = np.minimum(candidate_lengths, query_length).astype(np.float64)
    longer = np.maximum(candidate_lengths, query_length).astype(np.float64)
    denominator = shorter * WORST_SUBSTITUTION_COST + (longer - shorter) * INDEL_COST
    with np.errstate(divide="ignore", invalid="ignore"):
        similarity = 1.0 - distances / denominator
    return np.clip(np.where(denominator > 0.0, similarity, 0.0), 0.0, 1.0)


@dataclass(frozen=True)
class _ChunkOption:
    """1 区間に当てられる 1 つの選択肢。"""

    surface: str
    reading: str
    similarity: float
    #: スコアに入る値。一般性と助詞の係数を含む。
    weight: float
    is_particle: bool
    phonemes: tuple[str, ...]


class PhraseComposer:
    """入力を「複数の語 + 助詞」の連なりに合成する。

    索引の行をモーラ数ごとに 1 度だけ選抜して持つ (`_pool`)。区間ごとに
    カテゴリのマスクを立て直すと、13 モーラの入力で 46 回払うことになる。
    実測で選抜は全体で 182ms かかるので、常駐する窓 (Web / MCP) では
    ここを使い回せるかどうかが体感を分ける。
    """

    def __init__(
        self,
        store: PhoneticStore,
        *,
        categories: tuple[Category, ...] | None = None,
    ) -> None:
        self.store = store
        self.categories = tuple(categories) if categories else tuple(DEFAULT_CATEGORIES)

    @cached_property
    def _particles(self) -> dict[int, tuple[_ChunkOption, ...]]:
        """モーラ数ごとの助詞の選択肢。

        助詞は音素列が固定なので、区間との照合は実行時に距離を取るだけ。
        ここではモーラ数で引けるように分類しておく。
        """
        by_moras: dict[int, list[_ChunkOption]] = {}
        for surface, reading in _PARTICLES:
            pronunciation = analyze_reading(reading)
            if not pronunciation.phonemes:
                continue
            option = _ChunkOption(
                surface=surface,
                reading=pronunciation.reading,
                similarity=0.0,
                weight=0.0,
                is_particle=True,
                phonemes=pronunciation.phonemes,
            )
            by_moras.setdefault(pronunciation.mora_count, []).append(option)
        return {moras: tuple(options) for moras, options in by_moras.items()}

    @cached_property
    def _pool(self) -> dict[int, np.ndarray]:
        """モーラ数ごとの、カテゴリを通った索引の行。

        行はモーラ数の昇順に並んでいるので (`store._locality_order`)、
        モーラ数ごとの区間は連続スライスで切れる (`store.mora_range`)。
        カテゴリのマスクだけをそこに積む。
        """
        category_ids = self.store.category_ids
        wanted = [self.store.category_id(c) for c in self.categories]
        wanted = [i for i in wanted if i >= 0]
        if not wanted:
            return {}

        size = max(int(category_ids.max(initial=0)), max(wanted)) + 1
        lookup = np.zeros(size, dtype=bool)
        lookup[wanted] = True

        pool: dict[int, np.ndarray] = {}
        moras = self.store.mora_counts
        highest = int(moras.max(initial=0))
        for width in range(1, highest + 1):
            start, end = self.store.mora_range(width, width)
            if end <= start:
                continue
            keep = lookup[category_ids[start:end]]
            rows = np.arange(start, end, dtype=np.int64)[keep]
            if rows.size:
                pool[width] = rows
        return pool

    def compose(
        self,
        text: str,
        *,
        pronunciation: Pronunciation | None = None,
        limit: int = 10,
        max_chunk_moras: int = DEFAULT_MAX_CHUNK_MORAS,
        chunk_candidates: int = DEFAULT_CHUNK_CANDIDATES,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        min_chunk_score: float = DEFAULT_MIN_CHUNK_SCORE,
        allow_particles: bool = True,
    ) -> tuple[Pronunciation, list[PhraseCandidate]]:
        """`text` の音を「複数の語 + 助詞」で組み立てた候補を返す。

        戻り値は (入力の音韻表現, 候補リスト)。

        `pronunciation` を渡すと読みの解析を省く (呼び出し側が既に持っている
        場合のため)。渡さないときは索引の読み取得器ではなく `analyze_reading`
        を使うので、**入力はかなである必要がある**。漢字を含むテキストは
        `PhoneticSearcher.pronounce` で読みに直してから渡すこと — ここで
        Sudachi を持たないのは、合成器が索引だけで完結するようにするため。

        `max_chunk_moras` を上げると 1 区間に長い語を当てられるが、区間の数が
        入力長に対して線形に増えるぶん遅くなる。`beam_width` は連結の探索幅で、
        上げると「そこまでは少し悪いが続きが合う」経路を拾いやすくなる。
        """
        if pronunciation is None:
            pronunciation = analyze_reading(text)
        moras = pronunciation.moras
        if not moras:
            return pronunciation, []

        options = self._chunk_options(
            moras,
            max_chunk_moras=max_chunk_moras,
            chunk_candidates=chunk_candidates,
            min_chunk_score=min_chunk_score,
            allow_particles=allow_particles,
        )
        paths = self._search_paths(
            moras,
            options,
            beam_width=beam_width,
            limit=limit,
        )
        return pronunciation, paths

    # --- ラティス (候補群を 1 枚の DAG に畳む) -----------------------------

    def lattice(
        self,
        text: str,
        *,
        pronunciation: Pronunciation | None = None,
        node_budget: int = DEFAULT_NODE_BUDGET,
        max_nodes_per_span: int = DEFAULT_MAX_NODES_PER_SPAN,
        max_chunk_moras: int = DEFAULT_MAX_CHUNK_MORAS,
        chunk_candidates: int = DEFAULT_CHUNK_CANDIDATES,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        min_chunk_score: float = DEFAULT_MIN_CHUNK_SCORE,
        allow_particles: bool = True,
    ) -> tuple[Pronunciation, PhraseLattice]:
        """候補群を 1 枚の DAG に畳んで返す。

        戻り値は (入力の音韻表現, ラティス)。

        **`node_budget` に届くまでビーム幅を広げる。** 上位数件を畳んだだけでは
        構造が薄く (5 件で 8 ノード)、全候補を載せると画面が埋まるので、
        ノード数を予算として与えてそこまで経路を足す。

        候補数 (`limit`) を増やしてもノードは増えない — 経路の数はビーム幅で
        決まるので、`limit=500` でも `beam_width=24` なら候補 24 件で飽和する
        (実測)。だからここが動かすのはビーム幅のほう。

        `max_nodes_per_span` は 1 区間に並べるノード数の上限。予算は全体の数しか
        抑えないので、これが無いと短い入力で 1 か所に候補が集中して縦長になる
        (実測: 「チクビ」が 24 行・高さ 1138px)。

        区間ごとの照合 (`_chunk_options`) はビーム幅に依存しないので、
        広げて作り直すあいだ**使い回す**。これが支配的なコストなので、
        作り直すと試行回数ぶん丸ごと払うことになる。
        """
        if pronunciation is None:
            pronunciation = analyze_reading(text)
        moras = pronunciation.moras
        if not moras:
            return pronunciation, PhraseLattice((), (), (), 0, beam_width, False)

        options = self._chunk_options(
            moras,
            max_chunk_moras=max_chunk_moras,
            chunk_candidates=chunk_candidates,
            min_chunk_score=min_chunk_score,
            allow_particles=allow_particles,
        )

        # 予算に届くまでビームを広げる。**採用するのは予算を超えない最後の幅**。
        # 超えた幅の結果をそのまま返すと `node_budget` が守られない。
        #
        # ただし最小の幅でも予算を超えることがある (3 モーラの入力は区間の
        # 選択肢が多く、ビーム 24 で 26 ノードになる)。この場合は広げずに
        # `_trim_lattice` でノードを削って予算に収める — 予算は画面が埋まらない
        # ことの保証なので、下回れないなら削るしかない。
        width = max(beam_width, 1)
        accepted: PhraseLattice | None = None
        while True:
            paths = self._search_paths(moras, options, beam_width=width, limit=width)
            built = _build_lattice(paths, beam_width=width)
            # 区間あたりの上限は幅に関係なく効かせる。1 か所への集中は予算では
            # 抑えられない (`DEFAULT_MAX_NODES_PER_SPAN`)。
            built = _fit_lattice(
                built, node_budget=node_budget, max_nodes_per_span=max_nodes_per_span
            )

            if built.node_count > node_budget and accepted is not None:
                # 予算を超えた。収まっていた 1 つ前を採る。
                return pronunciation, accepted

            accepted = built
            # 予算を使い切った / これ以上広げられない / 経路が増えなくなった
            # (選択肢を使い切った) — いずれも打ち止め。
            exhausted = built.path_count < width
            if built.node_count >= node_budget or width >= _LATTICE_MAX_BEAM or exhausted:
                return pronunciation, accepted
            width = min(width * _LATTICE_BEAM_GROWTH, _LATTICE_MAX_BEAM)

    # --- 区間ごとの照合 ---------------------------------------------------

    def _chunk_options(
        self,
        moras: tuple[Mora, ...],
        *,
        max_chunk_moras: int,
        chunk_candidates: int,
        min_chunk_score: float,
        allow_particles: bool,
    ) -> dict[tuple[int, int], tuple[_ChunkOption, ...]]:
        """区間 (start, end) ごとの選択肢を求める。

        同じ音素列の区間は 1 度しか照合しない。入力に同じ音の繰り返しがある
        とき (「アルミカンノウエニアルミカン」) に効くが、実測では重複は
        46 区間中 2 件しかないので、これは主に正しさのためではなく
        繰り返しの多い入力で破綻しないための保険。
        """
        blob, bounds, distance_ids = self.store.phoneme_csr
        total = len(moras)
        cache: dict[tuple[tuple[str, ...], int], tuple[_ChunkOption, ...]] = {}
        options: dict[tuple[int, int], tuple[_ChunkOption, ...]] = {}

        for start in range(total):
            for end in range(start + 1, min(start + max_chunk_moras, total) + 1):
                width = end - start
                phonemes = _mora_phonemes(moras[start:end])
                if not phonemes:
                    continue
                key = (phonemes, width)
                found = cache.get(key)
                if found is None:
                    found = self._match_chunk(
                        phonemes,
                        width,
                        blob=blob,
                        bounds=bounds,
                        distance_ids=distance_ids,
                        chunk_candidates=chunk_candidates,
                        min_chunk_score=min_chunk_score,
                        allow_particles=allow_particles,
                    )
                    cache[key] = found
                if found:
                    options[(start, end)] = found
        return options

    def _match_chunk(
        self,
        phonemes: tuple[str, ...],
        width: int,
        *,
        blob: np.ndarray,
        bounds: np.ndarray,
        distance_ids: np.ndarray,
        chunk_candidates: int,
        min_chunk_score: float,
        allow_particles: bool,
    ) -> tuple[_ChunkOption, ...]:
        """1 区間に当てられる語を上位から集める。

        母集団は**同じモーラ数の語だけ** (モジュール docstring 参照)。
        編集距離は索引の CSR をそのまま Rust に渡してまとめて計算する。
        """
        query_ids = phoneme_ids(phonemes)
        collected: list[_ChunkOption] = []

        rows = self._pool.get(width)
        if rows is not None and rows.size:
            # 音素 CSR はグループで引く (`store.phoneme_csr`)。同音異表記は
            # 同じ音素列なので、代表 1 件の距離を残りへ配れば済む。行は昇順に
            # 並ぶので (`_pool` が連続区間から作る)、グループ列も昇順になり
            # 代表判定は隣接比較だけで済む (`np.unique` のソートが要らない)。
            groups = self.store.group_ids[rows].astype(np.int64)
            first = np.empty(groups.size, dtype=bool)
            first[0] = True
            np.not_equal(np.diff(groups), 0, out=first[1:])
            leaders = groups[first]
            distances = edit_distance_csr(query_ids, leaders, blob, bounds, distance_ids)
            if leaders.size != groups.size:
                distances = distances[np.cumsum(first) - 1]
            lengths = self.store.phoneme_lengths(rows)
            similarity = _chunk_similarity(distances, query_ids.size, lengths)

            # 一般性は索引が列で持つ (`frequency` が構築時に埋める)。
            familiarity = self.store.familiarities[rows].astype(np.float64)
            # 表層の情報量は文字列を復号しないと出せないので、配列で出せる分
            # (音韻 + 一般性) だけで広めに絞り、そこから先は 1 件ずつ見る。
            partial = WEIGHT_PHONETIC * similarity + WEIGHT_FAMILIARITY * familiarity

            eligible = np.flatnonzero(similarity >= min_chunk_score)
            if eligible.size:
                # 情報量で順位が入れ替わる分を見込んで多めに起こす。ここを
                # `chunk_candidates` ぴったりにすると、カタカナ見出しばかりが
                # 先に採られて漢字表記が候補から消える。
                take = min(chunk_candidates * 4, eligible.size)
                # `entry()` は 1 件ずつ Python オブジェクトを作るので、全候補に
                # 対して呼ばない (`search._materialize` と同じ理由)。
                order = eligible[np.argpartition(-partial[eligible], take - 1)[:take]]
                order = order[np.argsort(-partial[order])]
                # 同音異表記は区間内で 1 件に畳む。同じ音を複数の表記で並べても
                # 合成結果の列としては同じものしか作れない。**畳むときは情報量の
                # 高い表記を残す** — 到着順で残すと「価格」に対する「カカク」の
                # ようなカタカナ見出しが代表になる (`search._representative_rank`
                # と同じ問題)。
                best: dict[str, _ChunkOption] = {}
                for position in order:
                    row = int(rows[position])
                    reading = self.store.reading(row)
                    surface = self.store.surface(row)
                    if not _is_readable_surface(surface):
                        continue
                    option = _ChunkOption(
                        surface=surface,
                        reading=reading,
                        similarity=float(similarity[position]),
                        weight=float(
                            partial[position]
                            + WEIGHT_INFORMATIVENESS * _informativeness(surface, reading)
                        ),
                        is_particle=False,
                        phonemes=self.store.phonemes(row),
                    )
                    current = best.get(reading)
                    if current is None or option.weight > current.weight:
                        best[reading] = option
                collected.extend(best.values())

        if allow_particles:
            collected.extend(
                self._match_particles(phonemes, width, min_chunk_score=min_chunk_score)
            )

        collected.sort(key=lambda option: -option.weight)
        return tuple(collected[:chunk_candidates])

    def _match_particles(
        self,
        phonemes: tuple[str, ...],
        width: int,
        *,
        min_chunk_score: float,
    ) -> list[_ChunkOption]:
        """同じモーラ数の助詞を区間に照合する。

        件数が数十しかないので、索引の経路 (CSR のバッチ) を通さず
        1 件ずつ距離を取る。**1 モーラの区間を埋められるのはここだけ** —
        索引に 1 モーラの語が無い (モジュール docstring 参照)。
        """
        found: list[_ChunkOption] = []
        for option in self._particles.get(width, ()):
            distance = weighted_edit_distance(phonemes, option.phonemes)
            similarity = _chunk_similarity(
                np.array([distance], dtype=np.float64),
                len(phonemes),
                np.array([len(option.phonemes)], dtype=np.int64),
            )[0]
            if similarity < min_chunk_score:
                continue
            found.append(
                _ChunkOption(
                    surface=option.surface,
                    reading=option.reading,
                    similarity=float(similarity),
                    # 助詞はどれも既知の語なので一般性は満点。情報量はひらがな
                    # 表記なので中間 (`_informativeness` の 0.5 と揃える) —
                    # 語と同じ式に載せておかないと、区間スコアの尺度が
                    # 助詞のところだけ別物になる。
                    weight=(
                        WEIGHT_PHONETIC * float(similarity)
                        + WEIGHT_FAMILIARITY
                        + WEIGHT_INFORMATIVENESS * 0.5
                    )
                    * PARTICLE_BONUS,
                    is_particle=True,
                    phonemes=option.phonemes,
                )
            )
        return found

    # --- 連結 -------------------------------------------------------------

    def _search_paths(
        self,
        moras: tuple[Mora, ...],
        options: dict[tuple[int, int], tuple[_ChunkOption, ...]],
        *,
        beam_width: int,
        limit: int,
    ) -> list[PhraseCandidate]:
        """区間の選択肢を繋いで、入力全体を覆う列を作る。

        モーラ位置に沿った DP + ビーム。位置 `i` までを覆う経路の上位
        `beam_width` 本を持ち、そこから使える区間を伸ばす。厳密な DP
        (位置ごとに 1 本) にできないのは、区間スコアの平均と語数の減点で
        順位が決まるため — 部分列の最良が全体の最良の部分列とは限らない
        (`DEFAULT_BEAM_WIDTH` の項を参照)。
        """
        total = len(moras)
        # beam[i] = 位置 i までを覆う経路のリスト。
        # 経路は (累積スコア, 区間数, 選択肢の列)。
        beam: list[list[tuple[float, tuple[_ChunkOption, ...], tuple[tuple[int, int], ...]]]] = [
            [] for _ in range(total + 1)
        ]
        beam[0].append((0.0, (), ()))

        # 区間の最大長は選択肢の張り方と一致させる。ここを `_chunk_options` と
        # 別の値にすると、照合していない区間を伸ばそうとして黙って取り落とす。
        reach = max((end - start for start, end in options), default=1)

        for position in range(total):
            # 到達した経路は必ずビーム幅まで刈ってから伸ばす。刈るのは
            # 「これから伸ばす位置」であって直後の位置ではない — 位置 i へは
            # 複数の区間長から合流するので、i を処理する時点で初めて全部の
            # 流入が揃う。
            if not beam[position]:
                continue
            if len(beam[position]) > beam_width:
                beam[position] = _prune(beam[position], beam_width)

            for end in range(position + 1, min(position + reach, total) + 1):
                span = (position, end)
                for option in options.get(span, ()):
                    for accumulated, chosen, spans in beam[position]:
                        beam[end].append(
                            (
                                accumulated + option.weight,
                                (*chosen, option),
                                (*spans, span),
                            )
                        )

        # 終端に届いた経路を評価する。終端は上のループが処理しないので
        # (伸ばす先が無い) ここで刈る。刈らないと limit を返すだけのために
        # 数万本の経路を `PhraseCandidate` に起こすことになる。
        final = beam[total]
        if len(final) > beam_width:
            final = _prune(final, beam_width)

        candidates: list[PhraseCandidate] = []
        for accumulated, chosen, spans in final:
            if not chosen:
                continue
            average = accumulated / len(chosen)
            score = average - SEGMENT_PENALTY * (len(chosen) - 1)
            segments = tuple(
                PhraseSegment(
                    surface=option.surface,
                    reading=option.reading,
                    start=start,
                    end=end,
                    source_reading="".join(m.kana for m in moras[start:end]),
                    similarity=round(option.similarity, 4),
                    is_particle=option.is_particle,
                    phonemes=option.phonemes,
                )
                for option, (start, end) in zip(chosen, spans, strict=True)
            )
            candidates.append(
                PhraseCandidate(
                    text="".join(s.surface for s in segments),
                    reading="".join(s.reading for s in segments),
                    score=round(score, 4),
                    phonetic_similarity=round(average, 4),
                    segments=segments,
                )
            )

        # 同じ表層に落ちる経路 (区間の切り方が違っても表層が同じ) を畳む。
        best: dict[str, PhraseCandidate] = {}
        for candidate in candidates:
            current = best.get(candidate.text)
            if current is None or candidate.score > current.score:
                best[candidate.text] = candidate

        ordered = sorted(best.values(), key=lambda c: (-c.score, c.segment_count, c.text))
        return ordered[:limit]


def _node_id(segment: PhraseSegment) -> str:
    """ノードの識別子。同じ区間に同じ表層を当てたものを 1 つに畳む。

    読みではなく表層で畳む。画面に出るのが表層なので、違う表層 (「白」と「篠」)
    が同じノードに見えると図が読めなくなる。
    """
    return f"{segment.start}:{segment.end}:{segment.surface}"


def _build_lattice(
    paths: list[PhraseCandidate],
    *,
    beam_width: int,
) -> PhraseLattice:
    """候補リストを DAG に畳む。

    **候補と同じ経路集合を表す。** ノードは (区間, 表層) で畳み、経路上で
    隣接した対を辺にする。始端・終端との接続も辺として持つ (`source` /
    `target` が `None`) — どのノードが文の先頭・末尾になり得るかが図に要る。

    ノードごとに通過した経路数と最良スコアを数える。「よく使われる語」を
    太さや濃さに写せるようにするためで、順位を 1 本の列に戻さずに
    「どの語が効いているか」を読ませる。
    """
    node_paths: dict[str, int] = {}
    node_best: dict[str, float] = {}
    node_of: dict[str, PhraseSegment] = {}
    edge_paths: dict[tuple[str | None, str | None], int] = {}

    for candidate in paths:
        segments = candidate.segments
        if not segments:
            continue
        ids = [_node_id(s) for s in segments]
        for identifier, segment in zip(ids, segments, strict=True):
            node_paths[identifier] = node_paths.get(identifier, 0) + 1
            # 最良スコアは経路のスコア。ノード単体のスコアではなく「このノードを
            # 通る最良の経路がどれだけ良いか」を示す。
            if candidate.score > node_best.get(identifier, -1.0):
                node_best[identifier] = candidate.score
            node_of.setdefault(identifier, segment)
        for left, right in itertools.pairwise(ids):
            key = (left, right)
            edge_paths[key] = edge_paths.get(key, 0) + 1
        edge_paths[(None, ids[0])] = edge_paths.get((None, ids[0]), 0) + 1
        edge_paths[(ids[-1], None)] = edge_paths.get((ids[-1], None), 0) + 1

    nodes = tuple(
        LatticeNode(
            id=identifier,
            surface=segment.surface,
            reading=segment.reading,
            start=segment.start,
            end=segment.end,
            source_reading=segment.source_reading,
            similarity=segment.similarity,
            is_particle=segment.is_particle,
            path_count=node_paths[identifier],
            best_score=round(node_best[identifier], 4),
        )
        for identifier, segment in sorted(
            node_of.items(),
            # 図は左から右に読むので、位置順に並べる。同じ位置では経路数の
            # 多い順 (よく使われる語を上に置く)。
            key=lambda item: (item[1].start, item[1].end, -node_paths[item[0]], item[1].surface),
        )
    )
    edges = tuple(
        LatticeEdge(source=source, target=target, path_count=count)
        for (source, target), count in sorted(
            edge_paths.items(), key=lambda item: (-item[1], str(item[0]))
        )
    )
    return PhraseLattice(
        nodes=nodes,
        edges=edges,
        paths=tuple(paths),
        path_count=len(paths),
        beam_width=beam_width,
        truncated=False,
    )


def _fit_lattice(
    lattice: PhraseLattice,
    *,
    node_budget: int,
    max_nodes_per_span: int,
) -> PhraseLattice:
    """ノードを削って予算と区間あたりの上限に収める。

    最小のビーム幅でも予算を超えるとき (区間の選択肢が多い短い入力) の受け皿。
    予算は「画面が埋まらないこと」の保証なので、幅を下げられないなら削る。

    **経路を丸ごと残す形で削る。** ノードを個別に落とすと、残ったノードが
    始端から終端まで繋がらなくなり「経路が読める図」でなくなる。スコアの高い
    経路から順に採り、両方の制約に収まる範囲の経路だけでラティスを組み直す。

    区間あたりの上限が要るのは、予算だけでは 1 か所への集中を抑えられないため。
    実測で「チクビ」は [0:3] に 10 個・[1:3] に 12 個が並び、24 行の縦長に
    なった (`DEFAULT_MAX_NODES_PER_SPAN`)。
    """
    over_budget = lattice.node_count > node_budget
    spans: dict[tuple[int, int], int] = {}
    for node in lattice.nodes:
        key = (node.start, node.end)
        spans[key] = spans.get(key, 0) + 1
    over_span = any(count > max_nodes_per_span for count in spans.values())
    if not over_budget and not over_span:
        return lattice

    kept: list[PhraseCandidate] = []
    seen: set[str] = set()
    per_span: dict[tuple[int, int], set[str]] = {}
    for candidate in lattice.paths:
        ids = {_node_id(s) for s in candidate.segments}
        # この経路を足したときに区間あたりの上限を破らないか。
        fits_spans = True
        for segment in candidate.segments:
            key = (segment.start, segment.end)
            current = per_span.get(key, set())
            if _node_id(segment) not in current and len(current) >= max_nodes_per_span:
                fits_spans = False
                break
        if not fits_spans or len(seen | ids) > node_budget:
            # 1 本も採れていなければ最上位だけは採る (空の図を返すより、
            # 制約を少し超えても 1 本見せる)。
            if kept:
                continue
            kept.append(candidate)
            break
        seen |= ids
        for segment in candidate.segments:
            per_span.setdefault((segment.start, segment.end), set()).add(_node_id(segment))
        kept.append(candidate)

    trimmed = _build_lattice(kept, beam_width=lattice.beam_width)
    return PhraseLattice(
        nodes=trimmed.nodes,
        edges=trimmed.edges,
        paths=trimmed.paths,
        path_count=trimmed.path_count,
        beam_width=lattice.beam_width,
        # 元の経路のうち一部しか載せていないことを呼び出し側に伝える。
        truncated=True,
    )


def _prune(
    paths: list[tuple[float, tuple[_ChunkOption, ...], tuple[tuple[int, int], ...]]],
    width: int,
) -> list[tuple[float, tuple[_ChunkOption, ...], tuple[tuple[int, int], ...]]]:
    """経路を上位 `width` 本に絞る。

    比較は**区間あたりの平均**で行う。累積スコアで比べると区間数の多い経路が
    有利になり (1 区間あたり最大 1.25 が積み上がる)、同じ位置に届いた経路の
    優劣が「どれだけ細かく刻んだか」で決まってしまう。
    """
    paths.sort(key=lambda path: -path[0] / max(len(path[1]), 1))
    return paths[:width]


__all__ = [
    "DEFAULT_BEAM_WIDTH",
    "DEFAULT_CHUNK_CANDIDATES",
    "DEFAULT_MAX_CHUNK_MORAS",
    "DEFAULT_MAX_NODES_PER_SPAN",
    "DEFAULT_MIN_CHUNK_SCORE",
    "DEFAULT_NODE_BUDGET",
    "LatticeEdge",
    "LatticeNode",
    "PhraseCandidate",
    "PhraseComposer",
    "PhraseLattice",
    "PhraseSegment",
]

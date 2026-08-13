"""音素間距離と weighted Levenshtein による音韻類似度。

音素を記号として扱わず、音声学的素性 (調音位置・調音方法・有声性・母音の
高さ/前後/円唇性) のベクトルとして表現し、素性の不一致から距離を導く。
これにより k/g のような有声性のみが違う対は近く、k/m のように調音方法まで
違う対は遠くなる。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import jpr_distance as _rust
import numpy as np

from .phonology import GEMINATE, LONG, MORAIC_N, Pronunciation

# --- 素性の定義 -------------------------------------------------------------

# 調音位置。数値は口腔内の前後位置の近さを表す序数で、差が小さいほど近い。
_PLACE = {
    "bilabial": 0,
    "labiodental": 1,
    "alveolar": 2,
    "postalveolar": 3,
    "palatal": 4,
    "velar": 5,
    "glottal": 6,
}

# 調音方法。序数の隣接ではなく対ごとの距離表で持つ。序数にすると
# nasal と liquid のように聴覚的にはっきり区別される対が不当に近くなる。
_MANNER_DISTANCE: dict[frozenset[str], float] = {
    frozenset({"stop", "affricate"}): 0.35,
    frozenset({"stop", "fricative"}): 0.65,
    frozenset({"stop", "nasal"}): 0.55,
    frozenset({"stop", "liquid"}): 0.85,
    frozenset({"stop", "approximant"}): 1.0,
    frozenset({"affricate", "fricative"}): 0.3,
    frozenset({"affricate", "nasal"}): 0.8,
    frozenset({"affricate", "liquid"}): 0.9,
    frozenset({"affricate", "approximant"}): 1.0,
    frozenset({"fricative", "nasal"}): 0.8,
    frozenset({"fricative", "liquid"}): 0.85,
    frozenset({"fricative", "approximant"}): 0.9,
    frozenset({"nasal", "liquid"}): 0.6,
    frozenset({"nasal", "approximant"}): 0.8,
    frozenset({"liquid", "approximant"}): 0.45,
}


@dataclass(frozen=True)
class Consonant:
    place: str
    manner: str
    voiced: bool
    palatalized: bool = False


@dataclass(frozen=True)
class Vowel:
    # 舌の高さ: 0=high, 1=mid, 2=low
    height: int
    # 前後: 0=front, 1=central, 2=back
    backness: int
    rounded: bool


CONSONANTS: dict[str, Consonant] = {
    "k": Consonant("velar", "stop", False),
    "g": Consonant("velar", "stop", True),
    "ky": Consonant("velar", "stop", False, palatalized=True),
    "gy": Consonant("velar", "stop", True, palatalized=True),
    "t": Consonant("alveolar", "stop", False),
    "d": Consonant("alveolar", "stop", True),
    "p": Consonant("bilabial", "stop", False),
    "b": Consonant("bilabial", "stop", True),
    "py": Consonant("bilabial", "stop", False, palatalized=True),
    "by": Consonant("bilabial", "stop", True, palatalized=True),
    "s": Consonant("alveolar", "fricative", False),
    "z": Consonant("alveolar", "fricative", True),
    "sh": Consonant("postalveolar", "fricative", False, palatalized=True),
    "j": Consonant("postalveolar", "affricate", True, palatalized=True),
    "ts": Consonant("alveolar", "affricate", False),
    "ch": Consonant("postalveolar", "affricate", False, palatalized=True),
    "h": Consonant("glottal", "fricative", False),
    "hy": Consonant("palatal", "fricative", False, palatalized=True),
    # 日本語の /f/ (フ・ファ行) は IPA では [ɸ] — 両唇摩擦音であって [f] ではない。
    # labiodental に置くと alveolar と隣接するので f-s が 0.067 まで落ち、「フトン」の
    # 最近傍が「ストン」・「フウフ」が「スウフ」になった。両唇に直すと f-p 0.26 /
    # f-s 0.13 と逆転し、調音を共有する p/b/m/h 側に寄る。
    "f": Consonant("bilabial", "fricative", False),
    "fy": Consonant("bilabial", "fricative", False, palatalized=True),
    # ヴ は借用語の [v] で、こちらは実際に唇歯音。
    "v": Consonant("labiodental", "fricative", True),
    "n": Consonant("alveolar", "nasal", True),
    "ny": Consonant("palatal", "nasal", True, palatalized=True),
    "m": Consonant("bilabial", "nasal", True),
    "my": Consonant("bilabial", "nasal", True, palatalized=True),
    "r": Consonant("alveolar", "liquid", True),
    "ry": Consonant("alveolar", "liquid", True, palatalized=True),
    "y": Consonant("palatal", "approximant", True),
    "w": Consonant("velar", "approximant", True),
}

VOWELS: dict[str, Vowel] = {
    "i": Vowel(height=0, backness=0, rounded=False),
    "e": Vowel(height=1, backness=0, rounded=False),
    "a": Vowel(height=2, backness=1, rounded=False),
    # 日本語の /u/ は円唇性が弱い中舌寄りの母音。
    "o": Vowel(height=1, backness=2, rounded=True),
    "u": Vowel(height=0, backness=2, rounded=False),
}

# --- IPA 表記 ---------------------------------------------------------------

# 音素記号 -> IPA。距離計算には一切使わない表示専用の表だが、**素性表の隣に置く**。
# 音素記号が ASCII のヘボン式寄りなのは `phonology.py` がかなから機械的に写せる
# ためで、その音が実際に何であるかは素性側が持っている (`f` を [ɸ] として
# bilabial に置いたのがその例)。IPA を別ファイルに置くと素性を直したときに
# 表記だけが古いまま残るので、同じ場所で目に入るようにする。
#
# 粒度は音素との 1 対 1 に留める。環境による異音 (ン の [m][n][ŋ]、母音の無声化)
# は写さない — 距離側も異音を区別していないので、表示だけ細かくすると
# 「近い理由」の説明と食い違う。
_IPA: dict[str, str] = {
    # 破裂音・破擦音
    "k": "k",
    "g": "ɡ",
    "ky": "kʲ",
    "gy": "ɡʲ",
    "t": "t",
    "d": "d",
    "p": "p",
    "b": "b",
    "py": "pʲ",
    "by": "bʲ",
    "ts": "t͡s",
    "ch": "t͡ɕ",
    "j": "d͡ʑ",
    # 摩擦音
    "s": "s",
    "z": "z",
    "sh": "ɕ",
    "h": "h",
    "hy": "ç",
    "f": "ɸ",
    "fy": "ɸʲ",
    "v": "v",
    # 鼻音・流音・接近音
    "n": "n",
    "ny": "ɲ",
    "m": "m",
    "my": "mʲ",
    "r": "ɾ",
    "ry": "ɾʲ",
    "y": "j",
    "w": "w",
    # 母音。/u/ は円唇性の弱い中舌寄りなので [ɯ] を当てる (VOWELS の rounded=False
    # と対応する)。
    "a": "a",
    "i": "i",
    "u": "ɯ",
    "e": "e",
    "o": "o",
    # 特殊モーラ。長音は長さ記号そのもの。撥音は語末での実現 [ɴ] を代表に採る
    # (後続子音への同化は上記のとおり写さない)。
    #
    # 促音の単独表記は語末促音 (「あっ」) の声門閉鎖。語中では後続子音の重複に
    # なるので `ipa_transcription` が書き換える (下記)。
    LONG: "ː",
    GEMINATE: "ʔ",
    MORAIC_N: "ɴ",
}


def phoneme_ipa(phoneme: str) -> str:
    """音素記号を IPA に写す。未知の記号はそのまま返す。"""
    return _IPA.get(phoneme, phoneme)


def ipa_transcription(phonemes: tuple[str, ...]) -> str:
    """音素列を IPA の連続表記にする。

    音素の区切りは入れない。IPA は連続した発音の写しなので、空白で区切ると
    音素列 (`phoneme_string()`) と同じものの記号違いに見えてしまう。

    **促音だけは 1 対 1 に写せない。** 促音は後続子音の重複として実現するので
    (キッテ = [kitte])、単独記号の [ʔ] を語中に置くと実際には起きない声門閉鎖を
    書いたことになる。後続音素が見えるこの関数で重複に書き換え、[ʔ] は後続子音が
    無いとき (語末の「あっ」) だけ残す。**表を引くだけにしないのはこのため。**

    長音 [ː] と撥音 [ɴ] は直前・単独で成立するので書き換えない。
    """
    out: list[str] = []
    for i, phoneme in enumerate(phonemes):
        if phoneme == GEMINATE:
            following = phonemes[i + 1] if i + 1 < len(phonemes) else None
            if following in CONSONANTS:
                # 重複させるのは閉鎖・摩擦の本体だけ。口蓋化や破擦音の結合記号まで
                # 二重に書くと [t͡ɕt͡ɕ] のようになり読めないので、先頭の 1 文字を採る。
                out.append(phoneme_ipa(following)[0])
                continue
        out.append(phoneme_ipa(phoneme))
    return "".join(out)


# --- 素性差から距離を作る重み ----------------------------------------------

# 子音の素性ごとの寄与。合計で 1.0 を超えないよう正規化する。
_W_PLACE = 0.40
_W_MANNER = 0.40
_W_VOICE = 0.14
_W_PALATAL = 0.06

# 母音の素性ごとの寄与。
_W_HEIGHT = 0.45
_W_BACK = 0.40
_W_ROUND = 0.15

# 子音と母音、あるいは特殊モーラとの置換は音節構造を壊すため高コスト。
_CROSS_CLASS_COST = 1.0

# 挿入・削除の基本コスト。
_INDEL_COST = 0.9
#: 通常音素の挿入削除コスト。類似度の正規化 (`similarity_normalizer`) に使う。
INDEL_COST = _INDEL_COST
# 特殊モーラ (長音・促音・撥音) の挿入削除は聞き取り差として起きやすい。
_SPECIAL_INDEL_COST = 0.45

# 特殊モーラ同士の置換コスト。
_SPECIAL_SUB = {
    frozenset({LONG, MORAIC_N}): 0.55,
    frozenset({LONG, GEMINATE}): 0.55,
    frozenset({GEMINATE, MORAIC_N}): 0.6,
}


def _manner_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0
    return _MANNER_DISTANCE[frozenset({a, b})]


def _consonant_distance(a: Consonant, b: Consonant) -> float:
    place = abs(_PLACE[a.place] - _PLACE[b.place]) / (len(_PLACE) - 1)
    manner = _manner_distance(a.manner, b.manner)
    voice = 0.0 if a.voiced == b.voiced else 1.0
    palatal = 0.0 if a.palatalized == b.palatalized else 1.0
    return _W_PLACE * place + _W_MANNER * manner + _W_VOICE * voice + _W_PALATAL * palatal


def _vowel_distance(a: Vowel, b: Vowel) -> float:
    height = abs(a.height - b.height) / 2
    back = abs(a.backness - b.backness) / 2
    rounded = 0.0 if a.rounded == b.rounded else 1.0
    return _W_HEIGHT * height + _W_BACK * back + _W_ROUND * rounded


@cache
def phoneme_distance(a: str, b: str) -> float:
    """2 音素間の距離。同一なら 0.0、最大 1.0。"""
    if a == b:
        return 0.0

    a_special = a in (LONG, GEMINATE, MORAIC_N)
    b_special = b in (LONG, GEMINATE, MORAIC_N)
    if a_special or b_special:
        if a_special and b_special:
            return _SPECIAL_SUB.get(frozenset({a, b}), 0.6)
        # 特殊モーラと通常音素の置換。撥音 <-> 鼻子音は近いものとして扱う。
        special, other = (a, b) if a_special else (b, a)
        if special == MORAIC_N:
            cons = CONSONANTS.get(other)
            if cons is not None and cons.manner == "nasal":
                return 0.35
        return _CROSS_CLASS_COST

    a_vowel = a in VOWELS
    b_vowel = b in VOWELS
    if a_vowel and b_vowel:
        return _vowel_distance(VOWELS[a], VOWELS[b])

    a_cons = CONSONANTS.get(a)
    b_cons = CONSONANTS.get(b)
    if a_cons is not None and b_cons is not None:
        return _consonant_distance(a_cons, b_cons)

    # 子音と母音の置換、または未知音素。
    return _CROSS_CLASS_COST


def _worst_substitution_cost() -> float:
    """通常音素同士の置換で実際に到達しうる最大コスト。

    類似度の正規化に使う。素性の重み表を変えても追従するよう定数を埋め込まず
    実際の音素集合から求める。
    """
    symbols = [*CONSONANTS, *VOWELS]
    return max(phoneme_distance(a, b) for a in symbols for b in symbols)


_WORST_SUBSTITUTION = _worst_substitution_cost()

#: 通常音素同士の置換で到達しうる最大コスト。類似度の正規化に使う。
WORST_SUBSTITUTION_COST = _WORST_SUBSTITUTION


def _indel_cost(phoneme: str) -> float:
    if phoneme in (LONG, GEMINATE, MORAIC_N):
        return _SPECIAL_INDEL_COST
    return _INDEL_COST


def weighted_edit_distance(
    a: tuple[str, ...],
    b: tuple[str, ...],
    max_distance: float | None = None,
) -> float:
    """音素列間の重み付き編集距離。

    `max_distance` を渡すと、行ごとの下限がそれを超えた時点で打ち切り、
    `max_distance` より大きい値を返す。上位 N 件だけが必要な検索では
    見込みのない候補に全 DP を回さずに済む。
    """
    if not a:
        return sum(_indel_cost(p) for p in b)
    if not b:
        return sum(_indel_cost(p) for p in a)

    prev = [0.0] * (len(b) + 1)
    for j, pb in enumerate(b, start=1):
        prev[j] = prev[j - 1] + _indel_cost(pb)

    for pa in a:
        cur = [prev[0] + _indel_cost(pa)] + [0.0] * len(b)
        row_min = cur[0]
        for j, pb in enumerate(b, start=1):
            value = min(
                prev[j - 1] + phoneme_distance(pa, pb),
                prev[j] + _indel_cost(pa),
                cur[j - 1] + _indel_cost(pb),
            )
            cur[j] = value
            if value < row_min:
                row_min = value
        # 残りの行でコストが減ることはないので、行の下限が上限を超えたら確定で除外。
        if max_distance is not None and row_min > max_distance:
            return row_min
        prev = cur

    return prev[-1]


# --- ID ベースの高速経路 ---------------------------------------------------
#
# 検索の rerank では 1 クエリあたり数千の候補に編集距離を掛ける。音素記号を
# そのまま扱うと `phoneme_distance` と `_indel_cost` の Python 呼び出しが
# 支配的になる (実測でクエリ時間の半分以上)。音素を整数 ID に落とし、
# 置換コストを (P, P) 行列、挿入削除コストを (P,) 配列に事前計算しておけば
# DP の内側ループが NumPy のベクトル演算 1 回で済む。

_ALL_PHONEMES: tuple[str, ...] = (*CONSONANTS, *VOWELS, LONG, GEMINATE, MORAIC_N)

#: 音素記号 -> ID。未知音素は含まない。
PHONEME_TO_ID: dict[str, int] = {symbol: index for index, symbol in enumerate(_ALL_PHONEMES)}

#: 未知音素に割り当てる ID。どの音素との置換も最大コストになる。
UNKNOWN_PHONEME_ID: int = len(_ALL_PHONEMES)

_PHONEME_COUNT = len(_ALL_PHONEMES) + 1


def _build_cost_tables() -> tuple[np.ndarray, np.ndarray]:
    substitution = np.full((_PHONEME_COUNT, _PHONEME_COUNT), _CROSS_CLASS_COST, dtype=np.float64)
    for i, a in enumerate(_ALL_PHONEMES):
        for j, b in enumerate(_ALL_PHONEMES):
            substitution[i, j] = phoneme_distance(a, b)
    # 未知音素は自分自身とだけ距離 0。
    substitution[UNKNOWN_PHONEME_ID, UNKNOWN_PHONEME_ID] = 0.0

    indel = np.full(_PHONEME_COUNT, _INDEL_COST, dtype=np.float64)
    for symbol in (LONG, GEMINATE, MORAIC_N):
        indel[PHONEME_TO_ID[symbol]] = _SPECIAL_INDEL_COST
    return substitution, indel


SUBSTITUTION_COSTS, INDEL_COSTS = _build_cost_tables()

#: バッチ版が使う float32 のコスト表。
#:
#: バッチ DP は (候補数, 音素長) の行列を音素列の長さぶん更新するので、要素あたり
#: のバイト数がそのまま帯域に効く。float64 から float32 に落とすと 53 万候補で
#: 2034ms -> 761ms (2.7 倍) になった。支配的なのは演算ではなくメモリ転送で、
#: とくに置換コストの (クエリ長, 候補数, 音素長) テンソルが float64 では 306MB に
#: なりキャッシュに全く乗らない。
#:
#: 精度の心配は要らない。コストは 0.0〜1.0 の値を音素長 (最大 24) ぶん足すだけなので
#: 絶対値は 25 未満に収まり、float32 の仮数 24 bit に対して桁落ちの余地がない。
#: 実測の最大誤差は 9e-07 で、検索スコアの丸め (小数 4 桁) には届かない。
#: 一致性は `tests/test_distance.py` が float64 の実装との比較で担保する。
SUBSTITUTION_COSTS32 = SUBSTITUTION_COSTS.astype(np.float32)
INDEL_COSTS32 = INDEL_COSTS.astype(np.float32)

# --- Rust 拡張 (必須) -------------------------------------------------------
#
# 編集距離は Rust で計算する (`rust/`、ビルド手順は README)。
#
# NumPy 実装は持たない。53 万候補で 1097ms 対 41ms (26.5 倍) と差が大きく、
# 両者を揃えて維持する手間に見合わなかった。NumPy では「候補方向をベクトル化して
# DP の 1 行を全候補ぶん進める」形しか取れず、(クエリ長 x 候補数 x 音素長) の
# 置換コストテンソルを実体化してしまう (53 万候補で float32 でも 153MB)。Rust なら
# 1 候補ずつ DP を回して候補方向を並列化できるので、作業配列が L1 に収まり帯域を
# 使わない — 構造的に追いつけない。

# Rust に渡すコスト表。(P, P) を行優先で平坦化したものを毎回作り直さずに持つ。
_SUB_FLAT32 = np.ascontiguousarray(SUBSTITUTION_COSTS32).reshape(-1)
_INDEL_FLAT32 = np.ascontiguousarray(INDEL_COSTS32)


def phoneme_ids(phonemes: tuple[str, ...]) -> np.ndarray:
    """音素列を ID 配列にする。未知音素は UNKNOWN_PHONEME_ID になる。"""
    return np.fromiter(
        (PHONEME_TO_ID.get(p, UNKNOWN_PHONEME_ID) for p in phonemes),
        dtype=np.int32,
        count=len(phonemes),
    )


#: パディング行列で「音素なし」を表す値。
PAD_ID = -1


def edit_distance_batch(
    query_ids: np.ndarray,
    candidates: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """1 つのクエリと複数候補の編集距離をまとめて計算する。

    `candidates` は (C, L) のパディング済み ID 行列で、余りは `PAD_ID`。
    `lengths` は候補ごとの実際の音素数。戻り値は (C,) の距離。

    検索が使うのは `edit_distance_csr` のほうで、こちらは索引を用意せずに
    任意の音素列を渡せる経路として残してある。記号版との一致を検証する足場が
    ここにしかない (CSR 版は索引の持ち方に縛られる)。

    計算は float32 で行う (`SUBSTITUTION_COSTS32` の説明を参照)。
    `weighted_edit_distance` と同じ値を返さなければならない
    (`tests/test_distance.py` が対応を検証する)。
    """
    return _rust.edit_distance_batch(
        np.ascontiguousarray(query_ids, dtype=np.int32),
        np.ascontiguousarray(candidates, dtype=np.int64),
        np.ascontiguousarray(lengths, dtype=np.int64),
        _SUB_FLAT32,
        _INDEL_FLAT32,
        _PHONEME_COUNT,
    )


def edit_distance_csr(
    query_ids: np.ndarray,
    rows: np.ndarray,
    phoneme_ids_blob: np.ndarray,
    phoneme_bounds: np.ndarray,
    distance_ids: np.ndarray,
) -> np.ndarray:
    """索引の CSR から直接、候補行の編集距離をまとめて計算する。**検索の本線。**

    索引は音素列を連結した 1 本の配列と境界インデックスで持っているので
    (`store.py` の `_encode_entries`)、それを Rust にそのまま渡せば
    (候補数, 音素長) の行列を Python 側で組む必要がなくなる。実測で 53 万候補の
    行列作成に 102ms かかっていた分がまるごと消える。
    """
    return _rust.edit_distance_csr(
        np.ascontiguousarray(query_ids, dtype=np.int32),
        np.ascontiguousarray(rows, dtype=np.int64),
        np.ascontiguousarray(phoneme_ids_blob, dtype=np.uint8),
        np.ascontiguousarray(phoneme_bounds, dtype=np.int32),
        np.ascontiguousarray(distance_ids, dtype=np.int32),
        _SUB_FLAT32,
        _INDEL_FLAT32,
        _PHONEME_COUNT,
    )


#: 包含判定で挿入を無料にする音素 (長音・促音・撥音)。
#:
#: この 3 つは単独で音素を成さず、前続や後続の伸縮として実現する。「リンゴー」
#: 「リンゴッ」は記号列としては riNgo と一致しないが、音としては riNgo を
#: 完全に含んでいる。それ以外の音素が挟まれば別の音になるので許さない。
ELASTIC_PHONEMES = (LONG, GEMINATE, MORAIC_N)

_ELASTIC_IDS = np.array([PHONEME_TO_ID[symbol] for symbol in ELASTIC_PHONEMES], dtype=np.int32)


def containment_ratio(query: tuple[str, ...], candidate: tuple[str, ...]) -> float:
    """`candidate` が `query` を完全な形で含むなら占有率、含まないなら 0.0。

    **編集距離では表現できない性質**を測る。距離は挿入を一律に減点するので、
    クエリを丸ごと含む語 (riNgo in riNgoku、類似度 0.735) が 1 音素だけ違う
    同じ長さの語 (riNbo、0.933) に必ず負ける。「入っているかどうか」は連続一致
    という離散的な性質なので、距離の重みを緩めて近似するのではなく判定として持つ。

    占有率はクエリの音素数を**候補全体の音素数**で割った値。余分が多い語ほど
    下がるので、短いクエリを含む長い地名が上位に来るのを抑える。分母を
    「一致に消費した長さ」にすると特殊モーラを挟んだ語が 1.0 になり、余分の
    多寡が消えてしまう。

    特殊モーラの挿入だけは無料 (`ELASTIC_PHONEMES`)。

    **Rust 版 (`containment_scan`) と同じ判定を返さなければならない**
    (`tests/test_distance.py` が対応を検証する)。こちらは記号のまま書いた
    参照実装で、検索が通るのは Rust 側。
    """
    if not query or len(candidate) < len(query):
        return 0.0
    elastic = frozenset(ELASTIC_PHONEMES)
    for offset in range(len(candidate) - len(query) + 1):
        if candidate[offset] != query[0]:
            continue
        matched = 0
        position = offset
        while matched < len(query) and position < len(candidate):
            symbol = candidate[position]
            if symbol == query[matched]:
                matched += 1
                position += 1
            elif matched > 0 and symbol in elastic:
                # 途中に挟まった特殊モーラは飛ばす。先頭より前の特殊モーラは
                # 開始位置の走査が担うので、ここでは matched > 0 のときだけ許す。
                position += 1
            else:
                break
        if matched == len(query):
            return len(query) / len(candidate)
    return 0.0


def containment_scan(
    query_ids: np.ndarray,
    phoneme_ids_blob: np.ndarray,
    phoneme_bounds: np.ndarray,
    distance_ids: np.ndarray,
    group_start: int,
    group_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """索引の CSR を走査し、クエリを完全な形で含むグループと占有率を返す。

    **候補生成の Top-K では拾えないので別の経路にしてある。** 実測で「りんご」
    を含む 204 グループのうち phonetic 空間の Top-8000 に入るのは 48 件しか
    ない (モーラ帯に限っても 123 件のうち 48 件)。包含は phonetic 空間の
    近さと相関しないため、候補を後段で絞る形では解決しない。
    """
    return _rust.containment_scan(
        np.ascontiguousarray(query_ids, dtype=np.int32),
        np.ascontiguousarray(phoneme_ids_blob, dtype=np.uint8),
        np.ascontiguousarray(phoneme_bounds, dtype=np.int32),
        np.ascontiguousarray(distance_ids, dtype=np.int32),
        _ELASTIC_IDS,
        int(group_start),
        int(group_end),
    )


def similarity_normalizer(len_a: int, len_b: int, worst: float = -1.0) -> float:
    """類似度の分母 = その長さの組が取り得る最悪の距離。

    短い方を全置換し、長さの余りを挿入するコストに相当する。素性ベースの置換は
    距離が 1.0 に達しないため、max(len) をそのまま分母にすると無関係語でも
    0.5 前後に張り付き、スコアの解像度が失われる。
    """
    if worst < 0.0:
        worst = _WORST_SUBSTITUTION
    shorter, longer = min(len_a, len_b), max(len_a, len_b)
    return shorter * worst + (longer - shorter) * _INDEL_COST


#: アライメント 1 対。`op` は "match" (同一) / "sub" (置換) / "del" (a 側のみ)
#: / "ins" (b 側のみ)。存在しない側は None。
AlignedPair = tuple[str | None, str | None, float, str]


def align_phonemes(a: tuple[str, ...], b: tuple[str, ...]) -> list[AlignedPair]:
    """音素列を `weighted_edit_distance` と同じコストで対応付ける。

    距離が返すのはスカラー 1 つだが、「どこがどう違うのか」を示すには
    どの音素がどれに対応したかが必要になる。同じ DP を回して経路を復元する。
    コスト定義を共有しているので、対の距離の総和は編集距離に一致する。
    """
    n, m = len(a), len(b)
    # dp[i][j] = a[:i] と b[:j] の距離。
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i, pa in enumerate(a, start=1):
        dp[i][0] = dp[i - 1][0] + _indel_cost(pa)
    for j, pb in enumerate(b, start=1):
        dp[0][j] = dp[0][j - 1] + _indel_cost(pb)
    for i, pa in enumerate(a, start=1):
        for j, pb in enumerate(b, start=1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + phoneme_distance(pa, pb),
                dp[i - 1][j] + _indel_cost(pa),
                dp[i][j - 1] + _indel_cost(pb),
            )

    pairs: list[AlignedPair] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = phoneme_distance(a[i - 1], b[j - 1])
            if abs(dp[i][j] - (dp[i - 1][j - 1] + cost)) < 1e-9:
                op = "match" if a[i - 1] == b[j - 1] else "sub"
                pairs.append((a[i - 1], b[j - 1], cost, op))
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + _indel_cost(a[i - 1]))) < 1e-9:
            pairs.append((a[i - 1], None, _indel_cost(a[i - 1]), "del"))
            i -= 1
            continue
        pairs.append((None, b[j - 1], _indel_cost(b[j - 1]), "ins"))
        j -= 1

    pairs.reverse()
    return pairs


def phonetic_similarity(a: Pronunciation, b: Pronunciation) -> float:
    """音韻類似度を 0.0〜1.0 で返す。"""
    return _sequence_similarity(a.phonemes, b.phonemes)


def vowel_skeleton_similarity(a: Pronunciation, b: Pronunciation) -> float:
    """母音骨格 (母音 + 特殊モーラの列) の類似度を 0.0〜1.0 で返す。

    ダジャレ・韻のコーパス研究は一貫して「母音列の一致が日本語の音類似の
    第一法則で、子音の違いは類似度に応じて許容される」ことを示している
    (Kawahara 2007, Kawahara & Shinohara 2009)。この性質は列の照合であって、
    ビンにプーリングしたベクトルの内積では表現できない — プーリングは長さの
    情報を捨てるので、「カイギ」(a,i,i) と「カタギリシキ」(a,a,i,i,i,i) の
    ような長さ違いに 0.99 を与えてしまった。

    骨格の記号 (a/i/u/e/o/N/Q) はすべて音素なので、DP・コスト表・正規化は
    音素列の類似度とそのまま共有する。長さの不一致は挿入コストとして積み上がり、
    列が完全一致なら 1.0 になる。

    検索の rerank が通るのは索引の母音骨格 CSR + Rust の
    `edit_distance_csr` で、**こちらは記号のまま書いた参照実装**
    (`tests/test_search.py` が両者の一致を検証する)。
    """
    return _sequence_similarity(a.vowel_skeleton, b.vowel_skeleton)


def _sequence_similarity(pa: tuple[str, ...], pb: tuple[str, ...]) -> float:
    """音素記号列の重み付き編集距離を長さで正規化した類似度。"""
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0

    distance = weighted_edit_distance(pa, pb)
    denominator = similarity_normalizer(len(pa), len(pb))
    if denominator <= 0.0:
        return 1.0

    return max(0.0, 1.0 - distance / denominator)

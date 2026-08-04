"""音素間距離と weighted Levenshtein による音韻類似度。

音素を記号として扱わず、音声学的素性 (調音位置・調音方法・有声性・母音の
高さ/前後/円唇性) のベクトルとして表現し、素性の不一致から距離を導く。
これにより k/g のような有声性のみが違う対は近く、k/m のように調音方法まで
違う対は遠くなる。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

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
    "f": Consonant("labiodental", "fricative", False),
    "fy": Consonant("labiodental", "fricative", False, palatalized=True),
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


def phoneme_ids(phonemes: tuple[str, ...]) -> np.ndarray:
    """音素列を ID 配列にする。未知音素は UNKNOWN_PHONEME_ID になる。"""
    return np.fromiter(
        (PHONEME_TO_ID.get(p, UNKNOWN_PHONEME_ID) for p in phonemes),
        dtype=np.int32,
        count=len(phonemes),
    )


def edit_distance_ids(a: np.ndarray, b: np.ndarray) -> float:
    """ID 配列間の重み付き編集距離。

    `weighted_edit_distance` と同じ値を返すが、行ごとに NumPy のベクトル演算で
    処理する。挿入方向の依存 (cur[j] が cur[j-1] を参照する) だけは逐次なので、
    そこは行内の走査で解く。
    """
    if a.size == 0:
        return float(INDEL_COSTS[b].sum())
    if b.size == 0:
        return float(INDEL_COSTS[a].sum())

    b_indel = INDEL_COSTS[b]
    prev = np.empty(b.size + 1, dtype=np.float64)
    prev[0] = 0.0
    np.cumsum(b_indel, out=prev[1:])

    substitution_rows = SUBSTITUTION_COSTS[a][:, b]
    a_indel = INDEL_COSTS[a]

    cur = np.empty_like(prev)
    for row in range(a.size):
        cur[0] = prev[0] + a_indel[row]
        # 置換と削除は前の行だけに依存するのでまとめて計算できる。
        np.minimum(prev[:-1] + substitution_rows[row], prev[1:] + a_indel[row], out=cur[1:])
        # 挿入は同じ行の左隣に依存するため逐次に解く。
        for column in range(1, cur.size):
            candidate = cur[column - 1] + b_indel[column - 1]
            if candidate < cur[column]:
                cur[column] = candidate
        prev, cur = cur, prev

    return float(prev[-1])


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


def phonetic_similarity(a: Pronunciation, b: Pronunciation) -> float:
    """音韻類似度を 0.0〜1.0 で返す。"""
    pa, pb = a.phonemes, b.phonemes
    if not pa and not pb:
        return 1.0
    if not pa or not pb:
        return 0.0

    distance = weighted_edit_distance(pa, pb)
    denominator = similarity_normalizer(len(pa), len(pb))
    if denominator <= 0.0:
        return 1.0

    return max(0.0, 1.0 - distance / denominator)

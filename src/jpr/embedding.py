"""音素列を固定長ベクトルに変換する (phonetic embedding)。

ニューラルな学習は行わず、音声学的素性から決定的に組み立てる。学習済み
モデルを持たなくても再現でき、次元の意味が説明できるのが利点。

複数の空間を用意しているのが要点。「韻を踏む語」は語尾が重要で、
「聞き間違えやすい語」は全体の音韻が重要、と用途によって近さの定義が変わる
ため、単一のベクトルに混ぜず別々に持って検索時に重みを変える。

    phonetic  : 全体の音韻。既定の検索軸。
    consonant : 子音の骨格のみ。母音が違っても子音が揃う語を拾う。
    vowel     : 母音の骨格のみ。母音列が似た語 (韻) を拾う。
    rhythm    : モーラ数と特殊モーラの配置。リズムの近さ。
    coda      : 語尾 2 モーラ。韻を踏む用途で最も効く。

いずれも L2 正規化して返すので、内積 = コサイン類似度として扱える。
"""

from __future__ import annotations

from typing import Final

import numpy as np

from .distance import CONSONANTS, VOWELS, Consonant, Vowel
from .phonology import GEMINATE, LONG, MORAIC_N, Pronunciation

# --- 音素 1 つを表す素性ベクトル ------------------------------------------

# 子音: 調音位置 (one-hot 7) + 調音方法 (one-hot 6) + 有声性 + 口蓋化
_PLACES: Final = (
    "bilabial",
    "labiodental",
    "alveolar",
    "postalveolar",
    "palatal",
    "velar",
    "glottal",
)
_MANNERS: Final = ("stop", "affricate", "fricative", "nasal", "liquid", "approximant")

# 母音: 高さ + 前後 + 円唇 (連続値)
_CONSONANT_DIM = len(_PLACES) + len(_MANNERS) + 2
_VOWEL_DIM = 3
# 特殊モーラ (長音・促音・撥音) の one-hot
_SPECIAL_DIM = 3
# 音素種別のフラグ (子音 / 母音 / 特殊)
_KIND_DIM = 3

PHONEME_DIM: Final = _CONSONANT_DIM + _VOWEL_DIM + _SPECIAL_DIM + _KIND_DIM

_SPECIALS: Final = (LONG, GEMINATE, MORAIC_N)


def _consonant_vector(consonant: Consonant) -> np.ndarray:
    vector = np.zeros(_CONSONANT_DIM, dtype=np.float32)
    vector[_PLACES.index(consonant.place)] = 1.0
    vector[len(_PLACES) + _MANNERS.index(consonant.manner)] = 1.0
    vector[len(_PLACES) + len(_MANNERS)] = 1.0 if consonant.voiced else 0.0
    vector[len(_PLACES) + len(_MANNERS) + 1] = 1.0 if consonant.palatalized else 0.0
    return vector


def _vowel_vector(vowel: Vowel) -> np.ndarray:
    # 高さ・前後は 0..2 を 0..1 に正規化する。
    return np.array(
        [vowel.height / 2.0, vowel.backness / 2.0, 1.0 if vowel.rounded else 0.0],
        dtype=np.float32,
    )


def _build_phoneme_table() -> tuple[dict[str, int], np.ndarray]:
    """音素記号 -> 行番号 の索引と、素性行列を作る。"""
    symbols = [*CONSONANTS, *VOWELS, *_SPECIALS]
    table = np.zeros((len(symbols), PHONEME_DIM), dtype=np.float32)

    consonant_end = _CONSONANT_DIM
    vowel_end = consonant_end + _VOWEL_DIM
    special_end = vowel_end + _SPECIAL_DIM

    for row, symbol in enumerate(symbols):
        if symbol in CONSONANTS:
            table[row, :consonant_end] = _consonant_vector(CONSONANTS[symbol])
            table[row, special_end] = 1.0
        elif symbol in VOWELS:
            table[row, consonant_end:vowel_end] = _vowel_vector(VOWELS[symbol])
            table[row, special_end + 1] = 1.0
        else:
            table[row, vowel_end + _SPECIALS.index(symbol)] = 1.0
            table[row, special_end + 2] = 1.0

    return {symbol: row for row, symbol in enumerate(symbols)}, table


PHONEME_INDEX, PHONEME_FEATURES = _build_phoneme_table()

#: 音素記号 -> 整数 ID。未知音素は -1。
PHONEME_IDS: Final[dict[str, int]] = dict(PHONEME_INDEX)


# --- 位置エンコーディング --------------------------------------------------

#: 音素列を畳み込む際の位置ビンの分割数。系列を等間隔のビンに分けて素性を平均し、
#: 複数の粒度を連結する (マルチスケールプーリング)。
#:
#: 単一の粗い粒度 (1 ビン = 全体平均) では語順が消え、アナグラムが同一視される。
#: 逆に細かい粒度だけ (4 ビン) にすると語頭の 1 音素の差でビンが丸ごと外れ、
#: 「チクビ」と「テクビ」の類似度が 0.68 まで落ちて ANN の候補にすら入らなかった。
#: 全体 (1) と前後半 (2) を併せ持つことで、語順を保ちながら局所的な差に頑健になる。
#: 細かい位置合わせは rerank 段の編集距離が担う。
POSITION_SCALES: Final = (1, 2)
POSITION_BINS: Final = sum(POSITION_SCALES)

#: 語尾ベクトルで見るモーラ数。
CODA_MORAS: Final = 2


#: 系列長ごとのプーリング行列を事前計算しておく上限。日本語の語で
#: これを超える音素長は稀なので、超えた分は都度計算する。
_MAX_POOL_LENGTH: Final = 64


def _pooling_matrix(length: int, bins: int) -> np.ndarray:
    """(bins, length) の平均プーリング行列を作る。

    行 i がビン i の担当範囲を 1/幅 で重み付けした行列。これを素性列に
    左から掛けるとビンごとの平均が得られる。
    """
    matrix = np.zeros((bins, length), dtype=np.float32)
    edges = np.linspace(0, length, bins + 1)
    for index in range(bins):
        start, end = int(edges[index]), int(edges[index + 1])
        if end <= start:
            # ビン数より系列が短い場合は最近傍の 1 要素を割り当てる。
            start = min(start, length - 1)
            end = start + 1
        matrix[index, start:end] = 1.0 / (end - start)
    return matrix


def _multiscale_matrix(length: int) -> np.ndarray:
    """複数の粒度のプーリング行列を縦に積んだ (POSITION_BINS, length) 行列。"""
    return np.vstack([_pooling_matrix(length, bins) for bins in POSITION_SCALES])


_POOLING_CACHE: Final = (
    None,
    *(_multiscale_matrix(length) for length in range(1, _MAX_POOL_LENGTH + 1)),
)


def _compute_bin_ranges(length: int, bins: int) -> tuple[tuple[int, int], ...]:
    """系列を bins 個に分ける (start, end) の組を返す。"""
    ranges: list[tuple[int, int]] = []
    edges = np.linspace(0, length, bins + 1)
    for index in range(bins):
        start, end = int(edges[index]), int(edges[index + 1])
        if end <= start:
            start = min(start, length - 1)
            end = start + 1
        ranges.append((start, end))
    return tuple(ranges)


#: リズムベクトルで特殊モーラの位置分布を見る区間数。ここは位置の細かさが
#: 意味を持つので、プーリングのマルチスケールとは独立に固定粒度で分ける。
RHYTHM_BINS: Final = 4

_BIN_RANGE_CACHE: Final = (
    (),
    *(_compute_bin_ranges(length, RHYTHM_BINS) for length in range(1, _MAX_POOL_LENGTH + 1)),
)


def _bin_ranges(length: int) -> tuple[tuple[int, int], ...]:
    if length <= _MAX_POOL_LENGTH:
        return _BIN_RANGE_CACHE[length]
    return _compute_bin_ranges(length, RHYTHM_BINS)


def _bin_pool(features: np.ndarray) -> np.ndarray:
    """(L, D) の素性列を (POSITION_BINS, D) に畳み込む。

    単純な総和では語順が消え、「チクビ」と「ビクチ」が同一になる。系列を
    等間隔のビンに分けて平均することで、おおよその位置情報を残す。
    複数の粒度を積むことで語順の保持と局所差への頑健さを両立させる。

    ビンごとに `.mean()` を呼ぶと微小配列に対する NumPy 呼び出しが支配的に
    なるため、事前計算したプーリング行列との 1 回の積で済ませる。
    """
    length = features.shape[0]
    if length == 0:
        return np.zeros((POSITION_BINS, features.shape[1]), dtype=np.float32)

    matrix = _POOLING_CACHE[length] if length <= _MAX_POOL_LENGTH else _multiscale_matrix(length)
    return matrix @ features


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _feature_matrix(phonemes: tuple[str, ...]) -> np.ndarray:
    """音素列を (L, PHONEME_DIM) の素性行列にする。未知音素は落とす。"""
    rows = [PHONEME_INDEX[p] for p in phonemes if p in PHONEME_INDEX]
    if not rows:
        return np.zeros((0, PHONEME_DIM), dtype=np.float32)
    return PHONEME_FEATURES[rows]


# --- 各空間の次元 ----------------------------------------------------------

PHONETIC_DIM: Final = PHONEME_DIM * POSITION_BINS
CONSONANT_DIM: Final = _CONSONANT_DIM * POSITION_BINS
VOWEL_DIM: Final = _VOWEL_DIM * POSITION_BINS
CODA_DIM: Final = PHONEME_DIM * CODA_MORAS
# リズム: モーラ数 (連続) + 特殊モーラの比率 3 種 + 位置ごとの特殊モーラ有無
RHYTHM_DIM: Final = 1 + 3 + RHYTHM_BINS

#: リズムベクトルでモーラ数を正規化する上限。これを超える語は飽和させる。
_MAX_MORA_SCALE: Final = 12.0


def phonetic_vector(phonemes: tuple[str, ...]) -> np.ndarray:
    """全体の音韻を表すベクトル。"""
    features = _feature_matrix(phonemes)
    return _normalize(_bin_pool(features).reshape(-1))


def consonant_vector(phonemes: tuple[str, ...]) -> np.ndarray:
    """子音の骨格のみを表すベクトル。"""
    rows = [PHONEME_INDEX[p] for p in phonemes if p in CONSONANTS]
    if not rows:
        return np.zeros(CONSONANT_DIM, dtype=np.float32)
    features = PHONEME_FEATURES[rows][:, :_CONSONANT_DIM]
    return _normalize(_bin_pool(features).reshape(-1))


def vowel_vector(pronunciation: Pronunciation) -> np.ndarray:
    """母音の骨格 (韻) を表すベクトル。

    長音は直前の母音の伸長なので、母音列としては直前母音の繰り返しとみなす。
    `Pronunciation.vowel_skeleton` がその解釈を持つのでそれを使う。
    """
    skeleton = pronunciation.vowel_skeleton
    rows = [PHONEME_INDEX[v] for v in skeleton if v in VOWELS]
    if not rows:
        return np.zeros(VOWEL_DIM, dtype=np.float32)
    start = _CONSONANT_DIM
    features = PHONEME_FEATURES[rows][:, start : start + _VOWEL_DIM]
    return _normalize(_bin_pool(features).reshape(-1))


def coda_vector(pronunciation: Pronunciation) -> np.ndarray:
    """語尾 CODA_MORAS モーラ分のベクトル。韻を踏む用途で効く。"""
    tail = pronunciation.moras[-CODA_MORAS:]
    vector = np.zeros((CODA_MORAS, PHONEME_DIM), dtype=np.float32)
    # 語尾を右詰めで置く。短い語では先頭側が 0 のまま残る。
    offset = CODA_MORAS - len(tail)
    for index, mora in enumerate(tail):
        features = _feature_matrix(mora.phonemes)
        if features.shape[0]:
            vector[offset + index] = features.mean(axis=0)
    return _normalize(vector.reshape(-1))


def rhythm_vector(pronunciation: Pronunciation) -> np.ndarray:
    """モーラ数と特殊モーラの配置を表すベクトル。

    L2 正規化はしない。この空間は「3 モーラか 4 モーラか」という絶対量が
    意味を持つので、正規化すると特殊モーラを持たない語がすべて同一方向に
    潰れてしまう。内積ではなくユークリッド距離で比較する前提のベクトル。
    """
    moras = pronunciation.moras
    vector = np.zeros(RHYTHM_DIM, dtype=np.float32)
    if not moras:
        return vector

    count = len(moras)
    vector[0] = min(count, _MAX_MORA_SCALE) / _MAX_MORA_SCALE
    for index, special in enumerate(_SPECIALS):
        vector[1 + index] = sum(1 for m in moras if m.special == special) / count

    # 特殊モーラが系列のどのあたりに出るかをビンごとに記録する。
    for bin_index, (start, end) in enumerate(_bin_ranges(count)):
        window = moras[start:end]
        vector[4 + bin_index] = sum(1 for m in window if m.special) / len(window)

    return vector


#: 各空間の名前と次元。索引の構築側と検索側で共有する。
SPACES: Final[dict[str, int]] = {
    "phonetic": PHONETIC_DIM,
    "consonant": CONSONANT_DIM,
    "vowel": VOWEL_DIM,
    "coda": CODA_DIM,
    "rhythm": RHYTHM_DIM,
}


def embed(pronunciation: Pronunciation) -> dict[str, np.ndarray]:
    """1 語のすべての空間のベクトルを作る。

    phonetic と consonant は同じ素性行列から作れるので、行列引きを 1 回に
    まとめる。索引構築では 200 万語を処理するのでこの差が効く。
    """
    phonemes = pronunciation.phonemes
    rows = [PHONEME_INDEX[p] for p in phonemes if p in PHONEME_INDEX]
    features = PHONEME_FEATURES[rows] if rows else np.zeros((0, PHONEME_DIM), dtype=np.float32)

    phonetic = _normalize(_bin_pool(features).reshape(-1))

    # 子音空間は素性行列の子音ブロックだけを、子音の行に限って畳み込む。
    consonant_rows = [PHONEME_INDEX[p] for p in phonemes if p in CONSONANTS]
    if consonant_rows:
        consonant_features = PHONEME_FEATURES[consonant_rows][:, :_CONSONANT_DIM]
        consonant = _normalize(_bin_pool(consonant_features).reshape(-1))
    else:
        consonant = np.zeros(CONSONANT_DIM, dtype=np.float32)

    return {
        "phonetic": phonetic,
        "consonant": consonant,
        "vowel": vowel_vector(pronunciation),
        "coda": coda_vector(pronunciation),
        "rhythm": rhythm_vector(pronunciation),
    }


def embed_many(pronunciations: list[Pronunciation]) -> dict[str, np.ndarray]:
    """複数語をまとめて埋め込む。

    索引構築では 200 万語を処理するため、1 語ずつ `embed` を呼ぶと Python の
    関数呼び出しが支配的になる。空間ごとに (N, D) の配列へ直接書き込む。
    """
    count = len(pronunciations)
    out = {name: np.zeros((count, dim), dtype=np.float32) for name, dim in SPACES.items()}
    for row, pronunciation in enumerate(pronunciations):
        vectors = embed(pronunciation)
        for name, vector in vectors.items():
            out[name][row] = vector
    return out


__all__ = [
    "CODA_MORAS",
    "PHONEME_DIM",
    "POSITION_BINS",
    "POSITION_SCALES",
    "RHYTHM_BINS",
    "SPACES",
    "coda_vector",
    "consonant_vector",
    "embed",
    "embed_many",
    "phonetic_vector",
    "rhythm_vector",
    "vowel_vector",
]

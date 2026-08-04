"""phonetic embedding のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from jpr.embedding import SPACES, embed
from jpr.phonology import analyze_reading


def vectors(reading: str) -> dict[str, np.ndarray]:
    return embed(analyze_reading(reading))


def cosine(a: str, b: str, space: str) -> float:
    return float(vectors(a)[space] @ vectors(b)[space])


def test_all_spaces_have_declared_dimensions() -> None:
    result = vectors("チクビ")
    assert set(result) == set(SPACES)
    for name, dim in SPACES.items():
        assert result[name].shape == (dim,)


@pytest.mark.parametrize("space", ["phonetic", "consonant", "vowel", "coda"])
def test_inner_product_spaces_are_l2_normalized(space: str) -> None:
    """内積を類似度として使う空間は正規化されている必要がある。"""
    norm = float(np.linalg.norm(vectors("チクビ")[space]))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_rhythm_space_is_not_normalized() -> None:
    """リズム空間はモーラ数の絶対量が意味を持つので正規化しない。

    正規化すると特殊モーラを持たない語がすべて同じ方向に潰れてしまう。
    """
    three = vectors("チクビ")["rhythm"]
    five = vectors("コンニチハ")["rhythm"]
    assert not np.allclose(three, five)


def test_identical_readings_are_identical_vectors() -> None:
    for space in SPACES:
        assert np.allclose(vectors("ラーメン")[space], vectors("ラーメン")[space])


def test_phonetic_space_ranks_flagship_pair_high() -> None:
    """ANN の候補生成が成立する前提。編集距離では埋もれる語を拾える。"""
    assert cosine("チクビ", "チョコビ", "phonetic") > 0.85
    assert cosine("チクビ", "デンシャ", "phonetic") < 0.5


def test_consonant_space_ignores_vowels() -> None:
    """子音が同一で母音だけ違う語は、子音空間では最大類似になる。"""
    assert cosine("チクビ", "チョコビ", "consonant") == pytest.approx(1.0, abs=1e-5)


def test_vowel_space_ignores_consonants() -> None:
    """母音列が同じ語は、母音空間では最大類似になる。"""
    assert cosine("チクビ", "シクミ", "vowel") == pytest.approx(1.0, abs=1e-5)


def test_vowel_space_separates_different_vowel_patterns() -> None:
    assert cosine("チクビ", "チョコビ", "vowel") < cosine("チクビ", "テクビ", "vowel")


def test_coda_space_rewards_matching_tail() -> None:
    """語尾が一致する語は語尾空間で最大類似になる。韻の検索で効く。"""
    assert cosine("チクビ", "テクビ", "coda") == pytest.approx(1.0, abs=1e-5)


def test_position_information_is_preserved() -> None:
    """音素の並び順が違えば別のベクトルになる。

    素性を単純に総和すると語順が消え、アナグラムが同一視されてしまう。
    """
    assert cosine("チクビ", "ビクチ", "phonetic") < 0.99


def test_empty_reading_yields_zero_vectors() -> None:
    result = embed(analyze_reading(""))
    for space in SPACES:
        assert np.allclose(result[space], 0.0)

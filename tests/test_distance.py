"""音素距離と音韻類似度のテスト。"""

from __future__ import annotations

import pytest

from jpr.distance import (
    WORST_SUBSTITUTION_COST,
    align_phonemes,
    phoneme_distance,
    phonetic_similarity,
    weighted_edit_distance,
)
from jpr.phonology import analyze_reading


def similarity(a: str, b: str) -> float:
    return phonetic_similarity(analyze_reading(a), analyze_reading(b))


def test_identical_phonemes_have_zero_distance() -> None:
    assert phoneme_distance("k", "k") == 0.0


def test_distance_is_symmetric() -> None:
    for a, b in [("k", "g"), ("i", "e"), ("sh", "ch"), ("N", "n")]:
        assert phoneme_distance(a, b) == phoneme_distance(b, a)


@pytest.mark.parametrize(
    ("near", "far"),
    [
        # 有声性のみの違いは、調音方法まで違う対より近い。
        (("k", "g"), ("k", "m")),
        (("t", "d"), ("t", "n")),
        # 摩擦音同士は、摩擦音と流音より近い。
        (("s", "sh"), ("s", "r")),
        # 母音の高さが 1 段違いは、前後まで違う対より近い。
        (("i", "e"), ("i", "o")),
        # 撥音は鼻子音に近く、それ以外の子音より近い。
        (("N", "n"), ("N", "k")),
    ],
)
def test_relative_ordering(near: tuple[str, str], far: tuple[str, str]) -> None:
    assert phoneme_distance(*near) < phoneme_distance(*far)


def test_consonant_vowel_substitution_is_maximally_costly() -> None:
    """子音と母音の置換は音節構造を壊すので最大コスト。"""
    assert phoneme_distance("k", "a") == pytest.approx(1.0)


def test_identical_sequences_are_fully_similar() -> None:
    assert similarity("ラーメン", "ラーメン") == pytest.approx(1.0)


def test_single_voicing_difference_stays_high() -> None:
    """1 音素の有声性だけが違う語は非常に似ていると判定される。"""
    assert similarity("カガク", "カカク") > 0.95


def test_unrelated_words_score_low() -> None:
    """無関係な語は明確に低い。スコアの解像度が保たれていることの確認。"""
    assert similarity("チクビ", "デンシャ") < 0.45


def test_flagship_pair_similarity() -> None:
    """コンセプトの中心例。意味は無関係だが音は近い。"""
    assert similarity("チクビ", "チョコビ") > 0.75


def test_similarity_ordering_matches_intuition() -> None:
    assert similarity("チクビ", "テクビ") > similarity("チクビ", "マツタケ")
    assert similarity("チクビ", "チクワ") > similarity("チクビ", "ソラ")


def test_empty_input() -> None:
    assert similarity("", "") == pytest.approx(1.0)
    assert similarity("チクビ", "") == 0.0


def test_edit_distance_respects_max_distance_cutoff() -> None:
    """打ち切りを渡しても、上限内の距離は正確な値と一致する。"""
    a = analyze_reading("チクビ").phonemes
    b = analyze_reading("チョコビ").phonemes
    exact = weighted_edit_distance(a, b)

    # 上限が十分大きければ厳密な値を返す。
    assert weighted_edit_distance(a, b, exact + 1.0) == pytest.approx(exact)
    # 上限を下回る場合は上限超えの値を返し、除外の判断に使える。
    assert weighted_edit_distance(a, b, exact - 0.5) > exact - 0.5


def test_worst_substitution_cost_is_bounded() -> None:
    assert 0.0 < WORST_SUBSTITUTION_COST <= 1.0


#: アライメントは編集距離の内訳を見せるためのものなので、対ごとのコストの
#: 総和が距離と一致しなければ表示が嘘になる。距離の重みを変えたときに
#: 両者が食い違わないことをここで担保する。
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("チクビ", "テクビ"),
        ("チクビ", "チョコビ"),
        ("カガク", "カカク"),
        ("サカナ", "アカ"),
        ("ラーメン", "ローメン"),
        ("マツタケ", "ソラ"),
        ("", "アイ"),
        ("ア", ""),
    ],
)
def test_alignment_costs_sum_to_edit_distance(a: str, b: str) -> None:
    pa = analyze_reading(a).phonemes
    pb = analyze_reading(b).phonemes
    pairs = align_phonemes(pa, pb)
    assert sum(cost for _, _, cost, _ in pairs) == pytest.approx(weighted_edit_distance(pa, pb))


def test_alignment_preserves_both_sequences() -> None:
    """対応付けから両方の音素列を復元できる (取りこぼし・重複がない)。"""
    pa = analyze_reading("チクビ").phonemes
    pb = analyze_reading("チョコビ").phonemes
    pairs = align_phonemes(pa, pb)
    assert tuple(x for x, _, _, _ in pairs if x is not None) == pa
    assert tuple(y for _, y, _, _ in pairs if y is not None) == pb


def test_alignment_labels_operations() -> None:
    pairs = align_phonemes(("ch", "i", "k", "u"), ("t", "i", "k", "u"))
    assert [op for _, _, _, op in pairs] == ["sub", "match", "match", "match"]

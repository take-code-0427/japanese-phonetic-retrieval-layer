"""音素距離と音韻類似度のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from jpr.distance import (
    PAD_ID,
    WORST_SUBSTITUTION_COST,
    align_phonemes,
    edit_distance_batch,
    edit_distance_ids,
    phoneme_distance,
    phoneme_ids,
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


#: 同じ距離関数に 3 つの実装がある。記号ベース (可読性優先)、ID ベース
#: (rerank の 1 件経路)、バッチ (rerank の全件経路)。速度のために別実装を
#: 持っているだけなので、値が食い違えば検索結果が実装依存になる。
#: 素性表や重みを変えたときに 3 者が揃っていることをここで担保する。
#:
#: 下のアライメントのテストと前半の対が重なっているが、意図的に別に持つ。
#: あちらは空文字列の対を含み (こちらは空入力を専用のテストで見る)、
#: 検証したい性質も違う (実装間の一致 / コスト総和と距離の一致) ので、
#: 片方の都合で対を足したときにもう片方が巻き込まれないようにしている。
_DISTANCE_PAIRS = [
    ("チクビ", "テクビ"),
    ("チクビ", "チョコビ"),
    ("カガク", "カカク"),
    ("サカナ", "アカ"),
    ("ラーメン", "ローメン"),
    ("マツタケ", "ソラ"),
    ("トウキョウ", "トウギョウ"),
    ("ガッコウ", "ガクコウ"),
    ("アリガトウ", "アリタソウ"),
    ("ア", "アイウエオ"),
]


@pytest.mark.parametrize(("a", "b"), _DISTANCE_PAIRS)
def test_id_implementation_matches_symbolic(a: str, b: str) -> None:
    pa = analyze_reading(a).phonemes
    pb = analyze_reading(b).phonemes
    expected = weighted_edit_distance(pa, pb)
    actual = edit_distance_ids(phoneme_ids(pa), phoneme_ids(pb))
    assert actual == pytest.approx(expected)


def test_batch_implementation_matches_symbolic() -> None:
    """バッチ版は、長さの違う候補を混ぜても 1 件ずつの結果と一致する。

    パディングの扱いを間違えると長さの違う候補だけが狂うので、
    最長でない候補が混ざった行列で検証する。
    """
    query = analyze_reading("チクビ").phonemes
    candidates = [analyze_reading(b).phonemes for _, b in _DISTANCE_PAIRS]

    lengths = np.array([len(p) for p in candidates], dtype=np.int64)
    width = int(lengths.max())
    matrix = np.full((len(candidates), width), PAD_ID, dtype=np.int64)
    for row, phonemes in enumerate(candidates):
        matrix[row, : len(phonemes)] = phoneme_ids(phonemes)

    actual = edit_distance_batch(phoneme_ids(query), matrix, lengths)
    expected = [weighted_edit_distance(query, p) for p in candidates]
    assert actual == pytest.approx(expected)


def test_batch_handles_empty_query_and_no_candidates() -> None:
    empty = np.zeros(0, dtype=np.int32)
    phonemes = analyze_reading("チクビ").phonemes
    matrix = phoneme_ids(phonemes).reshape(1, -1)
    lengths = np.array([len(phonemes)], dtype=np.int64)

    # クエリが空なら候補を全削除するコスト。
    assert edit_distance_batch(empty, matrix, lengths) == pytest.approx(
        [weighted_edit_distance((), phonemes)]
    )
    # 候補が無ければ空の配列。
    assert edit_distance_batch(
        phoneme_ids(phonemes), np.zeros((0, 0), dtype=np.int64), np.zeros(0, dtype=np.int64)
    ).shape == (0,)


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

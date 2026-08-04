"""かな -> 音素/モーラ変換のテスト。"""

from __future__ import annotations

import pytest

from jpr.phonology import GEMINATE, LONG, MORAIC_N, analyze_reading, to_katakana


@pytest.mark.parametrize(
    ("reading", "expected"),
    [
        ("チクビ", "ch i k u b i"),
        ("チョコビ", "ch o k o b i"),
        ("チキュウギ", "ch i ky u u g i"),
        # 拗音は口蓋化子音 1 音素として扱う。
        ("キョウト", "ky o u t o"),
        ("シャンプー", "sh a N p u R"),
        # 促音・撥音・長音は独立した特殊モーラ。
        ("ガッコウ", "g a Q k o u"),
        ("ラーメン", "r a R m e N"),
        # 音韻的に子音が交替するもの。
        ("ツイタチ", "ts u i t a ch i"),
        ("フジサン", "f u j i s a N"),
        ("ヒト", "hy i t o"),
        # 外来語の表記。
        ("ヴァイオリン", "v a i o r i N"),
        ("ファイト", "f a i t o"),
        ("ジェット", "j e Q t o"),
    ],
)
def test_phoneme_conversion(reading: str, expected: str) -> None:
    assert analyze_reading(reading).phoneme_string() == expected


def test_hiragana_is_accepted() -> None:
    assert analyze_reading("ちくび").phonemes == analyze_reading("チクビ").phonemes


@pytest.mark.parametrize(
    ("reading", "count"),
    [
        ("チクビ", 3),
        ("チョコビ", 3),  # 拗音は 1 モーラ
        ("ラーメン", 4),  # 長音・撥音はそれぞれ 1 モーラ
        ("ガッコウ", 4),  # 促音も 1 モーラ
        ("キョウト", 3),
    ],
)
def test_mora_count(reading: str, count: int) -> None:
    assert analyze_reading(reading).mora_count == count


def test_special_moras_are_tagged() -> None:
    moras = analyze_reading("ガッコーン").moras
    specials = [m.special for m in moras if m.special]
    assert specials == [GEMINATE, LONG, MORAIC_N]


def test_vowel_skeleton_expands_long_vowel() -> None:
    """長音は直前の母音の伸長として扱う。韻の判定で効く。"""
    assert analyze_reading("ラーメン").vowel_skeleton == ("a", "a", "e", MORAIC_N)


def test_unknown_characters_are_skipped() -> None:
    """解釈できない文字は黙って読み飛ばす。"""
    assert analyze_reading("チクビ?!").phonemes == analyze_reading("チクビ").phonemes


def test_empty_reading() -> None:
    pronunciation = analyze_reading("")
    assert pronunciation.moras == ()
    assert pronunciation.phonemes == ()


def test_to_katakana_leaves_other_characters() -> None:
    assert to_katakana("ちくび乳首ABC") == "チクビ乳首ABC"

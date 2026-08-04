"""2 段検索 (ANN + rerank) のテスト。"""

from __future__ import annotations

import pytest

from jpr.index import Category
from jpr.search import PRESETS, PhoneticSearcher, ScoreWeights


def surfaces(searcher: PhoneticSearcher, query: str, **kwargs) -> list[str]:
    _, results = searcher.search(query, **kwargs)
    return [r.surface for r in results]


def test_search_returns_pronunciation(sample_searcher: PhoneticSearcher) -> None:
    pronunciation, _ = sample_searcher.search("乳首")
    assert pronunciation.reading == "チクビ"
    assert pronunciation.phoneme_string() == "ch i k u b i"


def test_query_itself_is_excluded(sample_searcher: PhoneticSearcher) -> None:
    assert "ラーメン" not in surfaces(sample_searcher, "ラーメン")


def test_results_are_sorted_by_score(sample_searcher: PhoneticSearcher) -> None:
    _, results = sample_searcher.search("乳首", limit=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_limit_is_respected(sample_searcher: PhoneticSearcher) -> None:
    _, results = sample_searcher.search("乳首", limit=3)
    assert len(results) <= 3


def test_person_and_place_excluded_by_default(sample_searcher: PhoneticSearcher) -> None:
    """人名・地名は索引の 7 割を占め音韻的に密集するので既定では引かない。"""
    _, results = sample_searcher.search("田中", limit=20)
    assert all(r.category not in (Category.PERSON, Category.PLACE) for r in results)


def test_categories_can_be_requested(sample_searcher: PhoneticSearcher) -> None:
    _, results = sample_searcher.search(
        "タナカ", limit=20, categories=[Category.PERSON], exclude_same_reading=False
    )
    assert all(r.category is Category.PERSON for r in results)


def test_same_reading_variants_are_deduped(sample_searcher: PhoneticSearcher) -> None:
    """「仕組」「仕組み」「し組み」は 1 件に畳まれる。"""
    _, results = sample_searcher.search("シクミ", limit=20, exclude_same_reading=False)
    readings = [r.reading for r in results]
    assert len(readings) == len(set(readings))


def test_dedupe_keeps_the_most_familiar_spelling(sample_searcher: PhoneticSearcher) -> None:
    """同音異表記のうち、より一般的な表記が残る。"""
    _, results = sample_searcher.search("シクミ", limit=20, exclude_same_reading=False)
    hit = next((r for r in results if r.reading == "シクミ"), None)
    assert hit is not None
    assert hit.surface == "仕組み"


def test_candidate_filter_enables_semantic_constraint(
    sample_searcher: PhoneticSearcher,
) -> None:
    """音韻空間だけでは答えが決まらない問いを、意味の制約で解く。

    「乳首みたいなお菓子」は音韻類似度が最大の語ではなく、音が近くかつ
    お菓子である語が答えになる。
    """
    snacks = {"チョコビ", "チョコボール", "チョコパイ"}
    _, results = sample_searcher.search(
        "乳首", limit=5, candidate_filter=lambda e: e.reading in snacks
    )
    assert results
    assert results[0].surface == "チョコビ"


def test_score_components_are_reported(sample_searcher: PhoneticSearcher) -> None:
    """「なぜ近いのか」を呼び出し側が検証できるよう内訳を返す。"""
    _, results = sample_searcher.search("乳首", limit=1)
    result = results[0]
    assert 0.0 <= result.phonetic_similarity <= 1.0
    assert 0.0 <= result.embedding_similarity <= 1.0
    assert 0.0 <= result.coda_similarity <= 1.0
    assert 0.0 <= result.familiarity <= 1.0


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_all_presets_work(sample_searcher: PhoneticSearcher, preset: str) -> None:
    _, results = sample_searcher.search("乳首", preset=preset, limit=5)
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_unknown_preset_is_rejected(sample_searcher: PhoneticSearcher) -> None:
    with pytest.raises(ValueError, match="未知のプリセット"):
        sample_searcher.search("乳首", preset="nonexistent")


def test_rhyme_preset_prefers_matching_tail_and_vowels(
    sample_searcher: PhoneticSearcher,
) -> None:
    """韻のプリセットでは語尾と母音列の一致が効く。

    「チクビ」に対する「テクビ」(語尾一致) と「シクミ」(母音列 i-u-i 一致) は
    どちらも韻として正当なので、順位ではなく上位に両方入ることを見る。
    """
    _, rhyme = sample_searcher.search("チクビ", preset="rhyme", limit=3)
    top_readings = {r.reading for r in rhyme}
    assert {"テクビ", "シクミ"} <= top_readings


def test_rhyme_preset_weights_tail_over_overall_sound(
    sample_searcher: PhoneticSearcher,
) -> None:
    """語尾が一致する語と、全体の音が近い語のスコア差がプリセットで逆転する。

    「テクビ」は語尾が一致するが語頭が違い、「チクワ」は語頭から一致するが
    語尾が違う。ダジャレでは後者、韻では前者が有利になる。
    """

    def score_of(preset: str, reading: str) -> float:
        _, results = sample_searcher.search("チクビ", preset=preset, limit=20)
        return next(r.score for r in results if r.reading == reading)

    pun_gap = score_of("pun", "チクワ") - score_of("pun", "テクビ")
    rhyme_gap = score_of("rhyme", "チクワ") - score_of("rhyme", "テクビ")
    assert pun_gap > rhyme_gap


def test_custom_weights_override_preset(sample_searcher: PhoneticSearcher) -> None:
    weights = ScoreWeights(
        phoneme=1.0, embedding=0.0, mora=0.0, coda=0.0, vowel=0.0, familiarity=0.0
    )
    _, results = sample_searcher.search("乳首", weights=weights, limit=5)
    # 音韻類似度のみで並ぶので、スコアと音韻類似度が一致する。
    for result in results:
        assert result.score == pytest.approx(result.phonetic_similarity, abs=1e-3)


def test_weights_must_not_sum_to_zero() -> None:
    zero = ScoreWeights(phoneme=0.0, embedding=0.0, mora=0.0, coda=0.0, vowel=0.0, familiarity=0.0)
    with pytest.raises(ValueError, match="重みの合計"):
        zero.normalized()


def test_min_score_filters_results(sample_searcher: PhoneticSearcher) -> None:
    _, results = sample_searcher.search("乳首", min_score=0.9, limit=20)
    assert all(r.score >= 0.9 for r in results)


def test_empty_query_returns_nothing(sample_searcher: PhoneticSearcher) -> None:
    pronunciation, results = sample_searcher.search("")
    assert pronunciation.phonemes == ()
    assert results == []


def test_unpronounceable_query_returns_nothing(sample_searcher: PhoneticSearcher) -> None:
    _, results = sample_searcher.search("!!!")
    assert results == []


# --- compare ---------------------------------------------------------------


def test_compare_reports_readings_and_similarity(sample_searcher: PhoneticSearcher) -> None:
    comparison = sample_searcher.compare("乳首", "チョコビ")
    assert comparison.a_reading == "チクビ"
    assert comparison.b_reading == "チョコビ"
    assert comparison.similarity > 0.75


def test_compare_separates_semantic_and_phonetic_axes(
    sample_searcher: PhoneticSearcher,
) -> None:
    """意味が無関係でも音が近ければ高く、意味が近くても音が遠ければ低い。"""
    phonetic_pair = sample_searcher.compare("乳首", "チョコビ")
    unrelated_sound = sample_searcher.compare("乳首", "電車")
    assert phonetic_pair.similarity > unrelated_sound.similarity


def test_compare_reports_space_breakdown(sample_searcher: PhoneticSearcher) -> None:
    comparison = sample_searcher.compare("乳首", "チョコビ")
    assert set(comparison.spaces) == {"phonetic", "consonant", "vowel", "coda", "rhythm"}
    # 子音は完全一致、母音は異なる。
    assert comparison.spaces["consonant"] > comparison.spaces["vowel"]


def test_compare_identical_words(sample_searcher: PhoneticSearcher) -> None:
    comparison = sample_searcher.compare("ラーメン", "ラーメン")
    assert comparison.similarity == pytest.approx(1.0)
    assert comparison.distance == pytest.approx(0.0)

"""2 段検索 (ANN + rerank) のテスト。"""

from __future__ import annotations

import pytest

from jpr.distance import (
    WORST_SUBSTITUTION_COST,
    similarity_normalizer,
    vowel_skeleton_similarity,
    weighted_edit_distance,
)
from jpr.index import Category
from jpr.phonology import analyze_reading
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


def test_reported_similarity_matches_the_reference_implementation(
    sample_searcher: PhoneticSearcher,
) -> None:
    """検索が返す音韻類似度が、記号ベースの実装から求めた値と一致する。

    rerank は速度のために距離をバッチで解き、分母 (`similarity_normalizer`)
    も配列で写している。片方だけ変えると検索スコアと `compare` の類似度が
    静かに食い違うので、突き合わせをここで担保する。
    """
    query = "乳首"
    query_phonemes = sample_searcher.pronounce(query).phonemes
    _, results = sample_searcher.search(query, limit=10)
    assert results

    for result in results:
        distance = weighted_edit_distance(query_phonemes, result.phonemes)
        denominator = similarity_normalizer(
            len(query_phonemes), len(result.phonemes), WORST_SUBSTITUTION_COST
        )
        expected = max(0.0, 1.0 - distance / denominator)
        assert result.phonetic_similarity == pytest.approx(expected, abs=5e-5), result.surface


def test_reported_vowel_similarity_matches_the_reference_implementation(
    sample_searcher: PhoneticSearcher,
) -> None:
    """検索が返す母音軸が、記号ベースの実装から求めた値と一致する。

    rerank は索引の母音骨格 CSR を Rust の DP に渡す (`_vowel_scores`)。
    参照実装は `distance.vowel_skeleton_similarity`。骨格の作り方 (長音の
    反復・特殊モーラの扱い) が構築側と検索側でずれると、ここが落ちる。
    """
    query = "乳首"
    query_pronunciation = sample_searcher.pronounce(query)
    _, results = sample_searcher.search(query, limit=10)
    assert results

    for result in results:
        expected = vowel_skeleton_similarity(query_pronunciation, analyze_reading(result.reading))
        assert result.vowel_similarity == pytest.approx(expected, abs=5e-5), result.surface


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
    """韻のプリセットでは語尾の一致が最終スコアを大きく動かす。

    「テクビ」は語頭が違うが語尾 2 モーラが完全一致し、「チクワ」は語頭から
    一致するが語尾が違う。プリセットの重みが実際に効いていることを、
    語尾成分の寄与差で確かめる。
    """
    from jpr.search import PRESETS

    def result_for(preset: str, reading: str):
        _, results = sample_searcher.search("チクビ", preset=preset, limit=20)
        return next(r for r in results if r.reading == reading)

    tail_match = result_for("rhyme", "テクビ")
    head_match = result_for("rhyme", "チクワ")

    # 語尾の一致度自体は「テクビ」が上。
    assert tail_match.coda_similarity > head_match.coda_similarity

    # 韻のプリセットは語尾に、ダジャレのプリセットは音韻全体に重みを置く。
    rhyme, pun = PRESETS["rhyme"].normalized(), PRESETS["pun"].normalized()
    assert rhyme.coda > pun.coda
    assert pun.phoneme > rhyme.phoneme

    # その結果、語尾一致の語は韻のプリセットの方が相対的に有利になる。
    rhyme_gap = rhyme.coda * tail_match.coda_similarity - rhyme.coda * head_match.coda_similarity
    pun_gap = pun.coda * tail_match.coda_similarity - pun.coda * head_match.coda_similarity
    assert rhyme_gap > pun_gap


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


# --- モーラ範囲での絞り込み --------------------------------------------------


def test_mora_range_restricts_results(sample_searcher: PhoneticSearcher) -> None:
    """範囲を指定すると、その長さの語だけが返る。"""
    _, results = sample_searcher.search("チョコビ", min_mora=4, max_mora=5, limit=20)
    assert results
    assert all(4 <= r.mora_count <= 5 for r in results)


def test_mora_range_accepts_an_open_end(sample_searcher: PhoneticSearcher) -> None:
    """下限だけ・上限だけの指定も範囲として扱う。"""
    _, lower = sample_searcher.search("チョコビ", min_mora=5, limit=20)
    assert [r.surface for r in lower] == ["チョコボール"]

    _, upper = sample_searcher.search("チョコビ", max_mora=2, limit=20)
    assert [r.surface for r in upper] == ["空"]


def test_mora_range_reaches_past_the_ann_neighbourhood(
    sample_searcher: PhoneticSearcher,
) -> None:
    """範囲指定は音韻空間の近傍に入らない語も候補にする。

    「チョコビ」(3 モーラ) の近傍は同じ長さの語で埋まるので、既定の検索で
    5 モーラの「チョコボール」は上位に来ない。範囲を切ると母集団が
    その長さだけになるので必ず土俵に載る。実索引ではこの差がもっと極端で、
    「筑前煮」は 5 モーラ内に限っても 23823/301551 位に沈む。
    """
    _, ranged = sample_searcher.search("チョコビ", min_mora=5, max_mora=5, limit=20)
    assert [r.surface for r in ranged] == ["チョコボール"]


def test_mora_range_ignores_the_max_gap_guard(sample_searcher: PhoneticSearcher) -> None:
    """範囲を明示したら `_MAX_MORA_GAP` は適用しない。

    ギャップ 3 の安全網は ANN 候補の粗さを補うためのもの。範囲を指定した
    検索では呼び出し側が意図を持っているので、そこに従わないと短いクエリから
    長い語を要求したときに全件落ちてしまう。「ソラ」(2) と
    「チョコボール」(5) はギャップ 3 を超える。
    """
    _, results = sample_searcher.search("ソラ", min_mora=5, max_mora=5, limit=20)
    assert [r.surface for r in results] == ["チョコボール"]


def test_mora_range_still_excludes_person_and_place(
    sample_searcher: PhoneticSearcher,
) -> None:
    """全走査経路でもカテゴリの既定 (人名・地名を除く) は効く。"""
    _, results = sample_searcher.search("チョコビ", min_mora=2, max_mora=12, limit=50)
    assert results
    assert all(r.category not in (Category.PERSON, Category.PLACE) for r in results)


def test_inverted_mora_range_is_rejected(sample_searcher: PhoneticSearcher) -> None:
    with pytest.raises(ValueError, match="モーラ範囲"):
        sample_searcher.search("チョコビ", min_mora=6, max_mora=3)


def test_mora_range_size_counts_the_population(sample_searcher: PhoneticSearcher) -> None:
    """全走査のコストの目安。範囲を指定しなければ索引全体。"""
    assert sample_searcher.mora_range_size(5, 5) == 1
    assert sample_searcher.mora_range_size(None, None) == len(sample_searcher.store)


def test_group_dedupe_matches_per_row_edit_distance(
    sample_searcher: PhoneticSearcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """同音異表記の編集距離の畳み込みが順位もスコアも変えない。

    「仕組み」「仕組」「し組み」は同じ音素列なので距離は代表 1 件の計算で
    済むはずで、配り間違えれば別の行の距離が混ざってここで割れる。
    実辞書の候補数では常に畳む側を通るので、畳まない側を対照にできるのは
    しきい値を動かせる小さな索引だけ。
    """
    from jpr import search as search_module

    def fingerprint() -> list[tuple]:
        _, results = sample_searcher.search(
            "チクビ", min_mora=2, max_mora=12, limit=None, exclude_same_reading=False
        )
        return [(r.surface, r.reading, r.score, r.phonetic_similarity) for r in results]

    monkeypatch.setattr(search_module, "_GROUP_DEDUPE_MIN_CANDIDATES", 1 << 30)
    baseline = fingerprint()

    monkeypatch.setattr(search_module, "_GROUP_DEDUPE_MIN_CANDIDATES", 0)
    assert fingerprint() == baseline


def test_default_search_keeps_the_ann_path(sample_searcher: PhoneticSearcher) -> None:
    """範囲を指定しない検索は従来どおり ANN 経路を通る。

    経路の取り違えを捕まえるために、上位の並びを固定する。1 位が「仕組」でなく
    「仕組み」なのは代表選びの同点解消による (`_representative_rank` 参照)。
    どちらも `familiarity` が 1.0 に飽和して分けられないので、表層の符号順で
    決まる。以前はここに到着順の偶然で「仕組」が入っていた。
    """
    assert surfaces(sample_searcher, "乳首", limit=3) == ["仕組み", "手首", "竹輪"]


# --- 件数の上限 --------------------------------------------------------------


def test_unlimited_limit_returns_more_than_a_capped_one(
    sample_searcher: PhoneticSearcher,
) -> None:
    """`limit=None` は上限なし。"""
    _, capped = sample_searcher.search("チョコビ", limit=3)
    _, everything = sample_searcher.search("チョコビ", limit=None)
    assert len(capped) == 3
    assert len(everything) > len(capped)
    # 同音異表記は畳まれたまま。
    assert len({r.reading for r in everything}) == len(everything)


def test_unlimited_respects_min_score(sample_searcher: PhoneticSearcher) -> None:
    """閾値で母集団を切って全件取る、という使い方が成り立つ。"""
    _, results = sample_searcher.search("チョコビ", limit=None, min_score=0.7)
    assert results
    assert all(r.score >= 0.7 for r in results)


def test_unlimited_combines_with_a_mora_range(sample_searcher: PhoneticSearcher) -> None:
    _, results = sample_searcher.search("チョコビ", limit=None, min_mora=4, max_mora=5)
    assert results
    assert all(4 <= r.mora_count <= 5 for r in results)


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

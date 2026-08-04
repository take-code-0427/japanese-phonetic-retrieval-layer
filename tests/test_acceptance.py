"""実辞書に対する受け入れテスト。

索引が未構築なら自動的にスキップされる (conftest の real_store フィクスチャ)。
小さなサンプル語彙では確認できない「202 万語の中から意図した語が引けるか」を見る。
"""

from __future__ import annotations

import time

import pytest

from jpr.index import Category
from jpr.search import PhoneticSearcher

# なぞなぞ「乳首みたいなお菓子ってなんだ？」の答えを引くための意味的制約。
# LLM が「お菓子である」という知識を与える想定。
SNACK_READINGS = frozenset(
    {
        "チョコビ",
        "チョコボール",
        "チョコパイ",
        "ポッキー",
        "キノコノヤマ",
        "ウマイボウ",
        "ハイチュウ",
        "ガリガリクン",
        "キットカット",
        "アルフォート",
        "ジャガリコ",
        "カラムーチョ",
    }
)


def test_index_is_substantial(real_searcher: PhoneticSearcher) -> None:
    """辞書全体が索引されている。"""
    assert len(real_searcher.store) > 1_000_000


def test_flagship_riddle_with_semantic_constraint(
    real_searcher: PhoneticSearcher,
) -> None:
    """「乳首みたいなお菓子」= チョコビ。

    音韻検索が候補を出し、意味的制約 (お菓子である) が答えを決める。
    このプロジェクトの設計思想そのものを検証する。
    """
    _, results = real_searcher.search(
        "乳首",
        limit=5,
        candidate_filter=lambda entry: entry.reading in SNACK_READINGS,
        candidates=20_000,
    )
    assert results, "お菓子の候補が 1 件も引けていない"
    assert results[0].surface == "チョコビ"


def test_ann_recall_reaches_the_target(real_searcher: PhoneticSearcher) -> None:
    """ANN の候補生成が「チョコビ」に到達できる。

    精密な編集距離では 0.808 で数百位に沈む語だが、埋め込み空間では上位に来る。
    2 段構成が成立する前提条件。
    """
    _, results = real_searcher.search("乳首", limit=400, candidates=2000)
    assert any(r.surface == "チョコビ" for r in results)


def test_concept_example_chikyugi(real_searcher: PhoneticSearcher) -> None:
    """コンセプトのもう 1 つの例。「乳首」と「地球儀」も音韻的に近い。"""
    comparison = real_searcher.compare("乳首", "地球儀")
    assert comparison.similarity > 0.7


def test_search_is_fast(real_searcher: PhoneticSearcher) -> None:
    """202 万語に対して 1 クエリが実用的な速度で返る。

    ANN 導入前の全件走査は 12 秒かかっていた。
    """
    # 初回は mmap のページフォールトを含むので、計測前に 1 回流す。
    real_searcher.search("乳首", limit=10)

    started = time.perf_counter()
    for query in ["ラーメン", "地球儀", "手首", "学校", "電車"]:
        real_searcher.search(query, limit=10)
    elapsed = (time.perf_counter() - started) / 5

    assert elapsed < 1.0, f"1 クエリ {elapsed * 1000:.0f}ms は遅すぎる"


def test_results_are_real_words(real_searcher: PhoneticSearcher) -> None:
    """結果に日本語として提示できない語が混ざらない。

    SudachiDict full には ASCII 見出しの外来語 (`cicli`, `kikugi`) が多数あり、
    これらは読みを持つため音韻的にはヒットしてしまう。
    """
    _, results = real_searcher.search("乳首", limit=20)
    assert results
    for result in results:
        assert any(
            "ぁ" <= ch <= "ヿ" or "一" <= ch <= "鿿" or ch == "ー" for ch in result.surface
        ), f"日本語の文字を含まない表層が返った: {result.surface}"


def test_person_and_place_are_excluded_by_default(
    real_searcher: PhoneticSearcher,
) -> None:
    """既定の検索では人名・地名が結果を埋め尽くさない。

    索引の 7 割が人名・地名で、音韻的に密集しているため。
    """
    _, results = real_searcher.search("乳首", limit=20)
    assert results
    assert all(r.category not in (Category.PERSON, Category.PLACE) for r in results)


def test_person_search_works_when_requested(real_searcher: PhoneticSearcher) -> None:
    """固有名詞の認識用途では人名を明示的に引ける。"""
    _, results = real_searcher.search(
        "タナカ", limit=10, categories=[Category.PERSON], exclude_same_reading=False
    )
    assert results
    assert all(r.category is Category.PERSON for r in results)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # 聞き間違いの補正。1 音素だけ違う語が上位に来る。
        ("科学", "価格"),
        # 語尾が一致する語。
        ("乳首", "手首"),
    ],
)
def test_expected_neighbours_appear(
    real_searcher: PhoneticSearcher, query: str, expected: str
) -> None:
    _, results = real_searcher.search(query, limit=30)
    surfaces = {r.surface for r in results}
    assert expected in surfaces, f"{query} の近傍に {expected} が無い: {sorted(surfaces)}"


def test_mishearing_preset_finds_asr_style_confusions(
    real_searcher: PhoneticSearcher,
) -> None:
    """ASR 補正用途。「地球日」から「地球儀」が引ける。"""
    _, results = real_searcher.search("地球日", preset="mishearing", limit=20)
    assert any(r.surface == "地球儀" for r in results)


def test_rhyme_preset_returns_rhyming_words(real_searcher: PhoneticSearcher) -> None:
    """韻のプリセットでは母音列か語尾が揃った語が返る。"""
    _, results = real_searcher.search("ラーメン", preset="rhyme", limit=10)
    assert results
    # 上位は母音骨格か語尾のどちらかが強く一致している。
    assert any(r.vowel_similarity > 0.9 or r.coda_similarity > 0.9 for r in results[:5])


def test_scores_stay_in_range(real_searcher: PhoneticSearcher) -> None:
    _, results = real_searcher.search("乳首", limit=50)
    for result in results:
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.phonetic_similarity <= 1.0

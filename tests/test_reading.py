"""読み取得のテスト。

Sudachi の辞書ロードが重いので、モジュール全体で 1 インスタンスを共有する。
"""

from __future__ import annotations

import pytest

from jpr.reading import ReadingExtractor


@pytest.fixture(scope="module")
def extractor() -> ReadingExtractor:
    # core 辞書で足りるテストのみを置く。full は索引構築側で使う。
    return ReadingExtractor(dict_type="core")


@pytest.mark.parametrize(
    ("text", "reading"),
    [
        ("乳首", "チクビ"),
        ("地球儀", "チキュウギ"),
        ("学校", "ガッコウ"),
        # 複合語・フレーズも語ごとの読みを連結する。
        ("東京特許許可局", "トウキョウトッキョキョカキョク"),
    ],
)
def test_reading_extraction(extractor: ReadingExtractor, text: str, reading: str) -> None:
    assert extractor.reading_of(text) == reading


def test_kana_input_passes_through(extractor: ReadingExtractor) -> None:
    """かな入力は形態素解析を経ずにカタカナ化する。"""
    assert extractor.reading_of("ちくび") == "チクビ"
    assert extractor.reading_of("チクビ") == "チクビ"


def test_unknown_katakana_word_keeps_its_form(extractor: ReadingExtractor) -> None:
    """辞書に無いカタカナ語は表層のまま読みとして使える。"""
    assert extractor.reading_of("チョコビ") == "チョコビ"


def test_whitespace_is_trimmed(extractor: ReadingExtractor) -> None:
    assert extractor.reading_of("  乳首  ") == "チクビ"


def test_empty_input(extractor: ReadingExtractor) -> None:
    assert extractor.reading_of("") == ""
    assert extractor.reading_of("   ") == ""
    assert extractor.normalize("") == ""


def test_results_are_cached(extractor: ReadingExtractor) -> None:
    """同じ入力は 2 回目以降キャッシュから返る。"""
    extractor.reading_of("乳首")
    before = len(extractor._reading_cache)
    assert extractor.reading_of("乳首") == "チクビ"
    assert len(extractor._reading_cache) == before


def test_cache_is_bounded(extractor: ReadingExtractor) -> None:
    """キャッシュが無制限に増えない。"""
    from jpr.reading import _CACHE_LIMIT

    for index in range(_CACHE_LIMIT + 10):
        extractor.reading_of(f"ア{index}")
    assert len(extractor._reading_cache) <= _CACHE_LIMIT


def test_normalize_absorbs_spelling_variants(extractor: ReadingExtractor) -> None:
    """表層の揺れを吸収する。クエリ自身を結果から除くのに使う。"""
    assert extractor.normalize("ふとん") == extractor.normalize("布団")


def test_concurrent_access_does_not_break_the_tokenizer() -> None:
    """複数スレッドから同時に呼んでも壊れない。

    Sudachi の Tokenizer は解析用バッファを内部に持つので、ロックを外すと
    Rust 側が `RuntimeError: Already borrowed` を投げる。web.py の非 async な
    エンドポイントは starlette のスレッドプールで動くため、同時リクエストが
    そのまま衝突する — 実際に本番の /api/similar が 500 を返した。

    キャッシュが埋まると tokenize に入らず衝突しないので、毎回キャッシュを
    捨てて解析させる。
    """
    import threading

    local = ReadingExtractor(dict_type="core")
    words = ["科学", "乳首", "布団", "手首", "価格", "東京特許許可局"]
    errors: list[BaseException] = []

    def hammer(offset: int) -> None:
        try:
            for i in range(60):
                word = words[(i + offset) % len(words)]
                local._reading_cache.clear()
                local._normalized_cache.clear()
                assert local.reading_of(word)
                assert local.normalize(word)
        except BaseException as error:
            # 何が飛ぶかを見るのが目的なので広く捕まえる。
            errors.append(error)

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"{type(errors[0]).__name__}: {errors[0]}"

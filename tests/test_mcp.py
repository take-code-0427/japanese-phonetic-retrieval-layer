"""MCP サーバのテスト。"""

from __future__ import annotations

import itertools
import json

import pytest

from jpr.mcp import create_server
from jpr.store import PhoneticStore


@pytest.fixture
def server(sample_store: PhoneticStore):
    return create_server(sample_store.path)


async def call(server, name: str, arguments: dict) -> dict:
    """tool を呼んで、返ってきたテキストを JSON としてパースする。"""
    result = await server.call_tool(name, arguments)
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_tools_are_registered(server) -> None:
    names = {tool.name for tool in await server.list_tools()}
    assert names == {
        "search_phonetically",
        "compose_phrase",
        "compare_phonetically",
        "pronounce",
    }


@pytest.mark.asyncio
async def test_search_returns_reading_and_results(server) -> None:
    payload = await call(server, "search_phonetically", {"query": "乳首", "limit": 5})
    assert payload["reading"] == "チクビ"
    assert payload["phonemes"] == ["ch", "i", "k", "u", "b", "i"]
    assert payload["mora_count"] == 3
    assert len(payload["results"]) <= 5
    for result in payload["results"]:
        assert set(result) >= {"word", "reading", "score", "phonetic_similarity"}


@pytest.mark.asyncio
async def test_search_warns_that_score_ignores_meaning(server) -> None:
    """LLM が音韻スコアを意味的な正しさと混同しないよう注意書きを返す。"""
    payload = await call(server, "search_phonetically", {"query": "乳首"})
    assert "意味" in payload["note"]


@pytest.mark.asyncio
async def test_search_rejects_unknown_preset(server) -> None:
    payload = await call(server, "search_phonetically", {"query": "乳首", "preset": "nonexistent"})
    assert "error" in payload
    assert "pun" in payload["available"]


@pytest.mark.asyncio
async def test_search_rejects_unknown_category(server) -> None:
    payload = await call(
        server, "search_phonetically", {"query": "乳首", "categories": ["nonexistent"]}
    )
    assert "error" in payload
    assert "common" in payload["available"]


@pytest.mark.asyncio
async def test_search_accepts_categories(server) -> None:
    payload = await call(
        server,
        "search_phonetically",
        {"query": "乳首", "categories": ["product"], "limit": 5},
    )
    assert all(r["category"] == "product" for r in payload["results"])


@pytest.mark.asyncio
async def test_search_accepts_a_mora_range(server) -> None:
    payload = await call(
        server,
        "search_phonetically",
        {"query": "チョコビ", "min_mora": 4, "max_mora": 5, "limit": 20},
    )
    assert payload["results"]
    assert all(4 <= r["mora_count"] <= 5 for r in payload["results"])
    # 全走査したことを LLM が報告できるよう、母集団の規模も返す。
    assert payload["scanned"] > 0
    assert payload["min_mora"] == 4


@pytest.mark.asyncio
async def test_search_rejects_an_inverted_mora_range(server) -> None:
    payload = await call(
        server, "search_phonetically", {"query": "チョコビ", "min_mora": 6, "max_mora": 3}
    )
    assert "モーラ範囲" in payload["error"]


@pytest.mark.asyncio
async def test_search_caps_limit_for_the_context_window(server) -> None:
    """MCP は無制限を露出しない。LLM のコンテキストを溢れさせないため。"""
    payload = await call(server, "search_phonetically", {"query": "チョコビ", "limit": 100_000})
    assert len(payload["results"]) <= 200
    # 0 や負の件数を渡されても 1 件以上は返す (0 件を無制限と誤解させない)。
    zero = await call(server, "search_phonetically", {"query": "チョコビ", "limit": 0})
    assert zero["results"]


@pytest.mark.asyncio
async def test_compose_splits_the_input_into_segments(server) -> None:
    """長い入力が複数の区間に分かれ、どこが何になったかが返る。"""
    payload = await call(server, "compose_phrase", {"text": "チョコビラーメン", "limit": 5})
    assert payload["results"]
    best = payload["results"][0]
    # 8 モーラの入力は sample_store の語 1 つでは覆えないので必ず分割される。
    assert best["segment_count"] >= 2
    # 区間の対応が読めないと空耳として検証できない。
    for segment in best["segments"]:
        assert set(segment) >= {"surface", "reading", "source_reading", "start", "end"}
        assert segment["start"] < segment["end"]
    # 区間は入力を隙間なく覆う。
    covered = [(s["start"], s["end"]) for s in best["segments"]]
    assert covered[0][0] == 0
    assert covered[-1][1] == payload["mora_count"]
    for left, right in itertools.pairwise(covered):
        assert left[1] == right[0]


@pytest.mark.asyncio
async def test_compose_warns_that_score_ignores_meaning(server) -> None:
    payload = await call(server, "compose_phrase", {"text": "チョコビラーメン"})
    assert "意味" in payload["note"]


@pytest.mark.asyncio
async def test_compare_separates_axes(server) -> None:
    payload = await call(server, "compare_phonetically", {"a": "乳首", "b": "チョコビ"})
    assert payload["a"]["reading"] == "チクビ"
    assert payload["b"]["reading"] == "チョコビ"
    assert payload["similarity"] > 0.75
    # 空間別の内訳で「どこが似ているか」を示す。
    assert payload["spaces"]["consonant"] > payload["spaces"]["vowel"]


@pytest.mark.asyncio
async def test_pronounce_returns_mora_structure(server) -> None:
    payload = await call(server, "pronounce", {"text": "学校"})
    assert payload["reading"] == "ガッコウ"
    assert payload["moras"] == ["ガ", "ッ", "コ", "ウ"]
    assert payload["mora_count"] == 4
    # 音素列は内部表記 (ヘボン式寄りの ASCII)、ipa は同じものの IPA。促音が
    # 後続子音の重複になっているのが LLM に渡る表記として正しい形。
    assert payload["phonemes"] == ["g", "a", "Q", "k", "o", "u"]
    assert payload["ipa"] == "ɡakkoɯ"

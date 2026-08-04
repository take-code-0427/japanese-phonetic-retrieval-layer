"""MCP サーバのテスト。"""

from __future__ import annotations

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
    assert names == {"search_phonetically", "compare_phonetically", "pronounce"}


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
    payload = await call(
        server, "search_phonetically", {"query": "乳首", "preset": "nonexistent"}
    )
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
async def test_compare_separates_axes(server) -> None:
    payload = await call(server, "compare_phonetically", {"a": "乳首", "b": "チョコビ"})
    assert payload["a"]["reading"] == "チクビ"
    assert payload["b"]["reading"] == "チョコビ"
    assert payload["phonetic_similarity"] > 0.75
    # 空間別の内訳で「どこが似ているか」を示す。
    assert payload["spaces"]["consonant"] > payload["spaces"]["vowel"]


@pytest.mark.asyncio
async def test_pronounce_returns_mora_structure(server) -> None:
    payload = await call(server, "pronounce", {"text": "学校"})
    assert payload["reading"] == "ガッコウ"
    assert payload["moras"] == ["ガ", "ッ", "コ", "ウ"]
    assert payload["mora_count"] == 4

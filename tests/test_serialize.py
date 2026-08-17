"""結果の形が窓をまたいで一致することの検証 (`jpr.serialize`)。

CLI (`--json`) / MCP / Web は同じ `SearchResult` / `PhraseCandidate` を
返すのに、以前はそれぞれの出口が辞書リテラルを書いていた。結果として MCP
だけ `ipa` と `pos` が落ち、区間の位置が `mora_range: [start, end]` と
`start` / `end` に分かれ、比較のキーが `phonetic_similarity` と
`similarity` に割れていた。**どれもテストが窓を別々に見ていたので
気付かれなかった。**

ここは窓どうしを突き合わせる唯一の場所なので、片方だけにフィールドを
足すと落ちる。窓ごとに違ってよいのは「1 回に何件返すか」や、その窓にしか
意味のない項目 (CLI の `elapsed_ms` など) のような呼び出しの制約であって、
1 件の表し方ではない。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from jpr.cli import main
from jpr.mcp import create_server
from jpr.store import PhoneticStore
from jpr.web import create_app


@pytest.fixture
def client(sample_store: PhoneticStore) -> TestClient:
    return TestClient(create_app(sample_store.path))


@pytest.fixture
def server(sample_store: PhoneticStore):
    return create_server(sample_store.path)


async def call(server, name: str, arguments: dict) -> dict:
    result = await server.call_tool(name, arguments)
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


def run_cli(capsys, *argv: str, index: PhoneticStore | None = None) -> dict:
    """`jpr ... --json` を呼んで標準出力を JSON としてパースする。

    `index` は索引を引くサブコマンドにだけ渡す。`compare` は Sudachi 辞書
    だけで動くので `--index` を取らない。
    """
    extra = ["--index", str(index.path)] if index is not None else []
    assert main([*argv, "--json", *extra]) == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.asyncio
async def test_search_results_have_the_same_shape_in_every_window(
    server, client, capsys, sample_store
) -> None:
    """検索結果 1 件のキーが CLI / MCP / Web で完全に一致する。"""
    from_mcp = await call(server, "search_phonetically", {"query": "乳首", "limit": 5})
    from_web = client.get("/api/similar", params={"q": "乳首", "limit": 5}).json()
    from_cli = run_cli(capsys, "similar", "乳首", "--limit", "5", index=sample_store)
    shapes = [set(p["results"][0]) for p in (from_mcp, from_web, from_cli) if p["results"]]
    assert len(shapes) == 3
    assert shapes[0] == shapes[1] == shapes[2]


@pytest.mark.asyncio
async def test_search_results_carry_the_full_score_breakdown(server) -> None:
    """内訳を窓で削らない。LLM も画面と同じ根拠で順位を検算できるべき。

    `embedding_similarity` / `vowel_similarity` / `pos` / `ipa` / `phonemes`
    は以前 MCP 側だけ落ちていた。
    """
    payload = await call(server, "search_phonetically", {"query": "乳首", "limit": 5})
    assert set(payload["results"][0]) >= {
        "word",
        "reading",
        "score",
        "phonetic_similarity",
        "embedding_similarity",
        "coda_similarity",
        "vowel_similarity",
        "containment",
        "mora_count",
        "category",
        "pos",
        "familiarity",
        "phonemes",
        "ipa",
    }


@pytest.mark.asyncio
async def test_phrase_segments_have_the_same_shape_in_every_window(
    server, client, capsys, sample_store
) -> None:
    """区間 1 つのキーが CLI / MCP / Web で一致する。

    MCP は位置を `mora_range: [start, end]` で返していた。同じ値を 2 通りに
    書き分ける理由がないので `start` / `end` に揃えてある。
    """
    from_mcp = await call(server, "compose_phrase", {"text": "チョコビラーメン", "limit": 5})
    from_web = client.get("/api/phrase", params={"text": "チョコビラーメン", "limit": 5}).json()
    from_cli = run_cli(capsys, "phrase", "チョコビラーメン", "--limit", "5", index=sample_store)
    payloads = [from_mcp, from_web, from_cli]
    assert all(p["results"] for p in payloads)
    candidates = [set(p["results"][0]) for p in payloads]
    assert candidates[0] == candidates[1] == candidates[2]
    segments = [set(p["results"][0]["segments"][0]) for p in payloads]
    assert segments[0] == segments[1] == segments[2]


@pytest.mark.asyncio
async def test_comparison_has_the_same_shape_in_every_window(server, client, capsys) -> None:
    """比較の応答が窓で一致する。キーは `similarity` / `distance`。

    MCP は `phonetic_similarity` / `phonetic_distance` という別名を使って
    いた。同じ値なので CLI / Web 側の名前に寄せてある。
    """
    from_mcp = await call(server, "compare_phonetically", {"a": "乳首", "b": "チョコビ"})
    from_web = client.get("/api/compare", params={"a": "乳首", "b": "チョコビ"}).json()
    from_cli = run_cli(capsys, "compare", "乳首", "チョコビ")
    assert set(from_mcp) == set(from_web) == set(from_cli)
    assert from_mcp["similarity"] == from_web["similarity"] == from_cli["similarity"]


@pytest.mark.asyncio
async def test_pronunciation_block_is_shared_by_every_endpoint(server, client) -> None:
    """クエリ側の音韻表現は全応答に同じ形で載る。"""
    common = {"reading", "phonemes", "ipa", "mora_count", "moras"}
    payloads = [
        await call(server, "search_phonetically", {"query": "乳首"}),
        await call(server, "compose_phrase", {"text": "チョコビラーメン"}),
        await call(server, "pronounce", {"text": "乳首"}),
        client.get("/api/similar", params={"q": "乳首"}).json(),
        client.get("/api/phrase", params={"text": "チョコビラーメン"}).json(),
        client.get("/api/phrase/lattice", params={"text": "チョコビラーメン"}).json(),
        client.get("/api/pronounce", params={"text": "乳首"}).json(),
    ]
    for payload in payloads:
        assert set(payload) >= common

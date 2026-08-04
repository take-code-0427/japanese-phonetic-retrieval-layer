"""Web API のテスト。

索引を必要とする経路は `sample_store` の小さな語彙で確認する。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from jpr.store import PhoneticStore
from jpr.web import create_app


@pytest.fixture
def client(sample_store: PhoneticStore) -> TestClient:
    return TestClient(create_app(sample_store.path))


def test_index_page_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "音韻検索" in response.text


def test_static_assets_are_served(client: TestClient) -> None:
    """フロントは CSS と JS を静的に引く。パッケージに同梱されていることの確認。"""
    for path in ("/static/app.css", "/static/app.js"):
        assert client.get(path).status_code == 200, path


def test_similar_returns_reading_and_results(client: TestClient) -> None:
    payload = client.get("/api/similar", params={"q": "乳首", "limit": 5}).json()
    assert payload["reading"] == "チクビ"
    assert payload["phonemes"] == ["ch", "i", "k", "u", "b", "i"]
    assert payload["mora_count"] == 3
    assert len(payload["results"]) <= 5
    for result in payload["results"]:
        assert set(result) >= {"word", "reading", "score", "phonemes", "category"}


def test_similar_filters_by_category(client: TestClient) -> None:
    payload = client.get(
        "/api/similar", params={"q": "田中", "categories": "person", "limit": 10}
    ).json()
    assert {r["category"] for r in payload["results"]} <= {"person"}


def test_similar_rejects_unknown_preset(client: TestClient) -> None:
    response = client.get("/api/similar", params={"q": "乳首", "preset": "bogus"})
    assert response.status_code == 400
    assert "bogus" in response.json()["detail"]


def test_similar_rejects_unknown_category(client: TestClient) -> None:
    response = client.get("/api/similar", params={"q": "乳首", "categories": "nope"})
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_similar_requires_query(client: TestClient) -> None:
    assert client.get("/api/similar").status_code == 422
    assert client.get("/api/similar", params={"q": ""}).status_code == 422


def test_similar_caps_limit(client: TestClient) -> None:
    """件数の上限を超える要求は弾く。件数に比例して復号とシリアライズが増えるため。"""
    assert client.get("/api/similar", params={"q": "乳首", "limit": 10_000}).status_code == 422
    # 負の件数は依然として弾く (0 だけを無制限の合図にする)。
    assert client.get("/api/similar", params={"q": "乳首", "limit": -1}).status_code == 422


def test_similar_limit_zero_is_unlimited(client: TestClient) -> None:
    """limit=0 は上限なし。切り詰めていないことを truncated で確かめる。"""
    payload = client.get("/api/similar", params={"q": "チョコビ", "limit": 0}).json()
    assert payload["results"]
    assert payload["truncated"] is False
    assert payload["total"] == len(payload["results"])


def test_similar_accepts_a_mora_range(client: TestClient) -> None:
    payload = client.get(
        "/api/similar", params={"q": "チョコビ", "min_mora": 4, "max_mora": 5, "limit": 20}
    ).json()
    assert payload["results"]
    assert all(4 <= r["mora_count"] <= 5 for r in payload["results"])
    # 全走査したことと、その母集団の規模を返す。
    assert payload["scanned"] > 0


def test_similar_reports_no_scan_without_a_mora_range(client: TestClient) -> None:
    """範囲を指定しなければ ANN 経路なので、走査件数は報告しない。"""
    payload = client.get("/api/similar", params={"q": "チョコビ"}).json()
    assert payload["scanned"] is None


def test_similar_rejects_an_inverted_mora_range(client: TestClient) -> None:
    response = client.get("/api/similar", params={"q": "チョコビ", "min_mora": 6, "max_mora": 3})
    assert response.status_code == 400
    assert "モーラ範囲" in response.json()["detail"]


def test_pronounce_returns_mora_structure(client: TestClient) -> None:
    payload = client.get("/api/pronounce", params={"text": "ラーメン"}).json()
    assert payload["reading"] == "ラーメン"
    assert payload["moras"] == ["ラ", "ー", "メ", "ン"]
    assert payload["mora_count"] == 4
    # 長音は直前の母音を引き継ぐ。
    assert payload["vowel_skeleton"] == ["a", "a", "e", "N"]


def test_compare_returns_space_breakdown(client: TestClient) -> None:
    payload = client.get("/api/compare", params={"a": "科学", "b": "価格"}).json()
    assert payload["a"]["reading"] == "カガク"
    assert payload["b"]["reading"] == "カカク"
    assert 0.0 <= payload["similarity"] <= 1.0
    assert set(payload["spaces"]) >= {"consonant", "vowel", "coda", "rhythm"}


def test_info_reports_index_metadata(client: TestClient, sample_store: PhoneticStore) -> None:
    payload = client.get("/api/info").json()
    assert payload["count"] == len(sample_store)
    assert payload["format_version"] == sample_store.meta.version
    assert {space["name"] for space in payload["spaces"]} == set(sample_store.meta.dims)
    # ANN を張る空間は候補生成に使うものだけ。
    roles = {space["name"]: space["role"] for space in payload["spaces"]}
    assert roles["phonetic"] == "ANN + rerank"
    assert roles["vowel"] == "rerank のみ"
    assert payload["presets"] == ["mishearing", "pun", "rhyme"]


def test_phonemes_exposes_features(client: TestClient) -> None:
    """フロントは音素チップの色をこの素性表から決めるので、素性が揃っている必要がある。"""
    payload = client.get("/api/phonemes").json()
    assert payload["consonants"]["k"] == {
        "place": "velar",
        "manner": "stop",
        "voiced": False,
        "palatalized": False,
    }
    assert payload["vowels"]["i"] == {"height": 0, "backness": 0, "rounded": False}
    assert set(payload["special"]) == {"R", "Q", "N"}


def test_align_pairs_sum_to_edit_distance(client: TestClient) -> None:
    """対ごとの距離の総和が編集距離に一致する。表示が距離の内訳であることの担保。"""
    from jpr.distance import weighted_edit_distance

    a = ("ch", "i", "k", "u", "b", "i")
    b = ("t", "e", "k", "u", "b", "i")
    payload = client.get("/api/align", params={"a": " ".join(a), "b": " ".join(b)}).json()

    assert [(p["a"], p["b"], p["op"]) for p in payload["pairs"]] == [
        ("ch", "t", "sub"),
        ("i", "e", "sub"),
        ("k", "k", "match"),
        ("u", "u", "match"),
        ("b", "b", "match"),
        ("i", "i", "match"),
    ]
    assert payload["total"] == pytest.approx(weighted_edit_distance(a, b), abs=1e-3)


def test_align_marks_insertions_and_deletions(client: TestClient) -> None:
    payload = client.get("/api/align", params={"a": "s a k a", "b": "a k a"}).json()
    ops = [p["op"] for p in payload["pairs"]]
    assert "del" in ops
    # 削除された側は候補側が空になる。
    deleted = next(p for p in payload["pairs"] if p["op"] == "del")
    assert deleted["a"] == "s"
    assert deleted["b"] is None


def test_missing_index_reports_how_to_build(tmp_path) -> None:
    """索引が無くてもサーバは起動でき、原因と対処を 503 で返す。"""
    client = TestClient(create_app(tmp_path / "absent"), raise_server_exceptions=False)
    response = client.get("/api/info")
    assert response.status_code == 503
    assert "build-index" in response.json()["detail"]

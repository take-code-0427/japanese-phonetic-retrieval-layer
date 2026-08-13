"""Web API のテスト。

索引を必要とする経路は `sample_store` の小さな語彙で確認する。
"""

from __future__ import annotations

from unittest import mock

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
    assert payload["ipa"] == "t͡ɕikɯbi"
    assert len(payload["results"]) <= 5
    for result in payload["results"]:
        assert set(result) >= {"word", "reading", "score", "phonemes", "ipa", "category"}


def test_similar_filters_by_category(client: TestClient) -> None:
    payload = client.get(
        "/api/similar", params={"q": "田中", "categories": "person", "limit": 10}
    ).json()
    assert {r["category"] for r in payload["results"]} <= {"person"}


def test_phrase_splits_the_input_into_segments(client: TestClient) -> None:
    """長い入力が区間に分かれ、入力側のモーラ列と対応が返る。"""
    payload = client.get("/api/phrase", params={"text": "チョコビラーメン", "limit": 5}).json()
    # チョ は拗音で 1 モーラ。ー と ン も 1 モーラずつ数える。
    assert payload["moras"] == ["チョ", "コ", "ビ", "ラ", "ー", "メ", "ン"]
    assert payload["mora_count"] == 7
    assert payload["results"]
    best = payload["results"][0]
    assert best["segment_count"] >= 2
    assert best["text"] == "".join(s["surface"] for s in best["segments"])
    assert best["reading"] == "".join(s["reading"] for s in best["segments"])
    # 区間は入力を隙間なく覆う。
    assert best["segments"][0]["start"] == 0
    assert best["segments"][-1]["end"] == payload["mora_count"]


def test_phrase_scores_stay_within_the_unit_range(client: TestClient) -> None:
    """スコアは 0〜1。通常検索と同じ尺度で読めなければ並べて比較できない。"""
    payload = client.get("/api/phrase", params={"text": "チョコビラーメン"}).json()
    for candidate in payload["results"]:
        assert 0.0 <= candidate["score"] <= 1.0
        assert 0.0 <= candidate["phonetic_similarity"] <= 1.0
        for segment in candidate["segments"]:
            assert 0.0 <= segment["similarity"] <= 1.0


def test_phrase_can_run_without_particles(client: TestClient) -> None:
    """助詞を切っても経路自体は成立する (1 モーラの区間が埋まらなくなるだけ)。"""
    payload = client.get(
        "/api/phrase", params={"text": "チョコビラーメン", "allow_particles": "false"}
    ).json()
    for candidate in payload["results"]:
        assert not any(s["is_particle"] for s in candidate["segments"])


def test_phrase_lattice_folds_nodes(client: TestClient) -> None:
    """ラティスが同じ語を 1 ノードに畳み、経路と辺を返す。"""
    payload = client.get(
        "/api/phrase/lattice", params={"text": "チョコビラーメン", "node_budget": 30}
    ).json()
    assert payload["nodes"]
    assert payload["edges"]
    assert payload["paths"]
    # ノード id は一意。
    ids = [n["id"] for n in payload["nodes"]]
    assert len(ids) == len(set(ids))
    # 経路が参照するノードは全部図にある。
    known = set(ids)
    for path in payload["paths"]:
        assert set(path["nodes"]) <= known
    # 辺の端点も図にある。
    for edge in payload["edges"]:
        for endpoint in (edge["source"], edge["target"]):
            assert endpoint is None or endpoint in known


def test_phrase_lattice_respects_node_budget(client: TestClient) -> None:
    """予算に収まる。画面が埋まらないことの保証。"""
    payload = client.get(
        "/api/phrase/lattice", params={"text": "チョコビラーメン", "node_budget": 12}
    ).json()
    if payload["path_count"] > 1:
        assert payload["node_count"] <= 12


def test_phrase_lattice_edges_are_contiguous(client: TestClient) -> None:
    """辺で結ばれたノードはモーラ位置として連続している。"""
    payload = client.get("/api/phrase/lattice", params={"text": "チョコビラーメン"}).json()
    by_id = {n["id"]: n for n in payload["nodes"]}
    for edge in payload["edges"]:
        if edge["source"] is None or edge["target"] is None:
            continue
        assert by_id[edge["source"]]["end"] == by_id[edge["target"]]["start"]


def test_phrase_lattice_rejects_a_bad_budget(client: TestClient) -> None:
    """予算の範囲外は弾く。届くまでビームを広げるので、大きすぎると探索が伸びる。"""
    assert (
        client.get(
            "/api/phrase/lattice", params={"text": "チョコビ", "node_budget": 100_000}
        ).status_code
        == 422
    )
    assert (
        client.get("/api/phrase/lattice", params={"text": "チョコビ", "node_budget": 1}).status_code
        == 422
    )


def test_info_exposes_phrase_defaults(client: TestClient) -> None:
    """フロントが既定値を固定表で持たないよう、サーバ側の値を配る。"""
    phrase = client.get("/api/info").json()["phrase"]
    assert set(phrase) >= {"max_chunk_moras", "chunk_candidates", "beam_width", "min_chunk_score"}


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


def test_similar_has_no_result_cap(client: TestClient) -> None:
    """件数に上限を持たない。母集団を切るのは min_score の役目。"""
    assert client.get("/api/similar", params={"q": "乳首", "limit": 10_000}).status_code == 200
    # 負の件数は依然として弾く (0 だけを無制限の合図にする)。
    assert client.get("/api/similar", params={"q": "乳首", "limit": -1}).status_code == 422


def test_similar_returns_everything_above_the_score_floor(client: TestClient) -> None:
    """既定は「下限を満たす語を全部」。件数で切らないので limit の指定も要らない。"""
    payload = client.get("/api/similar", params={"q": "チョコビ", "min_score": 0.0}).json()
    # 下限 0 は候補全件。件数で切らないので、limit を明示した側より多く返る。
    capped = client.get("/api/similar", params={"q": "チョコビ", "min_score": 0.0, "limit": 3})
    assert len(capped.json()["results"]) == 3
    assert len(payload["results"]) > 3
    # 下限を上げれば減る。母集団を決めているのが min_score だということ。
    strict = client.get("/api/similar", params={"q": "チョコビ", "min_score": 0.7}).json()
    assert strict["below_floor"] is False
    assert 0 < len(strict["results"]) < len(payload["results"])
    assert all(r["score"] >= 0.7 for r in strict["results"])


def test_similar_falls_back_when_nothing_meets_the_floor(client: TestClient) -> None:
    """下限を満たす語が無ければ下限を外して近い順を返す。

    スコアはクエリの長さに依存するので、下限に最近傍すら届かないクエリが実在する
    (実測で「わたしのなまえ」7 拍の最高が 0.791)。件数で切らない以上そのまま
    空の画面になるため、外したことを below_floor で伝えて近い順を見せる。
    """
    payload = client.get("/api/similar", params={"q": "チョコビ", "min_score": 1.0}).json()
    assert payload["below_floor"] is True
    assert payload["results"]
    # 外した結果なので、下限を下回るスコアが並ぶ。
    assert payload["results"][0]["score"] < 1.0


def test_similar_does_not_flag_a_fallback_when_the_floor_is_met(client: TestClient) -> None:
    payload = client.get("/api/similar", params={"q": "チョコビ", "min_score": 0.0}).json()
    assert payload["below_floor"] is False


def test_similar_defaults_to_the_advertised_score_floor(client: TestClient) -> None:
    """既定の下限は /api/info が配る値と一致する。フロントはこれを写して初期値にする。"""
    floor = client.get("/api/info").json()["default_min_score"]
    payload = client.get("/api/similar", params={"q": "チョコビ"}).json()
    # 下限を明示した結果と一致する。sample_store の 18 語では既定の下限に
    # 届かず below_floor に落ちるので、そちらの経路まで含めて同じであることを見る。
    explicit = client.get("/api/similar", params={"q": "チョコビ", "min_score": floor}).json()
    assert payload["below_floor"] == explicit["below_floor"]
    assert [r["word"] for r in payload["results"]] == [r["word"] for r in explicit["results"]]
    if not payload["below_floor"]:
        assert all(r["score"] >= floor for r in payload["results"])


def test_similar_accepts_a_mora_range(client: TestClient) -> None:
    payload = client.get(
        # 見るのは範囲の絞り込みなので、既定のスコア下限は外す (sample_store の
        # 18 語では 0.8 を超える語が無く、範囲の検証にならない)。
        "/api/similar",
        params={"q": "チョコビ", "min_mora": 4, "max_mora": 5, "min_score": 0.0},
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
    # 長音は IPA では長さ記号。撥音は語末の実現 [ɴ]。
    assert payload["ipa"] == "ɾaːmeɴ"


def test_compare_returns_space_breakdown(client: TestClient) -> None:
    payload = client.get("/api/compare", params={"a": "科学", "b": "価格"}).json()
    assert payload["a"]["reading"] == "カガク"
    assert payload["b"]["reading"] == "カカク"
    # IPA の [ɡ] は ASCII の g と別のコードポイント。有声性の差がここに出る。
    assert (payload["a"]["ipa"], payload["b"]["ipa"]) == ("kaɡakɯ", "kakakɯ")
    assert 0.0 <= payload["similarity"] <= 1.0
    assert set(payload["spaces"]) >= {"consonant", "vowel", "coda", "rhythm"}


def test_info_reports_index_metadata(client: TestClient, sample_store: PhoneticStore) -> None:
    payload = client.get("/api/info").json()
    assert payload["count"] == len(sample_store)
    assert payload["format_version"] == sample_store.meta.version
    assert {space["name"] for space in payload["spaces"]} == set(sample_store.meta.dims)
    # 候補生成に使うのは 1 空間だけ。他は rerank でスコアを足すのに使う。
    roles = {space["name"]: space["role"] for space in payload["spaces"]}
    assert roles["phonetic"] == "候補生成 + rerank"
    assert roles["vowel"] == "rerank のみ"
    assert payload["presets"] == ["mishearing", "pun", "rhyme"]


def test_phonemes_exposes_features(client: TestClient) -> None:
    """フロントは音素チップの色と IPA をこの表から決めるので、揃っている必要がある。"""
    payload = client.get("/api/phonemes").json()
    assert payload["consonants"]["k"] == {
        "place": "velar",
        "manner": "stop",
        "voiced": False,
        "palatalized": False,
        "ipa": "k",
    }
    assert payload["vowels"]["i"] == {
        "height": 0,
        "backness": 0,
        "rounded": False,
        "ipa": "i",
    }
    assert set(payload["special"]) == {"R", "Q", "N"}
    assert payload["special"]["N"] == {"label": "撥音", "ipa": "ɴ"}


def test_phonemes_ipa_covers_every_symbol(client: TestClient) -> None:
    """IPA が欠けた音素があるとチップの併記だけが黙って空になるので、全件を要求する。"""
    payload = client.get("/api/phonemes").json()
    for group in ("consonants", "vowels"):
        missing = [symbol for symbol, f in payload[group].items() if not f["ipa"]]
        assert not missing, f"{group} に IPA が無い: {missing}"


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


def test_concurrent_requests_do_not_return_errors(client: TestClient) -> None:
    """同時リクエストで 500 を返さない。

    非 async なエンドポイントは starlette のスレッドプールで動くので、
    リクエストごとに別スレッドから searcher へ入る。Sudachi の Tokenizer は
    スレッドセーフではないため、ロックを外すと `Already borrowed` で 500 に
    なる (本番の /api/similar で実際に起きた)。
    """
    import concurrent.futures

    paths = [
        ("/api/similar", {"q": "科学", "limit": 5}),
        ("/api/similar", {"q": "乳首", "limit": 5, "preset": "rhyme"}),
        ("/api/phrase", {"text": "ちくび", "limit": 3}),
        ("/api/pronounce", {"text": "布団"}),
        ("/api/compare", {"a": "布団", "b": "ストン"}),
    ]

    def fetch(index: int):
        path, params = paths[index % len(paths)]
        return client.get(path, params=params)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(fetch, range(40)))

    failed = [r for r in responses if r.status_code != 200]
    assert not failed, f"{len(failed)} 件が失敗: {failed[0].status_code} {failed[0].text[:200]}"


def test_mora_scan_concurrency_is_capped(sample_store: PhoneticStore) -> None:
    """モーラ範囲の全走査は同時実行数が制限される。

    全走査は母集団ぶんの内積を一時配列に起こすので、並行させるとメモリが
    溢れる (実測で 1 本 +279MB / 10 本 +348MB、2GB の本番が OOM killed に
    なった)。門が実際に閉じることを、同時に入った本数の最大値で見る。
    """
    import concurrent.futures
    import contextlib
    import threading

    from jpr import web as web_module

    app = create_app(sample_store.path)
    client = TestClient(app)

    workers = 8
    inside = 0
    peak = 0
    guard = threading.Lock()
    original = web_module.PhoneticSearcher.search
    # 全員が門をくぐろうとするまで中で待たせる。門が開いていれば workers 本が
    # 揃って peak == workers になり、閉じていれば揃わずタイムアウトで抜ける。
    # sleep で代用すると、小さな語彙では走査が一瞬で終わって同時に入らない。
    gathered = threading.Barrier(workers, timeout=1.0)

    def slow_search(self, *args, **kwargs):
        nonlocal inside, peak
        scanning = kwargs.get("min_mora") is not None or kwargs.get("max_mora") is not None
        if not scanning:
            return original(self, *args, **kwargs)
        with guard:
            inside += 1
            peak = max(peak, inside)
        try:
            # 門が閉じていれば全員が揃わずタイムアウトで割れる。それが期待動作。
            with contextlib.suppress(threading.BrokenBarrierError):
                gathered.wait()
            return original(self, *args, **kwargs)
        finally:
            with guard:
                inside -= 1

    with (
        mock.patch.object(web_module.PhoneticSearcher, "search", slow_search),
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        responses = list(
            pool.map(
                lambda _: client.get(
                    "/api/similar",
                    params={"q": "サカナ", "limit": 5, "min_mora": 1, "max_mora": 8},
                ),
                range(workers),
            )
        )

    assert all(r.status_code == 200 for r in responses)
    # 門の本数を超えないこと。加えて 8 本を投げているので、門が無ければ
    # barrier が揃って peak == workers になる — つまり workers 未満で
    # あることが「門が実際に絞った」証拠になる。
    assert peak <= web_module.MAX_CONCURRENT_SCANS, f"同時に {peak} 本入った"
    assert peak < workers, f"門が絞っていない (peak={peak} で {workers} 本すべてが同時に入った)"


def test_missing_index_reports_how_to_build(tmp_path) -> None:
    """索引が無くてもサーバは起動でき、原因と対処を 503 で返す。"""
    client = TestClient(create_app(tmp_path / "absent"), raise_server_exceptions=False)
    response = client.get("/api/info")
    assert response.status_code == 503
    assert "build-index" in response.json()["detail"]

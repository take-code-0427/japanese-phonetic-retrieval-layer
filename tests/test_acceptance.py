"""実辞書に対する受け入れテスト。

索引が未構築なら自動的にスキップされる (conftest の real_store フィクスチャ)。
小さなサンプル語彙では確認できない「202 万語の中から意図した語が引けるか」を見る。
"""

from __future__ import annotations

import time

import pytest

from jpr import search as search_module
from jpr.index import Category
from jpr.search import PhoneticSearcher
from jpr.store import PhoneticStore

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

    # 編集距離をバッチ化する前は 1 クエリ平均 440ms、最悪 1.4 秒かかっていた。
    # 現状は中央値 26ms・最大 41ms だが、他のテストと並走すると一時的に跳ねる。
    # 退行 (バッチ化を戻すと 440ms) を捕まえるには 150ms で十分なので、
    # 偶発的な失敗を避けてここに置く。
    assert elapsed < 0.15, f"1 クエリ {elapsed * 1000:.0f}ms は遅すぎる"


def test_mora_range_reaches_words_the_ann_cannot(real_searcher: PhoneticSearcher) -> None:
    """モーラ範囲の指定が ANN の近傍の外に届く。

    この経路が存在する理由そのもの。ANN の候補生成は phonetic 空間の Top-K
    なので、モーラ数の違う語は近傍に入らない。「乳首」(3 モーラ) に対して
    5 モーラの語は、候補数を 20000 に増やしても 1 件も上位に来なかった。
    """
    _, plain = real_searcher.search("乳首", limit=100, candidates=20_000)
    assert not any(r.mora_count == 5 for r in plain), (
        "ANN 経路で 5 モーラ語が届くなら、この機能の前提が変わっている"
    )

    _, ranged = real_searcher.search("乳首", min_mora=5, max_mora=5, limit=50)
    assert ranged
    assert all(r.mora_count == 5 for r in ranged)


def test_mora_range_returns_only_the_requested_lengths(
    real_searcher: PhoneticSearcher,
) -> None:
    _, results = real_searcher.search("乳首", min_mora=4, max_mora=6, limit=100)
    assert results
    assert all(4 <= r.mora_count <= 6 for r in results)
    # 既定の検索では上位が 3 モーラで埋まる。範囲指定はそれを 1 件も通さない。
    assert not any(r.mora_count == 3 for r in results)


def test_mora_range_scan_is_tolerable(real_searcher: PhoneticSearcher) -> None:
    """全走査は ANN より桁違いに遅いが、常駐サーバで使える範囲に収まる。

    実測 (暖まった状態): 5 モーラ単独 (30 万語) で約 1.0 秒、4〜6 モーラ
    (97 万語) で約 3.8 秒。初回は mmap のページフォールトを含むのでもっと
    かかる。行選択をやめて 202 万行すべてに内積を取るような退行を捕まえる
    ための上限として 20 秒を置く。
    """
    real_searcher.search("乳首", min_mora=5, max_mora=5, limit=10)

    started = time.perf_counter()
    real_searcher.search("乳首", min_mora=5, max_mora=5, limit=50)
    elapsed = time.perf_counter() - started
    assert elapsed < 20.0, f"5 モーラの全走査に {elapsed:.1f} 秒は遅すぎる"


def test_threshold_then_take_everything(real_searcher: PhoneticSearcher) -> None:
    """閾値で母集団を切って全件取る、という使い方が成り立つ。

    `limit` の上限で切られると「スコア 0.9 以上を全部見る」ができない。
    """
    _, results = real_searcher.search("ラーメン", limit=None, min_score=0.8)
    assert all(r.score >= 0.8 for r in results)
    # 既定の limit (10) や選抜幅 (limit * _RERANK_MARGIN = 80) で
    # 頭打ちになっていない。実測で 95 件返る。
    assert len(results) > 80, f"{len(results)} 件では上限で切られている疑いがある"

    # 閾値を上げれば件数は減る。上限ではなくスコアが件数を決めている。
    _, stricter = real_searcher.search("ラーメン", limit=None, min_score=0.9)
    assert 0 < len(stricter) < len(results)


def test_limit_is_filled_when_candidates_allow(real_searcher: PhoneticSearcher) -> None:
    """要求件数ぶん候補があるなら、その数だけ返る。

    rerank は上位だけを `SearchResult` に起こすので、選抜幅を固定にすると
    同音異表記を畳んだ後に件数が足りなくなる。実測では mishearing の
    「東京」が 20 件要求に対し 12 件しか返らなかった。
    """
    for preset in ("pun", "rhyme", "mishearing"):
        for query in ("東京", "科学", "ラーメン"):
            _, results = real_searcher.search(query, limit=20, preset=preset)
            assert len(results) == 20, f"{preset}:{query} が {len(results)} 件しか返らない"


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
    """ASR 補正用途。誤認識された読みから正しい語を引き戻せる。

    「チキュウビ」のように 1 音素だけ崩れた入力から「地球儀 (チキュウギ)」に
    戻す。コンセプトにある「地球日」は Sudachi が「地球 + 日」と解析して
    チキュウニチ (6 モーラ) になるので、音の崩れとしては別物になる。
    """
    _, results = real_searcher.search("チキュウビ", preset="mishearing", limit=20)
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


#: 速度のために持っている「素朴でない経路」が結果を変えていないことを見る。
#:
#: 内積の取り方には 2 通りある (`_space_scores`)。候補行だけを mmap から引く
#: 素朴な形と、全行と内積を取ってからマスクする形で、後者は 97 万行の選抜で
#: 14 倍速いが触るデータが違う。速いほうだけが使われる状況 (数十万件の全走査)
#: と遅いほうだけが使われる状況 (ANN の 2000 件) が実際に両方あるので、
#: **どちらを通っても同じ順位・同じスコアになる**ことを固定する。
#:
#: 索引が小さいと `_FULL_SCAN_RATIO` の片側しか踏めないので、実辞書のテストに置く。
_EQUIVALENCE_QUERIES = ["乳首", "科学", "コンピュータ", "図書館", "ありがとう"]


def _fingerprint(searcher: PhoneticSearcher, query: str, **kwargs: object) -> list[tuple]:
    """順位とスコアの内訳をまとめて、経路の違いを検出できる形にする。"""
    _, results = searcher.search(query, limit=30, **kwargs)  # type: ignore[arg-type]
    return [
        (
            r.surface,
            r.reading,
            r.score,
            r.phonetic_similarity,
            r.coda_similarity,
            r.vowel_similarity,
        )
        for r in results
    ]


@pytest.mark.parametrize("query", _EQUIVALENCE_QUERIES)
def test_full_scan_dot_product_matches_row_selection(real_store: PhoneticStore, query: str) -> None:
    """全行内積と行選抜が、ANN 経路でも全走査でも同じ結果を返す。"""
    always_full = PhoneticSearcher(real_store)
    always_full._FULL_SCAN_RATIO = 0.0
    always_rows = PhoneticSearcher(real_store)
    always_rows._FULL_SCAN_RATIO = float("inf")

    assert _fingerprint(always_full, query) == _fingerprint(always_rows, query)
    assert _fingerprint(always_full, query, min_mora=5, max_mora=5) == _fingerprint(
        always_rows, query, min_mora=5, max_mora=5
    )


@pytest.mark.parametrize("query", _EQUIVALENCE_QUERIES)
@pytest.mark.parametrize("preset", ["pun", "rhyme", "mishearing"])
def test_edit_distance_pruning_does_not_change_ranking(
    real_store: PhoneticStore,
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    preset: str,
) -> None:
    """編集距離の打ち切りが順位とスコアを変えない。

    `_survivors` は「上位 N 件は全件計算と一致する」ことだけを保証する厳密な
    打ち切りで、近似ではない。効き方は `weights.phoneme` に依存する
    (実測の生存率は rhyme 0.2% / pun 18.7% / mishearing 97.0%) ので、
    プリセットを変えて両端を踏む。

    候補が数十万件になる全走査経路でしか枝刈りは起きないため、実辞書に置く。
    """
    pruned = PhoneticSearcher(real_store)
    reference = PhoneticSearcher(real_store)

    baseline = _fingerprint(pruned, query, min_mora=4, max_mora=6, preset=preset)

    # 標本が候補数を上回ると `_survivors` は全件計算に落ちる。これを対照にする。
    monkeypatch.setattr(search_module, "_PROBE_CANDIDATES", 1 << 30)
    assert baseline == _fingerprint(reference, query, min_mora=4, max_mora=6, preset=preset)

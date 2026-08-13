"""実辞書に対する受け入れテスト。

索引が未構築なら自動的にスキップされる (conftest の real_store フィクスチャ)。
小さなサンプル語彙では確認できない「202 万語の中から意図した語が引けるか」を見る。
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from jpr import search as search_module
from jpr.embedding import embed
from jpr.index import Category
from jpr.phonology import analyze_reading
from jpr.search import PhoneticSearcher
from jpr.store import INDEXED_SPACES, PhoneticStore

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


def test_candidate_generation_reaches_the_target(real_searcher: PhoneticSearcher) -> None:
    """候補生成が「チョコビ」に到達できる。

    精密な編集距離では 0.808 で数百位に沈む語だが、埋め込み空間では上位に来る。
    2 段構成が成立する前提条件。

    候補生成が int8 の量子化内積になったので、**この経路は量子化の誤差が
    候補を落としていないことの確認も兼ねる** (`store._quantize` の項を参照)。
    """
    _, results = real_searcher.search("乳首", limit=400, candidates=2000)
    assert any(r.surface == "チョコビ" for r in results)


def test_concept_example_chikyugi(real_searcher: PhoneticSearcher) -> None:
    """コンセプトのもう 1 つの例。「乳首」と「地球儀」も音韻的に近い。"""
    comparison = real_searcher.compare("乳首", "地球儀")
    assert comparison.similarity > 0.7


def test_search_is_fast(real_searcher: PhoneticSearcher) -> None:
    """202 万語に対して 1 クエリが実用的な速度で返る。

    素朴な実装では 12 秒かかっていた (Python の編集距離を全候補に回していた頃)。
    捕まえたいのはその桁の退行で、数十 ms の揺れではない。

    **平均ではなく最小値で見る。** この計測は負荷の影響を強く受け、実測で
    load average 2 と 38 では同じコードが 40ms と 320ms にぶれた。平均を取ると
    重い 1 回に引きずられるので、「最も条件が良かった 1 回」を基準にする
    (`CLAUDE.md` のモーラ範囲全走査の項と同じ理由)。
    """
    queries = ["ラーメン", "地球儀", "手首", "学校", "電車"]
    # 初回は mmap のページフォールトを含むので、計測前に 1 周流す。
    for query in queries:
        real_searcher.search(query, limit=10)

    timings = []
    for query in queries:
        started = time.perf_counter()
        real_searcher.search(query, limit=10)
        timings.append(time.perf_counter() - started)

    # 低負荷時の実測は 27〜63ms (full / 202 万語、候補生成の内積が支配的)。
    # 桁違いの退行だけを捕まえたいので、最小値に対して 10 倍の余裕を取る。
    assert min(timings) < 0.5, (
        f"最速でも 1 クエリ {min(timings) * 1000:.0f}ms かかる "
        f"(全体: {[f'{t * 1000:.0f}ms' for t in timings]})"
    )


def test_mora_range_reaches_words_the_ann_cannot(real_searcher: PhoneticSearcher) -> None:
    """モーラ範囲の指定が ANN の近傍の外に届く。

    この経路が存在する理由そのもの。ANN の候補生成は phonetic 空間の Top-K
    なので、モーラ数の違う語は近傍に入らない。「乳首」(3 モーラ) に対して
    5 モーラの語は、候補数を 20000 に増やしても 1 件も上位に来なかった。

    **包含経路 (`_containment`) で届く語だけは例外。** あちらは音素列を索引
    全体で走査するのでモーラ帯を越えるが、拾えるのは「クエリを完全な形で
    含む語」(乳首券 = チクビケン) に限られる。それ以外の 5 モーラ語 —
    「筑前煮」のような音が近いだけの語 — は今も届かないので、範囲指定が
    要る理由は変わっていない。
    """
    _, plain = real_searcher.search("乳首", limit=100, candidates=20_000)
    stray = [r for r in plain if r.mora_count == 5 and not r.containment]
    assert not stray, (
        "包含以外の経路で 5 モーラ語が届くなら、この機能の前提が変わっている: "
        f"{[r.surface for r in stray]}"
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


def test_group_intervals_cover_every_row(real_store: PhoneticStore) -> None:
    """`group_starts` の区間を展開すると `group_ids` に一致する。

    候補生成はグループで Top-K を取ってから行へ展開するので
    (`search._expand_groups`)、この対応が 1 行でもずれると**別の語のスコアを
    配る**。実索引の全行で確かめる (full 202 万行で 30ms 程度)。
    """
    starts = real_store.group_starts
    assert starts.size == real_store.group_count + 1
    assert int(starts[-1]) == len(real_store)
    # 空のグループがあると展開の長さが合わなくなる。
    assert (np.diff(starts) > 0).all()

    expanded = np.repeat(np.arange(real_store.group_count), np.diff(starts))
    assert np.array_equal(expanded, np.asarray(real_store.group_ids))


def test_stored_vectors_match_the_reading(real_store: PhoneticStore) -> None:
    """索引のベクトルが、その行の読みから作り直した埋め込みと一致する。

    ベクトルは音素列グループごとに 1 本しか持たないので (v5)、行 -> グループの
    写像を間違えると**別の語のベクトルを黙って返す**。値そのものを突き合わせて
    確かめる — 行数や形だけの検証では写像のずれが通ってしまう。

    量子化してあるので許容誤差を置く (int8 の再構成誤差は 0.0026)。
    """
    generator = np.random.default_rng(0)
    rows = generator.choice(len(real_store), 200, replace=False)

    for row in rows:
        row = int(row)
        expected = embed(analyze_reading(real_store.reading(row)), INDEXED_SPACES)
        group = int(real_store.group_ids[row])
        for space in INDEXED_SPACES:
            stored = real_store.vectors(space)[group].astype(np.float32) * real_store.scale(space)
            assert stored == pytest.approx(expected[space], abs=0.02), (space, row)


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


@pytest.mark.parametrize("query", _EQUIVALENCE_QUERIES)
@pytest.mark.parametrize("preset", ["pun", "rhyme", "mishearing"])
def test_mora_band_only_drops_rows_the_filter_would_reject(
    real_store: PhoneticStore,
    query: str,
    preset: str,
) -> None:
    """候補生成のモーラ帯が、後段の判定と同じ行だけを落とす (v6)。

    `_top_candidates` は帯の外の行に内積を取らない。これが安全なのは、
    そこが直後に `_apply_cheap_filters` の `_MAX_MORA_GAP` で必ず捨てられる
    行だから — **帯の幅とギャップ判定が同じ定数を使っていることが根拠**で、
    片方だけを動かすとこの前提が崩れる。

    「帯を広げても結果が同じ」ではないことに注意する。帯を広げると Top-K が
    遠いモーラ数の語で埋まり、**近い語が候補から押し出されて結果が悪くなる**
    (実測で「乳首」の 8 位が 低み 0.8914 -> シクベ 0.8818 に落ちた)。帯は
    取りこぼしを減らす方向に働くので、対照に取れるのは「結果の全件がそもそも
    帯の中にある」ことのほう。

    **包含候補 (`_containment`) だけは帯の外から来る。** あちらは音素列を索引
    全体で走査するので帯を無視し、代わりに占有率が余分の多さを減点する。
    帯の主張はこの経路を除いた分について立てる。
    """
    searcher = PhoneticSearcher(real_store)
    pronunciation, results = searcher.search(query, limit=30, preset=preset)
    assert results

    gap = searcher._MAX_MORA_GAP
    for result in results:
        if result.containment:
            continue
        assert abs(result.mora_count - pronunciation.mora_count) <= gap, result


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # 語頭に余分が付く形 (axxx)。
        ("りんご", "ラリンゴ"),
        # 語末に余分が付く形 (xxxbbb)。
        ("科学", "外科学"),
        # 両端に余分が付く形。
        ("電話", "エア電話"),
    ],
)
def test_containment_reaches_words_the_top_k_cannot(
    real_searcher: PhoneticSearcher,
    query: str,
    expected: str,
) -> None:
    """クエリを完全な形で含む語が上位に来る。**この機能の存在理由。**

    2 つのことを同時に主張している。

    1. **候補生成の Top-K では拾えない。** 包含は phonetic 空間の近さと
       相関しないので、`candidates` をいくら増やしても届かない (実測で
       「りんご」を含む 204 グループのうち Top-8000 に入るのは 48 件、
       モーラ帯に限っても 123 件のうち 48 件)。だから `_containment` は
       後段のフィルタではなく候補生成そのものを足している。
    2. **編集距離の重みでは代わりにならない。** 距離は挿入を一律に減点する
       ので、包含語 (riNgo in rariNgo で 0.735) が 1 音素だけ違う同じ長さの語
       (riNbo で 0.933) に必ず負ける。
    """
    _, results = real_searcher.search(query, limit=10)
    surfaces = [r.surface for r in results]
    assert expected in surfaces, f"包含語が上位 10 件に入らない: {surfaces}"

    # 包含として認識されている (偶然音が近くて上がったのではない)。
    found = next(r for r in results if r.surface == expected)
    assert found.containment > 0.0

    # 重みを切ると届かなくなる = Top-K だけでは拾えていない。
    weights = search_module.PRESETS["pun"]
    without = search_module.ScoreWeights(
        phoneme=weights.phoneme,
        embedding=weights.embedding,
        mora=weights.mora,
        coda=weights.coda,
        vowel=weights.vowel,
        familiarity=weights.familiarity,
        containment=0.0,
    )
    _, plain = real_searcher.search(query, limit=10, weights=without, candidates=20_000)
    assert expected not in [r.surface for r in plain], (
        "包含の重みなしで届くなら、この経路の前提が変わっている"
    )


def test_containment_does_not_double_count_the_surplus(
    real_searcher: PhoneticSearcher,
) -> None:
    """包含した候補の編集距離は 1.0 になる (`_score_candidates`)。

    同じ「余分」を `phonetic` と `containment` が二重に数えると加点が
    打ち消される。実測で「りんご」->「ラリンゴ」は占有率 0.714 で +0.093
    得るのに、編集距離が 0.735 と非包含語 (0.93〜0.98) より低くて -0.087
    失い、重みを 0.25 まで上げないと順位が動かなかった。**役割を分けて
    いる** — 一致したかどうかは `phonetic`、余分の多寡は `containment`。
    """
    _, results = real_searcher.search("りんご", limit=20)
    contained = [r for r in results if r.containment]
    assert contained, "包含語が 1 件も上位に来ていない"
    for result in contained:
        assert result.phonetic_similarity == 1.0, result


def test_containment_is_scaled_by_how_much_of_the_word_it_covers(
    real_searcher: PhoneticSearcher,
) -> None:
    """占有率が余分の多さを減点する。

    一律 1.0 にすると、短いクエリを含む長い地名や複合語が同じ強さで上がる。
    分母は**候補全体の音素数**で、一致に消費した長さではない (そちらにすると
    特殊モーラを挟んだ語が 1.0 になり、余分の多寡が消える)。
    """
    _, results = real_searcher.search("電話", limit=50)
    contained = {r.surface: r for r in results if r.containment}
    assert "エア電話" in contained and "テレビ電話" in contained

    # エアデンワ (7 音素) は 5/7、テレビデンワ (11 音素) は 5/11。
    assert contained["エア電話"].containment > contained["テレビ電話"].containment


def test_containment_stays_inside_an_explicit_mora_range(
    real_searcher: PhoneticSearcher,
) -> None:
    """モーラ範囲を明示したら包含も範囲の外へ出ない。

    包含はモーラ帯 (`_MAX_MORA_GAP`) を無視するが、それは「安全網が候補生成の
    粗さを補うためのもの」だから。呼び出し側が範囲を明示した経路では意図が
    あるので、包含だからといって範囲外を返してよい理由がない。
    """
    _, results = real_searcher.search("科学", min_mora=4, max_mora=4, limit=50)
    assert results
    assert all(r.mora_count == 4 for r in results)
    # 4 モーラの包含語 (カガクブ・ゲカガク など) は範囲内なので届く。
    assert any(r.containment for r in results)


def test_rhyme_does_not_use_containment(real_searcher: PhoneticSearcher) -> None:
    """rhyme プリセットは包含を見ない。

    韻は語尾の一致を見るので、語頭に余分が付いているかどうかが関係しない。
    「りんご」に対する「ラリンゴ」は韻としては「リンゴ」と同じ扱いでよく、
    包含を足すと `coda` と競合して語尾の弱い包含語が混ざる。
    """
    assert search_module.PRESETS["rhyme"].containment == 0.0
    _, results = real_searcher.search("りんご", limit=20, preset="rhyme")
    assert results
    assert all(r.containment == 0.0 for r in results)


# ---------- 分割合成 (空耳) ----------
#
# 小さなサンプル語彙では「意図した空耳が引けるか」を確認できない。
# 202 万語の中から語 + 助詞の連なりが組めるかはここでしか捕まらない。


def test_phrase_reconstructs_a_known_sentence(real_searcher: PhoneticSearcher) -> None:
    """音が完全に一致する語 + 助詞の列を組める。

    「ワタシノナマエハ」は 8 モーラで、これに音が近い**単一の語**は辞書に
    無い (だから `search` では答えが返らない)。区間に切れば
    私 + の + 名前 + は で音が完全に一致する。
    """
    _, candidates = real_searcher.compose("わたしのなまえは", limit=10)
    assert candidates
    readings = {c.reading for c in candidates}
    assert "ワタシノナマエハ" in readings, readings
    # 音が完全に一致する候補が上位に来る。
    exact = [c for c in candidates if c.reading == "ワタシノナマエハ"]
    assert all(s.similarity == 1.0 for s in exact[0].segments)


def test_phrase_uses_particles_as_connectives(real_searcher: PhoneticSearcher) -> None:
    """1 モーラの区間が助詞で埋まる。

    索引に 1 モーラの語が 1 件も無いので (full 辞書で 2 モーラ 38210 件に対し
    1 モーラ 0 件)、助詞の内蔵表が無いとこの位置が埋まらず、
    「ワタシ|ノ|ナマエ|ハ」のような切り方そのものが作れない。
    """
    _, candidates = real_searcher.compose("わたしのなまえは", limit=10)
    best = next(c for c in candidates if c.reading == "ワタシノナマエハ")
    particles = [s for s in best.segments if s.is_particle]
    assert particles, best.segments
    assert {s.reading for s in particles} >= {"ノ"}


def test_phrase_segments_are_readable_words(real_searcher: PhoneticSearcher) -> None:
    """区間に当てる語が読める表記になっている。

    数字混じりの見出し (「2 士」「8 耐」) は索引に入っているが、列に混ざると
    全体が読めなくなる (`phrase._is_readable_surface`)。
    """
    for text in ("こんにちはせかい", "わたしのなまえは"):
        _, candidates = real_searcher.compose(text, limit=10)
        for candidate in candidates:
            for segment in candidate.segments:
                assert not any(ch.isascii() and ch.isalnum() for ch in segment.surface), (
                    f"{text}: {candidate.text}"
                )


def test_phrase_prefers_known_words_over_rare_kanji(real_searcher: PhoneticSearcher) -> None:
    """コスト 0 の語 (活用の断片・稀な異表記) が上位を埋めない。

    `index.familiarity_of` はコストの反転なので `cost <= 0` を一般性 1.0 と
    見るが、full 辞書の該当 51732 件は「炮り」「合ん」「アがん」のような
    断片で実際には一般的でない。`phrase.UNKNOWN_COST_FAMILIARITY` で
    「わからない」として扱っている。これが 1.0 のままだと
    「わたしのなまえは」が「分か死の七異派」になった (実測)。
    """
    _, candidates = real_searcher.compose("わたしのなまえは", limit=5)
    assert candidates
    # 上位に「私」で始まる候補が来る。
    assert any(c.segments[0].surface == "私" for c in candidates), [c.text for c in candidates]


def test_phrase_scores_stay_within_the_unit_range(real_searcher: PhoneticSearcher) -> None:
    """実辞書でもスコアが 0〜1 に収まる。通常検索と同じ尺度で読めること。"""
    _, candidates = real_searcher.compose("ありがとうございます", limit=10)
    assert candidates
    for candidate in candidates:
        assert 0.0 <= candidate.score <= 1.0
        for segment in candidate.segments:
            assert 0.0 <= segment.similarity <= 1.0


def test_phrase_covers_long_input_without_gaps(real_searcher: PhoneticSearcher) -> None:
    """長い入力でも区間が隙間なく覆う。ビームの刈り込みが経路を壊さないこと。"""
    pronunciation, candidates = real_searcher.compose("アルミカンノウエニアルミカン", limit=5)
    assert candidates
    for candidate in candidates:
        spans = [(s.start, s.end) for s in candidate.segments]
        assert spans[0][0] == 0
        assert spans[-1][1] == pronunciation.mora_count
        for left, right in itertools.pairwise(spans):
            assert left[1] == right[0]


def test_phrase_lattice_folds_repeated_words(real_searcher: PhoneticSearcher) -> None:
    """実辞書でラティスが候補の重複を畳む。

    候補リストでは同じ語が何度も出る (実測で区間の 65〜77% が重複し、
    「名前」「は」は上位 10 件の全部に現れた)。畳めば 1 度しか描かれない。
    """
    _, lattice = real_searcher.lattice("わたしのなまえは", node_budget=40)
    assert lattice.nodes
    # 畳む前の延べ区間数より少ない。
    spans = sum(len(c.segments) for c in lattice.paths)
    assert lattice.node_count < spans
    # 「名前」が 1 ノードだけ現れ、多くの経路が通る。
    names = [n for n in lattice.nodes if n.surface == "名前"]
    assert len(names) == 1, [n.surface for n in lattice.nodes]
    assert names[0].path_count > 1


def test_phrase_lattice_stays_connected_at_scale(real_searcher: PhoneticSearcher) -> None:
    """実辞書でも全ノードが始端から終端まで繋がる経路上にある。

    予算に収めるとき経路単位で削っているので (`phrase._fit_lattice`)、
    孤立ノードが出ないこと。ここが崩れると図として読めなくなる。
    """
    for text, budget in (
        ("わたしのなまえは", 40),
        ("わたしのなまえは", 10),
        ("アルミカンノウエニアルミカン", 40),
        ("ちくび", 40),
    ):
        pronunciation, lattice = real_searcher.lattice(text, node_budget=budget)
        ids = {n.id for n in lattice.nodes}
        by_id = {n.id: n for n in lattice.nodes}

        forward = {e.target for e in lattice.edges if e.source is None and e.target}
        changed = True
        while changed:
            changed = False
            for edge in lattice.edges:
                if edge.source in forward and edge.target and edge.target not in forward:
                    forward.add(edge.target)
                    changed = True
        backward = {e.source for e in lattice.edges if e.target is None and e.source}
        changed = True
        while changed:
            changed = False
            for edge in lattice.edges:
                if edge.target in backward and edge.source and edge.source not in backward:
                    backward.add(edge.source)
                    changed = True

        assert ids == forward & backward, f"{text} budget={budget}: 孤立ノード"
        for edge in lattice.edges:
            if edge.source is not None and edge.target is not None:
                assert by_id[edge.source].end == by_id[edge.target].start
            if edge.target is None and edge.source is not None:
                assert by_id[edge.source].end == pronunciation.mora_count


def test_phrase_lattice_widens_the_beam_to_fill_the_budget(
    real_searcher: PhoneticSearcher,
) -> None:
    """予算を上げるとノードが増える。

    候補数ではなくビーム幅を広げることで育つ経路なので (`limit` を増やしても
    ノードは増えない)、予算が実際に効いていることを見る。
    """
    _, small = real_searcher.lattice("わたしのなまえは", node_budget=10)
    _, large = real_searcher.lattice("わたしのなまえは", node_budget=60)
    assert small.node_count <= large.node_count
    assert large.beam_width >= small.beam_width

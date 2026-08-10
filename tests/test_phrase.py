"""分割合成 (空耳) のテスト。

`sample_store` の小さな語彙 (18 語) で完結する。実辞書で「意図した空耳が
引けるか」は `test_acceptance.py` 側で見る。
"""

from __future__ import annotations

import itertools

import pytest

from jpr.phonology import analyze_reading
from jpr.phrase import (
    PARTICLE_BONUS,
    WEIGHT_FAMILIARITY,
    WEIGHT_INFORMATIVENESS,
    WEIGHT_PHONETIC,
    PhraseComposer,
    _informativeness,
    _is_readable_surface,
)
from jpr.search import PhoneticSearcher
from jpr.store import PhoneticStore


@pytest.fixture(scope="module")
def composer(sample_store: PhoneticStore) -> PhraseComposer:
    return PhraseComposer(sample_store)


def compose(composer: PhraseComposer, reading: str, **kwargs):
    """かな読みをそのまま渡す。索引だけで完結するので Sudachi を通さない。"""
    return composer.compose(reading, pronunciation=analyze_reading(reading), **kwargs)


def test_segments_cover_the_input_without_gaps(composer: PhraseComposer) -> None:
    """区間は入力を隙間なく、重なりなく覆う。

    ここが崩れると「入力のどこが何になったか」が読めなくなり、合成結果を
    空耳として検証できない。
    """
    pronunciation, candidates = compose(composer, "チョコビラーメン", limit=5)
    assert candidates
    for candidate in candidates:
        spans = [(s.start, s.end) for s in candidate.segments]
        assert spans[0][0] == 0
        assert spans[-1][1] == pronunciation.mora_count
        for left, right in itertools.pairwise(spans):
            assert left[1] == right[0]


def test_text_and_reading_are_the_concatenation_of_segments(
    composer: PhraseComposer,
) -> None:
    _, candidates = compose(composer, "チョコビラーメン", limit=5)
    assert candidates
    for candidate in candidates:
        assert candidate.text == "".join(s.surface for s in candidate.segments)
        assert candidate.reading == "".join(s.reading for s in candidate.segments)


def test_long_input_is_split_into_several_words(composer: PhraseComposer) -> None:
    """語 1 つでは覆えない長さの入力が複数の区間に分かれる。

    これが分割合成の存在理由 — 通常の `search` は 1 語を 1 語に写すので、
    語彙に無い長さの入力には答えを返せない。
    """
    _, candidates = compose(composer, "チョコビラーメン", limit=5)
    assert candidates
    assert candidates[0].segment_count >= 2


def test_source_reading_matches_the_input_span(composer: PhraseComposer) -> None:
    """区間が持つ入力側の読みが、実際にその位置のモーラ列と一致する。"""
    pronunciation, candidates = compose(composer, "チョコビラーメン", limit=3)
    assert candidates
    for candidate in candidates:
        for segment in candidate.segments:
            expected = "".join(m.kana for m in pronunciation.moras[segment.start : segment.end])
            assert segment.source_reading == expected


def test_scores_stay_within_the_unit_range(composer: PhraseComposer) -> None:
    """スコアは 0〜1 に収まる。

    通常検索と同じ尺度でなければ 2 つの経路の数値を並べて読めない。
    区間スコアを加算 (`similarity + 0.25 * familiarity`) で作っていた頃は
    1.25 まで伸びていた。
    """
    _, candidates = compose(composer, "チョコビラーメン", limit=10)
    assert candidates
    for candidate in candidates:
        assert 0.0 <= candidate.score <= 1.0
        assert 0.0 <= candidate.phonetic_similarity <= 1.0
        for segment in candidate.segments:
            assert 0.0 <= segment.similarity <= 1.0


def test_results_are_sorted_by_score(composer: PhraseComposer) -> None:
    _, candidates = compose(composer, "チョコビラーメン", limit=10)
    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_limit_is_respected(composer: PhraseComposer) -> None:
    _, candidates = compose(composer, "チョコビラーメン", limit=3)
    assert len(candidates) <= 3


def test_duplicate_surfaces_are_folded(composer: PhraseComposer) -> None:
    """同じ表層に落ちる経路は 1 件に畳む。

    区間の切り方が違っても表層が同じなら、結果としては同じものしか読めない。
    """
    _, candidates = compose(composer, "チョコビラーメン", limit=20)
    texts = [c.text for c in candidates]
    assert len(texts) == len(set(texts))


def test_particles_fill_single_mora_spans(sample_store: PhoneticStore) -> None:
    """1 モーラの区間は助詞でしか埋まらない。

    索引には 1 モーラの語が 1 件も無い (実測: full 辞書で 2 モーラ 38210 件に
    対し 1 モーラ 0 件)。助詞の内蔵表はそのための必須部品で、利便のための
    ものではない。
    """
    composer = PhraseComposer(sample_store)
    pronunciation = analyze_reading("ソラ")
    options = composer._chunk_options(
        pronunciation.moras,
        max_chunk_moras=2,
        chunk_candidates=12,
        min_chunk_score=0.5,
        allow_particles=True,
    )
    single = options.get((0, 1))
    assert single, "1 モーラの区間に候補が無い"
    assert all(option.is_particle for option in single)


def test_particles_can_be_disabled(composer: PhraseComposer) -> None:
    _, candidates = compose(composer, "チョコビラーメン", limit=10, allow_particles=False)
    for candidate in candidates:
        assert not any(s.is_particle for s in candidate.segments)


def test_particle_readings_cover_both_spellings(sample_store: PhoneticStore) -> None:
    """「は」は ワ と ハ の両方の読みを持つ。

    助詞としての発音は [wa] だが、かなを並べた入力は表記どおりの ハ で来る。
    片方しか持たないと、もう片方の入力でその位置に助詞が当たらない。
    """
    composer = PhraseComposer(sample_store)
    single = composer._particles[1]
    readings = {option.reading for option in single if option.surface == "は"}
    assert readings == {"ワ", "ハ"}


def test_informativeness_prefers_kanji_over_kana_headwords() -> None:
    """読みをそのまま書いた見出しは最も低く見る。

    `search._surface_informativeness` と同じ判断。分割合成では区間ごとに
    表記を選ぶので、これが無いとカタカナだけの列が上位を埋める。
    """
    assert _informativeness("価格", "カカク") == 1.0
    assert _informativeness("カカク", "カカク") == 0.0
    assert _informativeness("しくみ", "シクミ") == 0.5


def test_surfaces_with_digits_are_rejected() -> None:
    """数字混じりの見出しは合成結果の一部として読めない。

    `index.is_searchable_surface` は「かなか漢字が 1 文字でもあれば通す」ので
    「2 士」「8 耐」が索引に入っている。1 語だけ返す通常検索では目立たないが、
    列に混ざると全体が読めなくなる。
    """
    assert _is_readable_surface("名前")
    assert _is_readable_surface("ラーメン")
    assert not _is_readable_surface("2士")
    assert not _is_readable_surface("8耐")
    assert not _is_readable_surface("Ｂ級")


def test_particle_weight_uses_the_same_scale_as_words() -> None:
    """助詞の重みも語と同じ配分の式に載る。

    別の式で作ると区間スコアの尺度が助詞のところだけ別物になり、
    語と助詞の優劣が比較できなくなる。
    """
    perfect = (WEIGHT_PHONETIC + WEIGHT_FAMILIARITY + WEIGHT_INFORMATIVENESS * 0.5) * PARTICLE_BONUS
    # 音が完全に一致する助詞の重み。1 を超えない範囲に収まる。
    assert 0.0 < perfect < 1.0


def test_empty_input_returns_nothing(composer: PhraseComposer) -> None:
    """読みが取れない入力では空を返す (例外にしない)。"""
    pronunciation, candidates = compose(composer, "")
    assert pronunciation.mora_count == 0
    assert candidates == []


def test_searcher_exposes_compose(sample_searcher: PhoneticSearcher) -> None:
    """`PhoneticSearcher.compose` が漢字入りの入力を読みに直して合成する。"""
    pronunciation, candidates = sample_searcher.compose("チョコビラーメン", limit=3)
    assert pronunciation.reading == "チョコビラーメン"
    assert candidates


def test_composer_is_reused_across_calls(sample_searcher: PhoneticSearcher) -> None:
    """合成器は使い回す。モーラ数ごとの行の選抜が毎回走ると遅い (実測 150〜330ms)。"""
    assert sample_searcher.composer is sample_searcher.composer


# ---------- ラティス (候補群を 1 枚の DAG に畳む) ----------


def test_lattice_folds_repeated_words_into_one_node(composer: PhraseComposer) -> None:
    """同じ区間の同じ語が 1 ノードに畳まれる。

    これがラティス表示の存在理由。候補リストでは同じ語が候補ごとに何度も
    現れる (実測で表示している区間の 65〜77% が重複)。
    """
    pronunciation = analyze_reading("チョコビラーメン")
    _, lattice = composer.lattice("チョコビラーメン", pronunciation=pronunciation)
    assert lattice.nodes
    # ノード id は一意。
    ids = [n.id for n in lattice.nodes]
    assert len(ids) == len(set(ids))
    # 畳む前の延べ区間数より必ず少ない (同じ語が複数の経路に出るので)。
    spans = sum(len(c.segments) for c in lattice.paths)
    assert lattice.node_count <= spans


def test_lattice_nodes_are_all_on_a_complete_path(composer: PhraseComposer) -> None:
    """全ノードが始端から終端まで繋がる経路上にある (孤立ノードが無い)。

    ノードを個別に削ると残りが繋がらなくなり「経路が読める図」でなくなるので、
    予算に収めるときも経路単位で削っている (`_fit_lattice`)。
    """
    for budget in (40, 15, 8, 4):
        pronunciation = analyze_reading("チョコビラーメン")
        _, lattice = composer.lattice(
            "チョコビラーメン", pronunciation=pronunciation, node_budget=budget
        )
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

        assert ids == forward & backward, f"budget={budget} で孤立ノードがある"
        # 始端の辺は位置 0 から、終端の辺は末尾で終わる。
        for edge in lattice.edges:
            if edge.source is None and edge.target:
                assert by_id[edge.target].start == 0
            if edge.target is None and edge.source:
                assert by_id[edge.source].end == pronunciation.mora_count


def test_lattice_edges_are_contiguous(composer: PhraseComposer) -> None:
    """辺で結ばれたノードはモーラ位置として連続している。

    図は経路を追うためのものなので、繋がっていないノードを結ぶと嘘になる。
    """
    _, lattice = composer.lattice(
        "チョコビラーメン", pronunciation=analyze_reading("チョコビラーメン")
    )
    by_id = {n.id: n for n in lattice.nodes}
    for edge in lattice.edges:
        if edge.source is None or edge.target is None:
            continue
        assert by_id[edge.source].end == by_id[edge.target].start


def test_lattice_respects_the_node_budget(composer: PhraseComposer) -> None:
    """ノード数が予算に収まる。

    予算は「画面が埋まらないこと」の保証。最小のビーム幅でも超えるときは
    経路を削って収める (`_fit_lattice`)。
    """
    for budget in (40, 20, 10, 6):
        _, lattice = composer.lattice(
            "チョコビラーメン",
            pronunciation=analyze_reading("チョコビラーメン"),
            node_budget=budget,
        )
        # 経路が 1 本しか採れないときだけ予算を超え得る (空の図を返さないため)。
        if lattice.path_count > 1:
            assert lattice.node_count <= budget, f"budget={budget}"


def test_lattice_limits_nodes_per_span(composer: PhraseComposer) -> None:
    """1 区間に並ぶノード数が上限に収まる。

    予算は全体の数しか抑えないので、これが無いと短い入力で 1 か所に候補が
    集中して縦長の図になる (実測: 「チクビ」が 24 行・高さ 1138px)。
    """
    limit = 3
    _, lattice = composer.lattice(
        "チョコビラーメン",
        pronunciation=analyze_reading("チョコビラーメン"),
        node_budget=100,
        max_nodes_per_span=limit,
    )
    counts: dict[tuple[int, int], int] = {}
    for node in lattice.nodes:
        key = (node.start, node.end)
        counts[key] = counts.get(key, 0) + 1
    if lattice.path_count > 1:
        assert max(counts.values()) <= limit


def test_lattice_path_counts_match_the_paths(composer: PhraseComposer) -> None:
    """ノードの経路数が実際に通る経路の数と一致する。

    太さや濃さに写す値なので、ずれると図が嘘をつく。
    """
    _, lattice = composer.lattice(
        "チョコビラーメン", pronunciation=analyze_reading("チョコビラーメン")
    )
    counted: dict[str, int] = {}
    for candidate in lattice.paths:
        for segment in candidate.segments:
            key = f"{segment.start}:{segment.end}:{segment.surface}"
            counted[key] = counted.get(key, 0) + 1
    for node in lattice.nodes:
        assert node.path_count == counted[node.id]


def test_lattice_represents_the_same_paths_as_compose(composer: PhraseComposer) -> None:
    """ラティスの経路が `compose` と同じ形をしている。

    見せ方が違うだけで別のものを出しているわけではない、という確認。
    ビーム幅が違えば件数は変わるので、上位の経路が含まれることを見る。
    """
    text = "チョコビラーメン"
    pronunciation = analyze_reading(text)
    _, candidates = composer.compose(text, pronunciation=pronunciation, limit=3)
    _, lattice = composer.lattice(text, pronunciation=pronunciation, node_budget=60)
    lattice_texts = {c.text for c in lattice.paths}
    assert candidates
    assert candidates[0].text in lattice_texts


def test_lattice_nodes_are_ordered_left_to_right(composer: PhraseComposer) -> None:
    """ノードが位置順に並ぶ。図は左から右に読むので、この順で配れると楽。"""
    _, lattice = composer.lattice(
        "チョコビラーメン", pronunciation=analyze_reading("チョコビラーメン")
    )
    starts = [n.start for n in lattice.nodes]
    assert starts == sorted(starts)


def test_lattice_of_empty_input_is_empty(composer: PhraseComposer) -> None:
    pronunciation, lattice = composer.lattice("", pronunciation=analyze_reading(""))
    assert pronunciation.mora_count == 0
    assert lattice.nodes == ()
    assert lattice.edges == ()
    assert lattice.path_count == 0


def test_searcher_exposes_lattice(sample_searcher: PhoneticSearcher) -> None:
    """`PhoneticSearcher.lattice` が漢字入りの入力を読みに直して畳む。"""
    pronunciation, lattice = sample_searcher.lattice("チョコビラーメン")
    assert pronunciation.reading == "チョコビラーメン"
    assert lattice.nodes

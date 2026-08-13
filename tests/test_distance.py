"""音素距離と音韻類似度のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from jpr.distance import (
    _ALL_PHONEMES,
    _IPA,
    PAD_ID,
    PHONEME_TO_ID,
    SUBSTITUTION_COSTS,
    WORST_SUBSTITUTION_COST,
    align_phonemes,
    containment_ratio,
    containment_scan,
    edit_distance_batch,
    edit_distance_csr,
    ipa_transcription,
    phoneme_distance,
    phoneme_ids,
    phoneme_ipa,
    phonetic_similarity,
    vowel_skeleton_similarity,
    weighted_edit_distance,
)
from jpr.phonology import analyze_reading


def similarity(a: str, b: str) -> float:
    return phonetic_similarity(analyze_reading(a), analyze_reading(b))


def vowel_similarity(a: str, b: str) -> float:
    return vowel_skeleton_similarity(analyze_reading(a), analyze_reading(b))


def test_identical_vowel_skeleton_is_maximally_similar() -> None:
    """母音列が同じ語は子音が違っても母音軸では最大類似になる。

    ダジャレ・韻のコーパス研究の第一法則 (Kawahara 2007)。「シャチョウ」と
    「サショウ」(a,o,u = a,o,u) がダジャレとして成立するのはこの性質による。
    """
    assert vowel_similarity("チクビ", "シクミ") == pytest.approx(1.0)
    assert vowel_similarity("シャチョウ", "サショウ") == pytest.approx(1.0)


def test_vowel_skeleton_penalizes_length_mismatch() -> None:
    """長さの違う母音列は挿入コストが積み上がって明確に下がる。

    プーリングした内積だった頃は「カイギ」(a,i,i) と「カタギリシキ」
    (a,a,i,i,i,i) に 0.99 を与えていた。列の照合ならダジャレとして成立する
    近傍 (カイキ) と長さ違いが分離できる。
    """
    matching = vowel_similarity("カイギ", "カイキ")
    stretched = vowel_similarity("カイギ", "カタギリシキ")
    assert matching == pytest.approx(1.0)
    assert stretched < 0.7


def test_vowel_skeleton_separates_different_vowel_patterns() -> None:
    assert vowel_similarity("チクビ", "チョコビ") < vowel_similarity("チクビ", "テクビ")


def test_vowel_skeleton_keeps_special_moras() -> None:
    """促音・撥音は拍を成すので骨格に残る。「フトン」と「フットン」は
    促音の挿入ぶんだけ下がるが、特殊モーラの挿入は安い (0.45) ので近い。"""
    value = vowel_similarity("フトン", "フットン")
    assert 0.8 < value < 1.0


def test_identical_phonemes_have_zero_distance() -> None:
    assert phoneme_distance("k", "k") == 0.0


def test_distance_is_symmetric() -> None:
    for a, b in [("k", "g"), ("i", "e"), ("sh", "ch"), ("N", "n")]:
        assert phoneme_distance(a, b) == phoneme_distance(b, a)


@pytest.mark.parametrize(
    ("near", "far"),
    [
        # 有声性のみの違いは、調音方法まで違う対より近い。
        (("k", "g"), ("k", "m")),
        (("t", "d"), ("t", "n")),
        # 摩擦音同士は、摩擦音と流音より近い。
        (("s", "sh"), ("s", "r")),
        # 母音の高さが 1 段違いは、前後まで違う対より近い。
        (("i", "e"), ("i", "o")),
        # 撥音は鼻子音に近く、それ以外の子音より近い。
        (("N", "n"), ("N", "k")),
    ],
)
def test_relative_ordering(near: tuple[str, str], far: tuple[str, str]) -> None:
    assert phoneme_distance(*near) < phoneme_distance(*far)


def test_consonant_vowel_substitution_is_maximally_costly() -> None:
    """子音と母音の置換は音節構造を壊すので最大コスト。"""
    assert phoneme_distance("k", "a") == pytest.approx(1.0)


def test_identical_sequences_are_fully_similar() -> None:
    assert similarity("ラーメン", "ラーメン") == pytest.approx(1.0)


def test_single_voicing_difference_stays_high() -> None:
    """1 音素の有声性だけが違う語は非常に似ていると判定される。"""
    assert similarity("カガク", "カカク") > 0.95


def test_unrelated_words_score_low() -> None:
    """無関係な語は明確に低い。スコアの解像度が保たれていることの確認。"""
    assert similarity("チクビ", "デンシャ") < 0.45


def test_flagship_pair_similarity() -> None:
    """コンセプトの中心例。意味は無関係だが音は近い。"""
    assert similarity("チクビ", "チョコビ") > 0.75


def test_similarity_ordering_matches_intuition() -> None:
    assert similarity("チクビ", "テクビ") > similarity("チクビ", "マツタケ")
    assert similarity("チクビ", "チクワ") > similarity("チクビ", "ソラ")


def test_empty_input() -> None:
    assert similarity("", "") == pytest.approx(1.0)
    assert similarity("チクビ", "") == 0.0


def test_edit_distance_respects_max_distance_cutoff() -> None:
    """打ち切りを渡しても、上限内の距離は正確な値と一致する。"""
    a = analyze_reading("チクビ").phonemes
    b = analyze_reading("チョコビ").phonemes
    exact = weighted_edit_distance(a, b)

    # 上限が十分大きければ厳密な値を返す。
    assert weighted_edit_distance(a, b, exact + 1.0) == pytest.approx(exact)
    # 上限を下回る場合は上限超えの値を返し、除外の判断に使える。
    assert weighted_edit_distance(a, b, exact - 0.5) > exact - 0.5


def test_worst_substitution_cost_is_bounded() -> None:
    assert 0.0 < WORST_SUBSTITUTION_COST <= 1.0


#: 同じ距離関数に 2 つの実装がある。記号ベース (`weighted_edit_distance`、
#: 可読性優先で `compare` と `align_phonemes` が使う) と Rust (rerank の本線)。
#: 速度のために別実装を持っているだけなので、値が食い違えば検索結果が実装依存に
#: なる。素性表や重みを変えたときに両者が揃っていることをここで担保する。
#:
#: 下のアライメントのテストと前半の対が重なっているが、意図的に別に持つ。
#: あちらは空文字列の対を含み (こちらは空入力を専用のテストで見る)、
#: 検証したい性質も違う (実装間の一致 / コスト総和と距離の一致) ので、
#: 片方の都合で対を足したときにもう片方が巻き込まれないようにしている。
_DISTANCE_PAIRS = [
    ("チクビ", "テクビ"),
    ("チクビ", "チョコビ"),
    ("カガク", "カカク"),
    ("サカナ", "アカ"),
    ("ラーメン", "ローメン"),
    ("マツタケ", "ソラ"),
    ("トウキョウ", "トウギョウ"),
    ("ガッコウ", "ガクコウ"),
    ("アリガトウ", "アリタソウ"),
    ("ア", "アイウエオ"),
]


@pytest.mark.parametrize(("a", "b"), _DISTANCE_PAIRS)
def test_batch_matches_symbolic_per_pair(a: str, b: str) -> None:
    """対ごとに記号版と一致する。1 件だけの行列で境界も見る。"""
    pa = analyze_reading(a).phonemes
    pb = analyze_reading(b).phonemes
    matrix = phoneme_ids(pb).reshape(1, -1).astype(np.int64)
    lengths = np.array([len(pb)], dtype=np.int64)
    actual = edit_distance_batch(phoneme_ids(pa), matrix, lengths)
    assert actual == pytest.approx([weighted_edit_distance(pa, pb)])


def test_batch_implementation_matches_symbolic() -> None:
    """バッチ版は、長さの違う候補を混ぜても 1 件ずつの結果と一致する。

    パディングの扱いを間違えると長さの違う候補だけが狂うので、
    最長でない候補が混ざった行列で検証する。
    """
    query = analyze_reading("チクビ").phonemes
    candidates = [analyze_reading(b).phonemes for _, b in _DISTANCE_PAIRS]

    lengths = np.array([len(p) for p in candidates], dtype=np.int64)
    width = int(lengths.max())
    matrix = np.full((len(candidates), width), PAD_ID, dtype=np.int64)
    for row, phonemes in enumerate(candidates):
        matrix[row, : len(phonemes)] = phoneme_ids(phonemes)

    actual = edit_distance_batch(phoneme_ids(query), matrix, lengths)
    expected = [weighted_edit_distance(query, p) for p in candidates]
    assert actual == pytest.approx(expected)


def test_batch_matches_symbolic_on_random_inputs() -> None:
    """ランダムな長さ・内容でも記号版と一致する。

    手で選んだ対ではパディングと長さの組み合わせを網羅できない。バッチ版の
    壊れ方は「長さの違う候補だけが静かに狂う」形で出るので、長短が混ざった
    行列を大量に投げる。空クエリと長さ 0 の候補も範囲に含めている。
    """
    rng = np.random.default_rng(20260804)
    symbols = list(_ALL_PHONEMES)

    for _ in range(200):
        query = tuple(rng.choice(symbols, size=int(rng.integers(0, 12))))
        count = int(rng.integers(1, 40))
        candidates = [
            tuple(rng.choice(symbols, size=int(rng.integers(0, 25)))) for _ in range(count)
        ]

        lengths = np.array([len(c) for c in candidates], dtype=np.int64)
        width = max(1, int(lengths.max()))
        matrix = np.full((count, width), PAD_ID, dtype=np.int64)
        for row, phonemes in enumerate(candidates):
            if phonemes:
                matrix[row, : len(phonemes)] = phoneme_ids(phonemes)

        actual = edit_distance_batch(phoneme_ids(query), matrix, lengths)
        expected = [weighted_edit_distance(query, c) for c in candidates]
        assert actual == pytest.approx(expected, abs=1e-5)


def test_csr_matches_padded_path() -> None:
    """CSR 経路とパディング行列経路が一致する。

    検索は CSR 経路を使うが、他のテストが検証しているのはパディング経路なので
    両者が揃っていることを明示的に見る。CSR 側は索引の音素 ID を距離テーブルの
    ID に写す段を余分に持つので、そこがずれると全候補が静かに狂う。
    """
    words = ["チクビ", "テクビ", "チョコビ", "カガク", "サカナ", "ア", "アイウエオ"]
    candidates = [analyze_reading(w).phonemes for w in words]
    query = phoneme_ids(analyze_reading("チクビ").phonemes)

    # 索引と同じ CSR を手で組む。ここでは索引 ID = 距離 ID にしておく。
    blob: list[int] = []
    bounds = np.zeros(len(candidates) + 1, dtype=np.int64)
    for row, phonemes in enumerate(candidates):
        blob.extend(int(i) for i in phoneme_ids(phonemes))
        bounds[row + 1] = len(blob)
    identity = np.arange(SUBSTITUTION_COSTS.shape[0], dtype=np.int32)

    actual = edit_distance_csr(
        query,
        np.arange(len(candidates), dtype=np.int64),
        np.array(blob, dtype=np.uint8),
        bounds,
        identity,
    )
    expected = [weighted_edit_distance(analyze_reading("チクビ").phonemes, p) for p in candidates]
    assert actual == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    ("query", "candidate", "expected"),
    [
        # 完全一致は占有率 1.0。
        ("リンゴ", "リンゴ", 1.0),
        # 語頭・語末・両端の余分。
        ("リンゴ", "リンゴク", 5 / 7),
        ("リンゴ", "ゴリンゴ", 5 / 7),
        ("リンゴ", "アオリンゴ", 5 / 7),
        # 余分が多いほど占有率が下がる。
        ("リンゴ", "リンゴジュース", 5 / 10),
        # 特殊モーラの挿入は「完全な形」を壊さない。
        ("リンゴ", "リンゴー", 5 / 6),
        ("カガク", "カンガク", 6 / 7),
        # 特殊モーラ以外が挟まれば別の音。
        ("リンゴ", "リンボ", 0.0),
        ("リンゴ", "レンゴ", 0.0),
        ("リンゴ", "リンガゴ", 0.0),
        # 候補のほうが短ければ含みようがない。
        ("リンゴ", "リン", 0.0),
    ],
)
def test_containment_ratio(query: str, candidate: str, expected: float) -> None:
    """包含判定は連続一致 + 特殊モーラの伸縮だけを許す。

    編集距離では表現できない性質なので独立した成分にしてある
    (`search.ScoreWeights.containment`)。距離は挿入を一律に減点するため、
    クエリを丸ごと含む語が 1 音素だけ違う同じ長さの語に必ず負ける。
    """
    ratio = containment_ratio(analyze_reading(query).phonemes, analyze_reading(candidate).phonemes)
    assert ratio == pytest.approx(expected)


def test_containment_scan_matches_the_symbolic_reference() -> None:
    """Rust の走査と記号版の判定が一致する。**検索が通るのは Rust 側。**

    記号版 (`containment_ratio`) が基準で、走査はそれを索引全体に対して
    並列に回すもの。索引 ID から距離テーブル ID への写像を挟むので、
    そこがずれると包含が黙って成立しなくなる (加点が消えるだけで例外に
    ならないため、テストでしか捕まらない)。
    """
    words = [
        "リンゴ",
        "リンゴク",
        "ゴリンゴ",
        "リンゴー",
        "リンゴジュース",
        "リンボ",
        "レンゴ",
        "リン",
        "カンガク",
        "アオリンゴ",
    ]
    candidates = [analyze_reading(w).phonemes for w in words]
    query = analyze_reading("リンゴ").phonemes

    blob: list[int] = []
    bounds = np.zeros(len(candidates) + 1, dtype=np.int32)
    for row, phonemes in enumerate(candidates):
        blob.extend(int(i) for i in phoneme_ids(phonemes))
        bounds[row + 1] = len(blob)
    identity = np.arange(SUBSTITUTION_COSTS.shape[0], dtype=np.int32)

    groups, ratios = containment_scan(
        phoneme_ids(query),
        np.array(blob, dtype=np.uint8),
        bounds,
        identity,
        0,
        len(candidates),
    )

    actual = dict(zip(groups.tolist(), ratios.tolist(), strict=True))
    for row, phonemes in enumerate(candidates):
        expected = containment_ratio(query, phonemes)
        if expected > 0.0:
            assert actual[row] == pytest.approx(expected, abs=1e-6), words[row]
        else:
            assert row not in actual, words[row]


def test_containment_scan_matches_on_random_inputs() -> None:
    """ランダムな音素列でも両実装が一致する。

    手で選んだ対では特殊モーラの飛ばしが絡む経路 (途中で失敗して開始位置を
    やり直す場合) を網羅できない。長短を混ぜた乱数で突き合わせる。
    """
    rng = np.random.default_rng(20260813)
    symbols = sorted(PHONEME_TO_ID)

    for _ in range(200):
        query = tuple(rng.choice(symbols, size=int(rng.integers(1, 6))))
        count = int(rng.integers(1, 30))
        candidates = [
            tuple(rng.choice(symbols, size=int(rng.integers(1, 12)))) for _ in range(count)
        ]

        blob: list[int] = []
        bounds = np.zeros(count + 1, dtype=np.int32)
        for row, phonemes in enumerate(candidates):
            blob.extend(int(i) for i in phoneme_ids(phonemes))
            bounds[row + 1] = len(blob)
        identity = np.arange(SUBSTITUTION_COSTS.shape[0], dtype=np.int32)

        groups, ratios = containment_scan(
            phoneme_ids(query),
            np.array(blob, dtype=np.uint8),
            bounds,
            identity,
            0,
            count,
        )
        actual = dict(zip(groups.tolist(), ratios.tolist(), strict=True))
        for row, phonemes in enumerate(candidates):
            expected = containment_ratio(query, phonemes)
            if expected > 0.0:
                assert actual[row] == pytest.approx(expected, abs=1e-6), (query, phonemes)
            else:
                assert row not in actual, (query, phonemes)


def test_batch_handles_empty_query_and_no_candidates() -> None:
    empty = np.zeros(0, dtype=np.int32)
    phonemes = analyze_reading("チクビ").phonemes
    matrix = phoneme_ids(phonemes).reshape(1, -1)
    lengths = np.array([len(phonemes)], dtype=np.int64)

    # クエリが空なら候補を全削除するコスト。
    assert edit_distance_batch(empty, matrix, lengths) == pytest.approx(
        [weighted_edit_distance((), phonemes)]
    )
    # 候補が無ければ空の配列。
    assert edit_distance_batch(
        phoneme_ids(phonemes), np.zeros((0, 0), dtype=np.int64), np.zeros(0, dtype=np.int64)
    ).shape == (0,)


#: アライメントは編集距離の内訳を見せるためのものなので、対ごとのコストの
#: 総和が距離と一致しなければ表示が嘘になる。距離の重みを変えたときに
#: 両者が食い違わないことをここで担保する。
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("チクビ", "テクビ"),
        ("チクビ", "チョコビ"),
        ("カガク", "カカク"),
        ("サカナ", "アカ"),
        ("ラーメン", "ローメン"),
        ("マツタケ", "ソラ"),
        ("", "アイ"),
        ("ア", ""),
    ],
)
def test_alignment_costs_sum_to_edit_distance(a: str, b: str) -> None:
    pa = analyze_reading(a).phonemes
    pb = analyze_reading(b).phonemes
    pairs = align_phonemes(pa, pb)
    assert sum(cost for _, _, cost, _ in pairs) == pytest.approx(weighted_edit_distance(pa, pb))


def test_alignment_preserves_both_sequences() -> None:
    """対応付けから両方の音素列を復元できる (取りこぼし・重複がない)。"""
    pa = analyze_reading("チクビ").phonemes
    pb = analyze_reading("チョコビ").phonemes
    pairs = align_phonemes(pa, pb)
    assert tuple(x for x, _, _, _ in pairs if x is not None) == pa
    assert tuple(y for _, y, _, _ in pairs if y is not None) == pb


def test_alignment_labels_operations() -> None:
    pairs = align_phonemes(("ch", "i", "k", "u"), ("t", "i", "k", "u"))
    assert [op for _, _, _, op in pairs] == ["sub", "match", "match", "match"]


# --- IPA 表記 ---------------------------------------------------------------


def test_every_phoneme_has_an_ipa_entry() -> None:
    """IPA 表が全音素を覆う。

    `phoneme_ipa` は未知の記号をそのまま返すので、表から漏れた音素は例外にならず
    **表示だけが黙ってヘボン式のまま残る**。素性表に音素を足したときにここで
    気付けるよう、表のキーを直接見る (同じ字を当てる音素 k / a / v などがあるため、
    戻り値の比較では漏れを検出できない)。
    """
    missing = [symbol for symbol in _ALL_PHONEMES if symbol not in _IPA]
    assert not missing, f"IPA 表に無い音素: {missing}"


@pytest.mark.parametrize(
    ("reading", "expected"),
    [
        ("チクビ", "t͡ɕikɯbi"),
        ("テクビ", "tekɯbi"),
        # /u/ は円唇性が弱いので [ɯ]。素性表の rounded=False と対応する。
        ("スシ", "sɯɕi"),
        # /f/ は [ɸ] (両唇)。綴りに引かれて [f] にしないことの担保。
        ("フトン", "ɸɯtoɴ"),
        # ヴ は借用語の [v] で、こちらは唇歯音。
        ("ヴァイオリン", "vaioɾiɴ"),
        # 長音は長さ記号。
        ("トーキョー", "toːkʲoː"),
        # 拗音は口蓋化の補助記号 1 つで写す。
        ("キャク", "kʲakɯ"),
    ],
)
def test_ipa_transcription(reading: str, expected: str) -> None:
    assert ipa_transcription(analyze_reading(reading).phonemes) == expected


@pytest.mark.parametrize(
    ("reading", "expected"),
    [
        # 促音は後続子音の重複。単独記号の [ʔ] を語中に置くと実際には起きない
        # 声門閉鎖を書いたことになる。
        ("キッテ", "kitte"),
        ("ガッコウ", "ɡakkoɯ"),
        ("イッショ", "iɕɕo"),
        # 破擦音の重複は先頭の 1 文字だけを重ねる ([t͡ɕt͡ɕ] にはしない)。
        ("ハッチャク", "hatt͡ɕakɯ"),
        ("バッジ", "badd͡ʑi"),
        # 語末促音は後続が無いので声門閉鎖のまま残る。
        ("アッ", "aʔ"),
    ],
)
def test_geminate_becomes_doubled_consonant(reading: str, expected: str) -> None:
    assert ipa_transcription(analyze_reading(reading).phonemes) == expected


def test_ipa_transcription_of_empty_sequence() -> None:
    assert ipa_transcription(()) == ""


def test_unknown_phoneme_passes_through() -> None:
    """未知の記号は落とさずそのまま返す。表示が黙って欠けるのを避ける。"""
    assert phoneme_ipa("zzz") == "zzz"

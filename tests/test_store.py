"""索引の永続化のテスト。"""

from __future__ import annotations

from pathlib import Path

import jpr_distance
import numpy as np
import pytest

from jpr.build import embed_entries
from jpr.distance import PAD_ID
from jpr.index import Category, IndexEntry
from jpr.phonology import analyze_reading
from jpr.store import FORMAT_VERSION, PhoneticStore, write_store


def make_entry(surface: str, reading: str, **kwargs) -> IndexEntry:
    pronunciation = analyze_reading(reading)
    return IndexEntry(
        surface=surface,
        reading=reading,
        phonemes=pronunciation.phonemes,
        mora_count=pronunciation.mora_count,
        pos=kwargs.get("pos", "普通名詞"),
        category=kwargs.get("category", Category.COMMON),
        cost=kwargs.get("cost", 5000),
    )


@pytest.fixture
def roundtrip(tmp_path: Path) -> PhoneticStore:
    entries = [
        make_entry("乳首", "チクビ"),
        make_entry("チョコビ", "チョコビ", category=Category.PRODUCT, cost=15000),
        # 長さの異なる表層を混ぜて可変長エンコードを確かめる。
        make_entry("東京特許許可局", "トウキョウトッキョキョカキョク", cost=9000),
        make_entry("ラーメン", "ラーメン", cost=1400),
    ]
    write_store(tmp_path, entries, embed_entries(entries), dict_type="core")
    return PhoneticStore(tmp_path)


def test_metadata_roundtrip(roundtrip: PhoneticStore) -> None:
    assert roundtrip.meta.version == FORMAT_VERSION
    assert roundtrip.meta.count == 4
    assert roundtrip.meta.dict_type == "core"


def test_entries_roundtrip_exactly(roundtrip: PhoneticStore) -> None:
    """可変長エンコードを経ても全フィールドが復元される。"""
    entry = roundtrip.entry(3)
    assert entry.surface == "東京特許許可局"
    assert entry.reading == "トウキョウトッキョキョカキョク"
    assert entry.phonemes == analyze_reading(entry.reading).phonemes
    assert entry.cost == 9000
    assert entry.category is Category.COMMON


def test_all_rows_are_readable(roundtrip: PhoneticStore) -> None:
    """行は入力順ではなく (モーラ数, 音素列) 順で格納される。"""
    surfaces = [roundtrip.surface(row) for row in range(len(roundtrip))]
    assert surfaces == ["乳首", "チョコビ", "ラーメン", "東京特許許可局"]


def test_rows_are_sorted_by_mora_count(roundtrip: PhoneticStore) -> None:
    """モーラ数昇順の格納は `mora_range` のスライスが依存する不変条件。"""
    moras = roundtrip.mora_counts
    assert (np.diff(moras) >= 0).all()


def test_mora_range_selects_contiguous_rows(roundtrip: PhoneticStore) -> None:
    """`mora_range` の区間はマスクで選んだ場合と同じ行を指す。"""
    moras = roundtrip.mora_counts
    for low, high in [(None, None), (3, 3), (4, None), (None, 3), (2, 4), (99, 99)]:
        start, end = roundtrip.mora_range(low, high)
        mask = np.ones(moras.size, dtype=bool)
        if low is not None:
            mask &= moras >= low
        if high is not None:
            mask &= moras <= high
        assert np.array_equal(np.arange(start, end), np.flatnonzero(mask)), (low, high)


def test_group_ids_fold_identical_phonemes(tmp_path: Path) -> None:
    """同じ音素列の行は同じグループ ID を持ち、隣接して格納される。"""
    entries = [
        make_entry("科学", "カガク", cost=2000),
        make_entry("価格", "カカク", cost=2000),
        make_entry("下顎", "カガク", cost=8000),
        make_entry("化学", "カガク", cost=3000),
    ]
    write_store(tmp_path, entries, embed_entries(entries), dict_type="core")
    store = PhoneticStore(tmp_path)

    groups: dict[tuple[str, ...], set[int]] = {}
    for row in range(len(store)):
        groups.setdefault(store.phonemes(row), set()).add(int(store.group_ids[row]))
    # 音素列とグループが 1 対 1 に対応する。
    assert all(len(ids) == 1 for ids in groups.values())
    assert len({next(iter(ids)) for ids in groups.values()}) == len(groups)
    # 同じグループの行は隣接する (単調非減少)。
    assert (np.diff(store.group_ids) >= 0).all()


def test_vectors_are_stored_once_per_phoneme_group(tmp_path: Path) -> None:
    """ベクトルは語ではなく音素列グループごとに 1 行だけ持つ (v5)。

    埋め込みは音素列だけの関数なので同音異表記は同じベクトルになる。
    行ごとに持つと full の 202 万行のうち 146 万ぶんしか情報が無いのに
    全部を保存し、内積も重複して計算することになる。
    """
    entries = [
        make_entry("科学", "カガク", cost=2000),
        make_entry("化学", "カガク", cost=3000),
        make_entry("下顎", "カガク", cost=8000),
        make_entry("価格", "カカク", cost=2000),
    ]
    write_store(tmp_path, entries, embed_entries(entries), dict_type="core")
    store = PhoneticStore(tmp_path)

    # カガク 3 件 + カカク 1 件 -> 語は 4、グループは 2。
    assert len(store) == 4
    assert store.group_count == 2
    for space in store.meta.dims:
        assert store.vectors(space).shape[0] == store.group_count


def test_group_starts_expand_back_to_rows(tmp_path: Path) -> None:
    """`group_starts` の区間が、そのグループに属する行と一致する。

    候補生成はグループで Top-K を取ってからここで行へ展開するので
    (`search._expand_groups`)、この対応が崩れると別の語のスコアを配る。
    """
    entries = [
        make_entry("科学", "カガク", cost=2000),
        make_entry("化学", "カガク", cost=3000),
        make_entry("価格", "カカク", cost=2000),
        make_entry("空", "ソラ", cost=2000),
    ]
    write_store(tmp_path, entries, embed_entries(entries), dict_type="core")
    store = PhoneticStore(tmp_path)

    starts = store.group_starts
    assert starts.size == store.group_count + 1
    assert int(starts[-1]) == len(store)
    for group in range(store.group_count):
        rows = range(int(starts[group]), int(starts[group + 1]))
        assert rows, "空のグループは存在しない"
        # 区間の行がすべてそのグループに属し、音素列が揃う。
        assert {int(store.group_ids[row]) for row in rows} == {group}
        assert len({store.phonemes(row) for row in rows}) == 1


def test_phonemes_survive_the_group_indirection(roundtrip: PhoneticStore) -> None:
    """行から引いた音素列が、その語の読みを解析した結果と一致する。

    音素列はグループごとに 1 本しか持たないので (v5)、行 -> グループの
    写像を間違えると**別の語の音素列を黙って返す**。復号経路を実際に通す。
    """
    for row in range(len(roundtrip)):
        expected = analyze_reading(roundtrip.reading(row)).phonemes
        assert roundtrip.phonemes(row) == expected
        # ID 経路 (rerank が使う) も同じ音素列を指す。
        assert roundtrip.phoneme_id_array(row).size == len(expected)


def test_phoneme_id_matrix_matches_row_by_row(roundtrip: PhoneticStore) -> None:
    """まとめて引いた行列は、1 行ずつ引いたものとパディング以外で一致する。

    音素数の散らばる語 (3 音素の「チクビ」と 20 音素超の「東京特許許可局」)
    を混ぜているので、パディングの扱いを間違えればここで落ちる。
    """
    rows = np.arange(len(roundtrip))
    matrix, lengths = roundtrip.phoneme_id_matrix(rows)

    assert lengths.tolist() == [roundtrip.phoneme_id_array(row).size for row in rows]
    assert matrix.shape == (len(roundtrip), int(lengths.max()))
    for row in rows:
        expected = roundtrip.phoneme_id_array(row)
        assert matrix[row, : expected.size].tolist() == expected.tolist()
        # 余りはパディングで埋まっていて、実データと混ざらない。
        assert (matrix[row, expected.size :] == PAD_ID).all()


def test_phoneme_id_matrix_handles_empty_selection(roundtrip: PhoneticStore) -> None:
    matrix, lengths = roundtrip.phoneme_id_matrix(np.zeros(0, dtype=np.int64))
    assert matrix.shape[0] == 0
    assert lengths.size == 0


def test_category_ids_map_back(roundtrip: PhoneticStore) -> None:
    product = roundtrip.category_id(Category.PRODUCT)
    assert product >= 0
    assert roundtrip.category_ids[1] == product
    assert roundtrip.category_of(1) is Category.PRODUCT


def test_absent_category_returns_negative(roundtrip: PhoneticStore) -> None:
    """索引に無いカテゴリは -1 を返し、フィルタで安全に無視できる。"""
    assert roundtrip.category_id(Category.PERSON) == -1


def test_vectors_are_memory_mapped(roundtrip: PhoneticStore) -> None:
    vectors = roundtrip.vectors("phonetic")
    assert isinstance(vectors, np.memmap)
    # 行は語ではなく音素列グループ (v5)。
    assert vectors.shape == (roundtrip.group_count, roundtrip.meta.dims["phonetic"])


def test_no_ann_graph_is_written(roundtrip: PhoneticStore) -> None:
    """HNSW のグラフは書かない。

    hnswlib の `load_index` はグラフをヒープに実体化し、しかもベクトルを
    内部に複製するので、mmap したベクトル行列と同じデータを二重に持つ
    (`PhoneticSearcher._top_candidates` 参照)。候補生成は内積で足りる。
    """
    assert not list(roundtrip.path.glob("hnsw-*.bin"))


def test_inner_product_ranks_self_first(roundtrip: PhoneticStore) -> None:
    """自身のベクトルで引けば自身が最も近い。候補生成の土台の確認。

    量子化スケールを戻す経路 (`top_groups`) を通す。行列は int8 なので、
    生の内積を取ると尺度が復元されない。返るのは行ではなくグループ (v5)。
    """
    original = roundtrip.dequantized("phonetic")
    group = int(roundtrip.group_ids[1])
    groups, scores = roundtrip.top_groups("phonetic", original[group], roundtrip.group_count)
    assert int(groups[0]) == group
    # 正規化済みベクトルの自己内積なのでコサイン類似度は 1.0。
    assert scores[0] == pytest.approx(1.0, abs=0.01)


def test_vectors_are_quantized_to_int8(roundtrip: PhoneticStore) -> None:
    """ベクトルは int8 で保存される (float32 の 1/4)。"""
    assert roundtrip.vectors("phonetic").dtype == np.int8
    assert roundtrip.scale("phonetic") > 0.0


def test_quantized_dot_matches_float32(roundtrip: PhoneticStore) -> None:
    """量子化した内積が元の float32 の内積と一致する (誤差の範囲で)。

    候補生成 (`top_groups`)・全走査 (`dot_groups`)・選抜
    (`dot_selected_groups`) の 3 経路が同じ尺度を返すことを確かめる。
    ここがずれると rerank のスコア合成が静かに狂う。
    """
    original = roundtrip.dequantized("phonetic")
    query = original[0]
    reference = original @ query

    scanned = roundtrip.dot_groups("phonetic", query, 0, roundtrip.group_count)
    assert scanned == pytest.approx(reference, abs=0.02)

    picked = np.array([0, roundtrip.group_count - 1])
    selected = roundtrip.dot_selected_groups("phonetic", query, picked)
    assert selected == pytest.approx(reference[picked], abs=0.02)

    groups, scores = roundtrip.top_groups("phonetic", query, roundtrip.group_count)
    assert scores == pytest.approx(reference[groups], abs=0.02)


def test_tied_scores_resolve_to_row_order() -> None:
    """同点の候補は行番号の小さい順に返る。

    候補生成を並列にすると、同点がどれも選ばれうるので「どれが k 件に
    残るか」が揺れる。代表選び (`search._representative_rank`) は候補の
    到着順に依存するので、揺れると**同じクエリが違う表記を返す**。

    スコアだけを見る比較関数では足りない。チャンク内の選抜は入力順を
    保証しないので (`select_nth_unstable`)、最後に並べ替えても生き残った
    中の順序が揃うだけになる — 実装当初はここで全件同点の 5000 行から
    行 0, 1 を飛ばして 2852 が入っていた。チャンク境界 (262144 行) を
    跨ぐ規模で確かめる。
    """
    matrix = np.ones((900000, 8), dtype=np.int8)
    query = np.ones(8, dtype=np.int8)

    for _ in range(5):
        rows, _ = jpr_distance.top_candidates(matrix, query, 20, 1.0)
        assert list(rows) == list(range(20))

    # 一部だけ高スコアにしても、残りは行番号順で埋まる。
    matrix[7] = 2
    matrix[500000] = 2
    for _ in range(5):
        rows, _ = jpr_distance.top_candidates(matrix, query, 5, 1.0)
        assert list(rows) == [7, 500000, 0, 1, 2]


def test_bounds_are_int32(roundtrip: PhoneticStore) -> None:
    """CSR の境界は int32。full でも blob は 3900 万バイトしかない。

    `edit_distance_csr` が int32 を受け取る前提でもあり、int64 に戻すと
    呼び出しごとに 16MB の変換コピーが走る。
    """
    for name in ("surface_bounds", "reading_bounds", "phoneme_bounds"):
        array = np.load(roundtrip.path / f"{name}.npy", allow_pickle=False)
        assert array.dtype == np.int32, name


def test_missing_index_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="索引が見つかりません"):
        PhoneticStore(tmp_path / "nonexistent")


def test_version_mismatch_raises(roundtrip: PhoneticStore) -> None:
    meta_file = roundtrip.path / "meta.json"
    meta_file.write_text(
        meta_file.read_text(encoding="utf-8").replace(
            f'"version": {FORMAT_VERSION}', '"version": 0'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="索引の形式が古い"):
        PhoneticStore(roundtrip.path)


def test_string_encoding_is_compact(tmp_path: Path) -> None:
    """固定幅 unicode 配列を使わないことで容量が抑えられている。

    NumPy の `<U` 配列は最長要素の幅で全行を埋めるため、長さの散らばる表層では
    実データの数倍を消費する。可変長で詰めていることを実測で確かめる。
    """
    # 1 件だけ極端に長い表層を混ぜる。固定幅ならこの幅が全行に波及する。
    entries = [make_entry("あ" * 40, "ア" * 40)] + [
        make_entry(f"語{i}", "アイ") for i in range(500)
    ]
    write_store(tmp_path, entries, embed_entries(entries), dict_type="core")

    surface_bytes = np.load(tmp_path / "surface_blob.npy", allow_pickle=False).nbytes

    # 固定幅なら 40 文字 × 4 バイト × 501 行 = 80KB 程度になる。
    fixed_width_estimate = 40 * 4 * len(entries)
    assert surface_bytes < fixed_width_estimate / 5

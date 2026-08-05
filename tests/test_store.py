"""索引の永続化のテスト。"""

from __future__ import annotations

from pathlib import Path

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
    assert vectors.shape == (4, roundtrip.meta.dims["phonetic"])


def test_ann_index_exists_for_phonetic_space(roundtrip: PhoneticStore) -> None:
    assert roundtrip.has_ann("phonetic")


def test_ann_query_returns_self_first(roundtrip: PhoneticStore) -> None:
    """自身のベクトルで引けば自身が最も近い。ANN が正しく張れている確認。"""
    index = roundtrip.ann("phonetic", ef=16)
    query = np.asarray(roundtrip.vectors("phonetic")[1]).reshape(1, -1)
    labels, _ = index.knn_query(query, k=1)
    assert int(labels[0][0]) == 1


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

    with np.load(tmp_path / "entries.npz", allow_pickle=False) as archive:
        surface_bytes = archive["surface_blob"].nbytes

    # 固定幅なら 40 文字 × 4 バイト × 501 行 = 80KB 程度になる。
    fixed_width_estimate = 40 * 4 * len(entries)
    assert surface_bytes < fixed_width_estimate / 5

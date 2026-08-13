"""テスト共通のフィクスチャ。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jpr.index import Category, IndexEntry
from jpr.phonology import analyze_reading
from jpr.search import PhoneticSearcher
from jpr.store import PhoneticStore, default_store_path, write_store

# 検索の挙動を検証するための小さな語彙。実際の辞書を引かずに済ませる。
# (表層, 読み, 品詞, カテゴリ, コスト)
_SAMPLE_VOCABULARY = [
    ("チョコビ", "チョコビ", "固有名詞", Category.PRODUCT, 15000),
    ("チョコボール", "チョコボール", "固有名詞", Category.PRODUCT, 8000),
    ("チョコパイ", "チョコパイ", "固有名詞", Category.PRODUCT, 8000),
    ("地球儀", "チキュウギ", "普通名詞", Category.COMMON, 7000),
    ("手首", "テクビ", "普通名詞", Category.COMMON, 4000),
    ("仕組み", "シクミ", "普通名詞", Category.COMMON, 4500),
    ("竹輪", "チクワ", "普通名詞", Category.COMMON, 6000),
    ("ラーメン", "ラーメン", "普通名詞", Category.COMMON, 1400),
    ("ローメン", "ローメン", "普通名詞", Category.COMMON, 9000),
    ("電車", "デンシャ", "普通名詞", Category.COMMON, 3000),
    ("松茸", "マツタケ", "普通名詞", Category.COMMON, 6500),
    ("空", "ソラ", "普通名詞", Category.COMMON, 3000),
    ("科学", "カガク", "普通名詞", Category.COMMON, 2000),
    ("価格", "カカク", "普通名詞", Category.COMMON, 2000),
    ("東京", "トウキョウ", "固有名詞", Category.PLACE, 2800),
    ("田中", "タナカ", "固有名詞", Category.PERSON, 3000),
    # 同音異表記。重複排除の検証に使う。
    ("仕組", "シクミ", "普通名詞", Category.COMMON, 5000),
    ("し組み", "シクミ", "一般", Category.COMMON, 12000),
]


def _sample_entries() -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    for surface, reading, pos, category, cost in _SAMPLE_VOCABULARY:
        pronunciation = analyze_reading(reading)
        entries.append(
            IndexEntry(
                surface=surface,
                reading=reading,
                phonemes=pronunciation.phonemes,
                mora_count=pronunciation.mora_count,
                pos=pos,
                category=category,
                cost=cost,
            )
        )
    return entries


@pytest.fixture(scope="session")
def sample_store(tmp_path_factory: pytest.TempPathFactory) -> PhoneticStore:
    """小さな語彙で作った索引。"""
    from jpr.build import embed_entries

    path = tmp_path_factory.mktemp("index")
    entries = _sample_entries()
    write_store(path, entries, embed_entries(entries))
    return PhoneticStore(path)


@pytest.fixture(scope="session")
def sample_searcher(sample_store: PhoneticStore) -> PhoneticSearcher:
    return PhoneticSearcher(sample_store)


@pytest.fixture(scope="session")
def real_store() -> PhoneticStore:
    """実際の辞書から作った索引。無ければテストをスキップする。"""
    path = default_store_path()
    if not (Path(path) / "meta.json").exists():
        pytest.skip("実辞書の索引が未構築 (`jpr build-index` で作成)")
    return PhoneticStore(path)


@pytest.fixture(scope="session")
def real_searcher(real_store: PhoneticStore) -> PhoneticSearcher:
    return PhoneticSearcher(real_store)

"""索引構築パイプライン (オフライン処理)。

SudachiDict system.dic
    -> 語彙 (表層 + 読み + 品詞 + コスト)
    -> 音素列 / モーラ列
    -> 各空間の phonetic embedding
    -> NumPy 行列 + HNSW 索引
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .dictionary import SystemDictionary, find_system_dic
from .embedding import SPACES, embed
from .index import (
    EXCLUDED_POS,
    IndexEntry,
    classify,
    has_pronounceable_reading,
    is_searchable_surface,
)
from .phonology import analyze_reading, to_katakana
from .store import write_store

ProgressCallback = Callable[[str], None] | None


def collect_entries(
    *,
    dict_type: str = "full",
    min_mora: int = 2,
    max_mora: int = 12,
    progress: ProgressCallback = None,
) -> list[IndexEntry]:
    """辞書から索引対象の語彙を集める。"""
    dictionary = SystemDictionary(find_system_dic(dict_type))
    if progress:
        progress(f"辞書を走査中 ({len(dictionary):,} 語)")

    entries: list[IndexEntry] = []
    seen: set[tuple[str, str]] = set()

    for count, entry in enumerate(dictionary.entries(), start=1):
        if progress and count % 500_000 == 0:
            progress(f"  {count:,} / {len(dictionary):,} 語")

        if entry.pos and entry.pos[0] in EXCLUDED_POS:
            continue
        if not is_searchable_surface(entry.surface):
            continue

        reading = to_katakana(entry.reading)
        if not has_pronounceable_reading(reading):
            continue

        key = (entry.surface, reading)
        if key in seen:
            continue
        seen.add(key)

        pronunciation = analyze_reading(reading)
        if not min_mora <= pronunciation.mora_count <= max_mora:
            continue

        entries.append(
            IndexEntry(
                surface=entry.surface,
                reading=reading,
                phonemes=pronunciation.phonemes,
                mora_count=pronunciation.mora_count,
                pos=entry.pos[1] if len(entry.pos) > 1 else (entry.pos[0] if entry.pos else ""),
                category=classify(entry.pos),
                cost=entry.cost,
            )
        )

    if progress:
        progress(f"索引対象 {len(entries):,} 語")
    return entries


def embed_entries(
    entries: list[IndexEntry],
    *,
    progress: ProgressCallback = None,
) -> dict[str, np.ndarray]:
    """全語彙をすべての空間で埋め込む。"""
    count = len(entries)
    matrices = {name: np.zeros((count, dim), dtype=np.float32) for name, dim in SPACES.items()}

    if progress:
        progress(f"埋め込みを計算中 ({count:,} 語 × {len(SPACES)} 空間)")

    for row, entry in enumerate(entries):
        if progress and row and row % 500_000 == 0:
            progress(f"  {row:,} / {count:,} 語")
        # 索引には音素列しか持たないが、母音骨格と語尾はモーラ境界を要するため
        # 読みから作り直す。
        pronunciation = analyze_reading(entry.reading)
        for name, vector in embed(pronunciation).items():
            matrices[name][row] = vector

    return matrices


def build_index(
    path: Path | str,
    *,
    dict_type: str = "full",
    min_mora: int = 2,
    max_mora: int = 12,
    progress: ProgressCallback = None,
) -> int:
    """索引を構築してディスクに書く。戻り値は索引した語数。"""
    entries = collect_entries(
        dict_type=dict_type, min_mora=min_mora, max_mora=max_mora, progress=progress
    )
    vectors = embed_entries(entries, progress=progress)
    write_store(
        path,
        entries,
        vectors,
        dict_type=dict_type,
        progress=progress,
    )
    return len(entries)


__all__ = ["build_index", "collect_entries", "embed_entries"]

"""音韻インデックスの永続化 (Phonetic RAG のオフライン側)。

保存するもの:

    meta.json          形式バージョンと件数、空間の次元
    entries.npz        語彙のメタデータ (表層・読み・音素列・カテゴリ・コスト)
    vectors-<空間>.npy 各埋め込み空間の (N, D) 行列
    hnsw-<空間>.bin    ANN 索引

pickle をやめた理由は 2 つ。ロードに 12〜18 秒かかり MCP サーバの起動コスト
として重すぎたこと、そして任意コード実行を招く形式を配布物に使いたくないこと。
NumPy 配列は mmap で開けるので実質 0 秒で立ち上がる。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embedding import SPACES
from .index import Category, IndexEntry

FORMAT_VERSION = 2

#: ANN 索引を作る空間。
#:
#: 候補生成は原則 phonetic 1 本で足りる。全空間に HNSW を張ると 1 空間あたり
#: 1.2GB (202 万語) かかる一方、rerank 側は行を直接引くだけなのでベクトル行列が
#: あれば済む。韻の検索で語尾から候補を引きたい場合のみ coda を足す。
INNER_PRODUCT_SPACES = ("phonetic", "coda")

_META_FILE = "meta.json"
_ENTRIES_FILE = "entries.npz"


def default_store_path() -> Path:
    """既定の索引ディレクトリ。"""
    return Path.home() / ".cache" / "jpr" / "index"


@dataclass(frozen=True)
class StoreMeta:
    version: int
    count: int
    dims: dict[str, int]
    dict_type: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "count": self.count,
                "dims": self.dims,
                "dict_type": self.dict_type,
            },
            ensure_ascii=False,
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> StoreMeta:
        payload = json.loads(text)
        return cls(
            version=payload["version"],
            count=payload["count"],
            dims=payload["dims"],
            dict_type=payload.get("dict_type", "full"),
        )


def _encode_strings(values: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """文字列列を UTF-8 バイト列と境界インデックスに落とす。

    NumPy の固定幅 unicode 配列 (`<U27`) は最長要素の幅で全行を埋めるため、
    表層のように長さの散らばる列では実データの数十倍を消費する
    (実測 202 万語で 218MB)。可変長で詰めれば 1/5 以下になる。
    """
    blob = bytearray()
    boundaries = np.zeros(len(values) + 1, dtype=np.int64)
    for row, value in enumerate(values):
        blob.extend(value.encode("utf-8"))
        boundaries[row + 1] = len(blob)
    return np.frombuffer(bytes(blob), dtype=np.uint8), boundaries


def _decode_string(blob: np.ndarray, boundaries: np.ndarray, row: int) -> str:
    start, end = int(boundaries[row]), int(boundaries[row + 1])
    return bytes(blob[start:end]).decode("utf-8")


def _encode_entries(entries: Sequence[IndexEntry]) -> dict[str, np.ndarray]:
    """語彙メタデータを NumPy 配列に落とす。

    可変長のもの (表層・読み・音素列) は連結した 1 本の配列と各語の終端位置で
    表す (CSR のような持ち方)。音素は記号ではなく ID で持つ。
    """
    surface_blob, surface_bounds = _encode_strings([e.surface for e in entries])
    reading_blob, reading_bounds = _encode_strings([e.reading for e in entries])

    # 品詞とカテゴリは値の種類が少ないので、辞書 + ID で持つ。
    pos_values = sorted({e.pos for e in entries})
    pos_ids = {value: index for index, value in enumerate(pos_values)}
    category_values = sorted({e.category.value for e in entries})
    category_ids = {value: index for index, value in enumerate(category_values)}

    # 音素列も同様に ID 化して連結する。
    phoneme_ids: list[int] = []
    phoneme_bounds = np.zeros(len(entries) + 1, dtype=np.int64)
    symbols: dict[str, int] = {}
    for row, entry in enumerate(entries):
        for symbol in entry.phonemes:
            phoneme_ids.append(symbols.setdefault(symbol, len(symbols)))
        phoneme_bounds[row + 1] = len(phoneme_ids)

    phoneme_vocabulary = [""] * len(symbols)
    for symbol, index in symbols.items():
        phoneme_vocabulary[index] = symbol

    return {
        "surface_blob": surface_blob,
        "surface_bounds": surface_bounds,
        "reading_blob": reading_blob,
        "reading_bounds": reading_bounds,
        "pos_vocabulary": np.array(pos_values, dtype=np.str_),
        "pos_ids": np.array([pos_ids[e.pos] for e in entries], dtype=np.int16),
        "category_vocabulary": np.array(category_values, dtype=np.str_),
        "category_ids": np.array(
            [category_ids[e.category.value] for e in entries], dtype=np.int8
        ),
        "costs": np.array([e.cost for e in entries], dtype=np.int32),
        "mora_counts": np.array([e.mora_count for e in entries], dtype=np.int16),
        "phoneme_vocabulary": np.array(phoneme_vocabulary, dtype=np.str_),
        "phoneme_ids": np.array(phoneme_ids, dtype=np.uint8),
        "phoneme_bounds": phoneme_bounds,
    }


class PhoneticStore:
    """ディスク上の音韻インデックスを読むための入口。

    語彙メタデータは mmap で開き、必要になった行だけ Python オブジェクトに
    起こす。200 万件すべてを起動時に materialize しない。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        meta_file = self.path / _META_FILE
        if not meta_file.exists():
            raise FileNotFoundError(
                f"索引が見つかりません: {self.path}\n"
                "`jpr build-index` で構築してください。"
            )
        self.meta = StoreMeta.from_json(meta_file.read_text(encoding="utf-8"))
        if self.meta.version != FORMAT_VERSION:
            raise ValueError(
                f"索引の形式が古いです (version={self.meta.version}, "
                f"期待={FORMAT_VERSION})。`jpr build-index --force` で再構築してください。"
            )

        with np.load(self.path / _ENTRIES_FILE, allow_pickle=False) as archive:
            self._data = {name: archive[name] for name in archive.files}

        self._pos_vocabulary = [str(v) for v in self._data["pos_vocabulary"]]
        self._category_vocabulary = [
            Category(str(v)) for v in self._data["category_vocabulary"]
        ]
        self._phoneme_vocabulary = [str(v) for v in self._data["phoneme_vocabulary"]]

        self._vectors: dict[str, np.ndarray] = {}
        self._ann: dict[str, object] = {}

    def __len__(self) -> int:
        return self.meta.count

    # --- 語彙メタデータ ---------------------------------------------------

    def surface(self, row: int) -> str:
        return _decode_string(
            self._data["surface_blob"], self._data["surface_bounds"], row
        )

    def reading(self, row: int) -> str:
        return _decode_string(
            self._data["reading_blob"], self._data["reading_bounds"], row
        )

    def phonemes(self, row: int) -> tuple[str, ...]:
        bounds = self._data["phoneme_bounds"]
        start, end = int(bounds[row]), int(bounds[row + 1])
        vocabulary = self._phoneme_vocabulary
        return tuple(vocabulary[i] for i in self._data["phoneme_ids"][start:end])

    def entry(self, row: int) -> IndexEntry:
        return IndexEntry(
            surface=self.surface(row),
            reading=self.reading(row),
            phonemes=self.phonemes(row),
            mora_count=int(self._data["mora_counts"][row]),
            pos=self._pos_vocabulary[int(self._data["pos_ids"][row])],
            category=self._category_vocabulary[int(self._data["category_ids"][row])],
            cost=int(self._data["costs"][row]),
        )

    def category_of(self, row: int) -> Category:
        return self._category_vocabulary[int(self._data["category_ids"][row])]

    @property
    def category_ids(self) -> np.ndarray:
        """行ごとのカテゴリ ID。フィルタをベクトル化するのに使う。"""
        return self._data["category_ids"]

    def category_id(self, category: Category) -> int:
        """カテゴリに対応する ID。索引に存在しない場合は -1。"""
        try:
            return self._category_vocabulary.index(category)
        except ValueError:
            return -1

    @property
    def costs(self) -> np.ndarray:
        return self._data["costs"]

    @property
    def mora_counts(self) -> np.ndarray:
        return self._data["mora_counts"]

    # --- ベクトルと ANN ---------------------------------------------------

    def vectors(self, space: str) -> np.ndarray:
        """指定した空間の (N, D) 行列を mmap で返す。"""
        if space not in self._vectors:
            self._vectors[space] = np.load(
                self.path / f"vectors-{space}.npy", mmap_mode="r"
            )
        return self._vectors[space]

    def ann(self, space: str, ef: int = 200):
        """指定した空間の HNSW 索引を返す。

        hnswlib のインポートは索引を実際に引くときまで遅らせる。
        """
        if space not in self._ann:
            import hnswlib

            dim = self.meta.dims[space]
            index = hnswlib.Index(space="ip", dim=dim)
            index.load_index(str(self.path / f"hnsw-{space}.bin"), max_elements=self.meta.count)
            self._ann[space] = index
        index = self._ann[space]
        index.set_ef(ef)  # type: ignore[attr-defined]
        return index

    def has_ann(self, space: str) -> bool:
        return (self.path / f"hnsw-{space}.bin").exists()


def write_store(
    path: Path | str,
    entries: Sequence[IndexEntry],
    vectors: dict[str, np.ndarray],
    *,
    dict_type: str = "full",
    ann_spaces: Iterable[str] = INNER_PRODUCT_SPACES,
    ef_construction: int = 100,
    m: int = 24,
    progress: object = None,
) -> None:
    """索引をディスクに書く。ANN 索引の構築もここで行う。"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    def report(message: str) -> None:
        if callable(progress):
            progress(message)

    report("語彙メタデータを書き出し中")
    np.savez(path / _ENTRIES_FILE, **_encode_entries(entries))

    for space, matrix in vectors.items():
        report(f"ベクトルを書き出し中: {space}")
        np.save(path / f"vectors-{space}.npy", matrix)

    import hnswlib

    for space in ann_spaces:
        if space not in vectors:
            continue
        matrix = vectors[space]
        report(f"ANN 索引を構築中: {space} ({matrix.shape[0]:,} 件)")
        index = hnswlib.Index(space="ip", dim=matrix.shape[1])
        index.init_index(max_elements=matrix.shape[0], ef_construction=ef_construction, M=m)
        index.add_items(matrix, np.arange(matrix.shape[0]))
        index.save_index(str(path / f"hnsw-{space}.bin"))

    meta = StoreMeta(
        version=FORMAT_VERSION,
        count=len(entries),
        dims={name: int(vectors[name].shape[1]) for name in vectors},
        dict_type=dict_type,
    )
    (path / _META_FILE).write_text(meta.to_json(), encoding="utf-8")
    report("完了")


__all__ = [
    "FORMAT_VERSION",
    "INNER_PRODUCT_SPACES",
    "SPACES",
    "PhoneticStore",
    "StoreMeta",
    "default_store_path",
    "write_store",
]

"""音韻インデックスの永続化 (Phonetic RAG のオフライン側)。

保存するもの:

    meta.json          形式バージョンと件数、空間の次元
    entries.npz        語彙のメタデータ (表層・読み・音素列・カテゴリ・コスト)
    vectors-<空間>.npy 各埋め込み空間の (N, D) 行列

pickle をやめた理由は 2 つ。ロードに 12〜18 秒かかり MCP サーバの起動コスト
として重すぎたこと、そして任意コード実行を招く形式を配布物に使いたくないこと。
NumPy 配列は mmap で開けるので実質 0 秒で立ち上がる。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jpr_distance as _rust
import numpy as np

from .distance import PAD_ID, PHONEME_TO_ID, UNKNOWN_PHONEME_ID
from .embedding import SPACES
from .index import Category, IndexEntry

#: 索引形式のバージョン。エントリの持ち方や行の並びを変えたら上げる。
#:
#: ANN のグラフを作らなくなったときは**上げていない**。読み込み側は
#: `hnsw-*.bin` を参照しないので、グラフが残っている古い索引もそのまま動く
#: (無駄なファイルが残るだけ)。バージョンを上げると再構築を強いることになり、
#: full なら 5.5 分かかる — 動くものを動かなくする理由がない。
#:
#: v4 でベクトルを int8 に量子化した (`_quantize`)。float32 の索引は読めない
#: ので再構築が要る。
FORMAT_VERSION = 4

#: 候補生成に使う空間。
#:
#: **phonetic 1 本だけ。** `PhoneticSearcher.candidate_space` (既定 "phonetic") が
#: 索引全体と内積を取る空間で、他の空間は rerank でスコアを足すだけ。
#: どちらもベクトル行列を mmap で読むので、空間を増やしてもディスクだけが
#: 増える (匿名メモリは増えない)。
CANDIDATE_SPACES = ("phonetic",)

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
    #: 空間ごとの量子化スケール。int8 の値にこれを掛けると元の float32 に戻る。
    scales: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "count": self.count,
                "dims": self.dims,
                "dict_type": self.dict_type,
                "scales": self.scales,
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
            scales=payload["scales"],
        )


def _quantize(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """(N, D) の float32 行列を int8 とスケールに落とす。

    **スケールは空間ごとに 1 つ。** 行ごとに持つ案も測ったが、`phonetic` は
    L2 正規化済みで行の最大値が 0.24〜0.49 に収まるため再構成誤差が変わらない
    (どちらも 0.0026)。行ごとのスケール配列を持つ複雑さに見合わない。
    `rhythm` だけは正規化しないので絶対量が大きいが (ノルム最大 2.24)、
    空間ごとに最大値を取るこの形なら自動的に追従する。

    量子化の誤差は内積で最大 0.012 (全 5 空間の実測)。順位を分ける
    スコア差は Top-K 境界でも 0.0002 程度あるので、この誤差は候補生成の
    順位をわずかに揺らす。`search.DEFAULT_CANDIDATES` を広げて吸収する
    (`_top_candidates` の項を参照)。
    """
    peak = float(np.abs(matrix).max())
    if peak == 0.0:
        return np.zeros(matrix.shape, dtype=np.int8), 1.0
    scale = peak / 127.0
    quantized = np.round(matrix / scale).astype(np.int8)
    return quantized, scale


#: 境界インデックスに int32 を使うので、blob はこの長さを超えられない。
_MAX_BLOB_BYTES = 2**31 - 1


def _check_blob_fits(name: str, size: int) -> None:
    """blob が int32 の境界で表せることを確かめる。

    full 辞書の実測は表層 2900 万 / 読み 3900 万 / 音素 2200 万バイトなので
    2 桁の余裕がある。**それでも黙って壊れるより落ちるほうがいい** —
    溢れると境界が負に回り、索引全体が静かに誤った文字列を返す。
    """
    if size > _MAX_BLOB_BYTES:
        raise ValueError(
            f"{name} が int32 の境界で表せる上限を超えています ({size} > {_MAX_BLOB_BYTES})。"
            "境界配列の dtype を int64 に戻す必要があります。"
        )


def _encode_strings(values: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """文字列列を UTF-8 バイト列と境界インデックスに落とす。

    NumPy の固定幅 unicode 配列 (`<U27`) は最長要素の幅で全行を埋めるため、
    表層のように長さの散らばる列では実データの数十倍を消費する
    (実測 202 万語で 218MB)。可変長で詰めれば 1/5 以下になる。

    境界は int32。full 辞書でも読みの blob が 3900 万バイトしかないので
    int64 の幅が要らない (3 本の境界配列で 24MB の差)。**2^31 を超える
    blob は表現できない** ので、`_check_blob_fits` が構築時に検証する。
    """
    blob = bytearray()
    offsets = [0]
    for value in values:
        blob.extend(value.encode("utf-8"))
        offsets.append(len(blob))
    # 境界を int32 の配列に落とす前に検証する。溢れてから確かめても、
    # そのときには既に値が負へ回っている。
    _check_blob_fits("文字列 blob", len(blob))
    boundaries = np.asarray(offsets, dtype=np.int32)
    return np.frombuffer(bytes(blob), dtype=np.uint8), boundaries


def _decode_string(blob: np.ndarray, boundaries: np.ndarray, row: int) -> str:
    start, end = int(boundaries[row]), int(boundaries[row + 1])
    return bytes(blob[start:end]).decode("utf-8")


def _locality_order(entries: Sequence[IndexEntry]) -> list[int]:
    """行の格納順。(モーラ数, 音素列) の昇順に並べる。

    モーラ数で揃えるのは、モーラ範囲の全走査 (`search.py` の `_scan_candidates`)
    を連続スライスにするため。散らばった行のマスク + fancy indexing は行数に
    対してコストが急伸する一方 (97 万行の選抜で 191ms)、連続読みなら範囲の
    行数ぶんの帯域で済む。

    音素列を第 2 キーにするのは、同じ音素列 (同音異表記) を隣接させるため。
    rerank の編集距離は音素列が同じなら同じ値になるので、隣接していれば
    先頭行だけ計算して残りへ配れる (`search.py` の `_group_representatives`)。
    4〜6 モーラ帯の候補 53 万件のうちユニークな音素列は 58.8% しかない。

    stable sort なので同一キー内は入力順が保たれ、構築が決定的になる。
    """
    return sorted(range(len(entries)), key=lambda i: (entries[i].mora_count, entries[i].phonemes))


def _encode_entries(entries: Sequence[IndexEntry]) -> dict[str, np.ndarray]:
    """語彙メタデータを NumPy 配列に落とす。

    可変長のもの (表層・読み・音素列) は連結した 1 本の配列と各語の終端位置で
    表す (CSR のような持ち方)。音素は記号ではなく ID で持つ。

    entries は `_locality_order` で並んでいる前提。`group_ids` (同じ音素列の
    連番) は隣接比較で振るので、並んでいなければ同音異表記が別グループに散る。
    """
    surface_blob, surface_bounds = _encode_strings([e.surface for e in entries])
    reading_blob, reading_bounds = _encode_strings([e.reading for e in entries])

    # 品詞とカテゴリは値の種類が少ないので、辞書 + ID で持つ。
    pos_values = sorted({e.pos for e in entries})
    pos_ids = {value: index for index, value in enumerate(pos_values)}
    category_values = sorted({e.category.value for e in entries})
    category_ids = {value: index for index, value in enumerate(category_values)}

    # 音素列も同様に ID 化して連結する。同時に、同じ (モーラ数, 音素列) が
    # 続く区間へ同じグループ ID を振る (ソート済みなので隣接比較で足りる)。
    phoneme_ids: list[int] = []
    phoneme_offsets = [0]
    group_ids = np.zeros(len(entries), dtype=np.int32)
    symbols: dict[str, int] = {}
    group = -1
    previous_key: tuple[int, tuple[str, ...]] | None = None
    for row, entry in enumerate(entries):
        for symbol in entry.phonemes:
            phoneme_ids.append(symbols.setdefault(symbol, len(symbols)))
        phoneme_offsets.append(len(phoneme_ids))
        key = (entry.mora_count, entry.phonemes)
        if key != previous_key:
            group += 1
            previous_key = key
        group_ids[row] = group

    _check_blob_fits("音素 blob", len(phoneme_ids))

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
        "category_ids": np.array([category_ids[e.category.value] for e in entries], dtype=np.int8),
        "costs": np.array([e.cost for e in entries], dtype=np.int32),
        "mora_counts": np.array([e.mora_count for e in entries], dtype=np.int16),
        "phoneme_vocabulary": np.array(phoneme_vocabulary, dtype=np.str_),
        "phoneme_ids": np.array(phoneme_ids, dtype=np.uint8),
        "phoneme_bounds": np.asarray(phoneme_offsets, dtype=np.int32),
        "group_ids": group_ids,
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
                f"索引が見つかりません: {self.path}\n`jpr build-index` で構築してください。"
            )
        self.meta = StoreMeta.from_json(meta_file.read_text(encoding="utf-8"))
        if self.meta.version != FORMAT_VERSION:
            raise ValueError(
                f"索引の形式が古いです (version={self.meta.version}, "
                f"期待={FORMAT_VERSION})。`jpr build-index --force` で再構築してください。"
            )

        with np.load(self.path / _ENTRIES_FILE, allow_pickle=False) as archive:
            self._data = {name: archive[name] for name in archive.files}

        # 行がモーラ数順であることは v3 の不変条件で、モーラ範囲の連続スライス
        # (`mora_range`) がこれに依存する。壊れた索引が黙って誤った母集団を
        # 返すより、開いた時点で落ちるほうがいい。検証は 2ms 程度。
        moras = self._data["mora_counts"]
        if moras.size > 1 and np.any(np.diff(moras) < 0):
            raise ValueError(
                f"索引の行がモーラ数順に並んでいません: {self.path}\n"
                "`jpr build-index --force` で再構築してください。"
            )

        self._pos_vocabulary = [str(v) for v in self._data["pos_vocabulary"]]
        self._category_vocabulary = [Category(str(v)) for v in self._data["category_vocabulary"]]
        self._phoneme_vocabulary = [str(v) for v in self._data["phoneme_vocabulary"]]
        # 索引内の音素 ID から距離テーブルの ID への写像。索引の語彙順は構築時に
        # 出現した順なので、距離テーブルの順番とは一致しない。
        self._distance_ids = np.array(
            [PHONEME_TO_ID.get(symbol, UNKNOWN_PHONEME_ID) for symbol in self._phoneme_vocabulary],
            dtype=np.int32,
        )

        self._vectors: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.meta.count

    # --- 語彙メタデータ ---------------------------------------------------

    def surface(self, row: int) -> str:
        return _decode_string(self._data["surface_blob"], self._data["surface_bounds"], row)

    def reading(self, row: int) -> str:
        return _decode_string(self._data["reading_blob"], self._data["reading_bounds"], row)

    def phonemes(self, row: int) -> tuple[str, ...]:
        bounds = self._data["phoneme_bounds"]
        start, end = int(bounds[row]), int(bounds[row + 1])
        vocabulary = self._phoneme_vocabulary
        return tuple(vocabulary[i] for i in self._data["phoneme_ids"][start:end])

    def phoneme_id_array(self, row: int) -> np.ndarray:
        """音素列を距離計算用の ID 配列として返す。

        索引が持つ ID は語彙内で振ったものなので、距離テーブルの ID に写し直す。
        記号のタプルを経由しないので rerank の内側で使える。
        """
        bounds = self._data["phoneme_bounds"]
        start, end = int(bounds[row]), int(bounds[row + 1])
        return self._distance_ids[self._data["phoneme_ids"][start:end]]

    def phoneme_id_matrix(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """複数行の音素列を (C, L) のパディング行列と長さ配列で返す。

        `edit_distance_batch` に渡して距離をまとめて計算するための形。
        行ごとに `phoneme_id_array` を呼ぶと 2000 件で 15ms かかるが、
        開始位置 + 列番号の外積で一度に引けば 2ms で済む (実測 7 倍)。
        """
        bounds = self._data["phoneme_bounds"]
        starts = bounds[rows]
        lengths = (bounds[rows + 1] - starts).astype(np.int64)
        if rows.size == 0:
            return np.zeros((0, 0), dtype=np.int32), lengths
        width = int(lengths.max())

        columns = np.arange(width)
        valid = columns[None, :] < lengths[:, None]
        # パディング部分は範囲外を指しうるので、引く前に有効域へ丸める。
        # 値は valid 側で捨てるため、どこを指していても構わない。
        flat = starts[:, None] + columns[None, :]
        np.clip(flat, 0, self._data["phoneme_ids"].size - 1, out=flat)
        matrix = np.where(valid, self._distance_ids[self._data["phoneme_ids"][flat]], PAD_ID)
        return matrix, lengths

    def phoneme_lengths(self, rows: np.ndarray) -> np.ndarray:
        """指定した行の音素数。類似度の正規化に使う。

        境界インデックスの差だけなので、音素列そのものを起こさずに済む。
        """
        bounds = self._data["phoneme_bounds"]
        return (bounds[rows + 1] - bounds[rows]).astype(np.int64)

    @property
    def phoneme_csr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """音素列の CSR 表現 (連結 ID, 境界, 距離テーブルへの写像)。

        `distance.edit_distance_csr` に渡してパディング行列を経由せずに
        編集距離を計算するための入口。行列を組む処理 (`phoneme_id_matrix`) は
        53 万候補で 102ms かかるので、Rust 側で CSR を直接読めるならその分が消える。
        """
        return self._data["phoneme_ids"], self._data["phoneme_bounds"], self._distance_ids

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

    def category_counts(self) -> dict[Category, int]:
        """カテゴリごとの語数。"""
        ids, counts = np.unique(self._data["category_ids"], return_counts=True)
        return {
            self._category_vocabulary[int(index)]: int(count)
            for index, count in zip(ids, counts, strict=True)
        }

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

    @property
    def group_ids(self) -> np.ndarray:
        """行ごとの音素列グループ ID。同じ (モーラ数, 音素列) の行が同じ値を持つ。

        行は構築時にグループが隣接するよう並べてあるので (`_locality_order`)、
        この配列は全体で単調非減少。rerank が同音異表記の編集距離を代表 1 件に
        畳むのに使う。
        """
        return self._data["group_ids"]

    def mora_range(self, min_mora: int | None, max_mora: int | None) -> tuple[int, int]:
        """モーラ数が範囲に入る行の連続区間 [start, end) を返す。

        行はモーラ数の昇順に並んでいるので (`_locality_order`)、範囲の切り出しは
        二分探索で済み、該当行は必ず連続する。マスク + fancy indexing で選抜する
        必要がなく、ベクトル行列や列配列をスライスで直接読める。
        """
        moras = self._data["mora_counts"]
        start = 0 if min_mora is None else int(np.searchsorted(moras, min_mora, side="left"))
        end = (
            moras.size if max_mora is None else int(np.searchsorted(moras, max_mora, side="right"))
        )
        return start, max(start, end)

    # --- ベクトル ---------------------------------------------------------

    def vectors(self, space: str) -> np.ndarray:
        """指定した空間の (N, D) int8 行列を mmap で返す。

        候補生成も rerank もこの行列だけで足りる。mmap なので匿名メモリに
        載らず、ページキャッシュとして回収できる — ANN のグラフを持っていた
        頃はここと同じデータをヒープに複製していた
        (`PhoneticSearcher._top_candidates` 参照)。

        **値は int8 の量子化済み。** そのまま内積を取っても元のコサイン
        類似度にはならないので、`scale(space)` を掛けて戻す。掛ける操作は
        Rust 側 (`jpr_distance.top_candidates` / `dot_all`) が担う。
        """
        if space not in self._vectors:
            self._vectors[space] = np.load(self.path / f"vectors-{space}.npy", mmap_mode="r")
        return self._vectors[space]

    def scale(self, space: str) -> float:
        """量子化スケール。int8 の内積にこれの 2 乗を掛けると元の値に戻る。

        行列側とクエリ側で同じスケールを使う (どちらも同じ空間の値なので
        分ける理由がない)。
        """
        return self.meta.scales[space]

    def dequantized(self, space: str) -> np.ndarray:
        """int8 の行列を float32 に戻したものを返す。

        **検索経路では使わない** (行列全体を実体化するので full だと 583MB)。
        量子化の誤差を確かめるテストと、索引を調べる用途のための入口。
        """
        return self.vectors(space).astype(np.float32) * self.scale(space)

    def quantize_query(self, space: str, vector: np.ndarray) -> np.ndarray:
        """クエリベクトルを索引と同じスケールで int8 に落とす。

        索引側と同じ量子化を通さないと内積の尺度が揃わない。範囲外に出た値は
        飽和させる — クエリは索引に無い語でもよいので、索引の最大値を超える
        成分が出うる。
        """
        scaled = np.round(np.asarray(vector, dtype=np.float32) / self.scale(space))
        return np.clip(scaled, -127, 127).astype(np.int8)

    def top_rows(self, space: str, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """索引全体と内積を取り、上位 `k` 行とコサイン類似度を返す。

        内積と Top-K の両方を Rust が担う。NumPy に int8 の GEMV 経路が無い
        ためで、`astype(np.int32)` を挟むと 202 万行ぶんの中間配列
        (582MB) を実体化してしまう。
        """
        vectors = self.vectors(space)
        quantized = self.quantize_query(space, query)
        scale = self.scale(space)
        return _rust.top_candidates(vectors, quantized, k, scale * scale)

    def dot_rows(self, space: str, query: np.ndarray, start: int, end: int) -> np.ndarray:
        """連続区間 [start, end) の全行とのコサイン類似度。

        母集団をスライスで渡すので、mmap 上の連続読みになる。
        """
        vectors = self.vectors(space)
        quantized = self.quantize_query(space, query)
        scale = self.scale(space)
        return _rust.dot_all(vectors[start:end], quantized, scale * scale)

    def dot_selected(self, space: str, query: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """指定した行だけとのコサイン類似度。

        候補が母集団のごく一部のときだけ使う経路 (`search._space_scores` が
        件数で切り替える)。`vectors[rows]` が飛び飛びの行を実体化するので、
        行数が増えるとコストが急伸する — 多いときは連続読みの `dot_rows` が
        速い。
        """
        vectors = self.vectors(space)
        quantized = self.quantize_query(space, query)
        scale = self.scale(space)
        return _rust.dot_all(np.ascontiguousarray(vectors[rows]), quantized, scale * scale)


def write_store(
    path: Path | str,
    entries: Sequence[IndexEntry],
    vectors: dict[str, np.ndarray],
    *,
    dict_type: str = "full",
    progress: object = None,
) -> None:
    """索引をディスクに書く。

    行は入力順ではなく `_locality_order` (モーラ数, 音素列) で格納する。
    検索側のモーラ範囲スライスと同音異表記の畳み込みがこの並びに依存する。
    ベクトル行列も同じ順に並べ替えるので、呼び出し側は entries と vectors の
    行対応だけ揃えればよい。

    ANN のグラフは作らない。候補生成はベクトル行列との内積で足りるので
    (`PhoneticSearcher._top_candidates`)、グラフを持つと同じデータを二重に
    保存することになる。
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    def report(message: str) -> None:
        if callable(progress):
            progress(message)

    report("行をモーラ数順に並べ替え中")
    order = _locality_order(entries)
    entries = [entries[i] for i in order]
    permutation = np.asarray(order, dtype=np.int64)
    vectors = {name: matrix[permutation] for name, matrix in vectors.items()}

    report("語彙メタデータを書き出し中")
    np.savez(path / _ENTRIES_FILE, **_encode_entries(entries))

    # ベクトルは int8 に量子化して保存する。float32 のままだと full で 1.47GB
    # あり、そのままイメージに焼くとコンテナが膨らむ (`_quantize` の項を参照)。
    scales: dict[str, float] = {}
    for space, matrix in vectors.items():
        report(f"ベクトルを量子化して書き出し中: {space}")
        quantized, scale = _quantize(matrix)
        scales[space] = scale
        np.save(path / f"vectors-{space}.npy", quantized)

    meta = StoreMeta(
        version=FORMAT_VERSION,
        count=len(entries),
        dims={name: int(vectors[name].shape[1]) for name in vectors},
        dict_type=dict_type,
        scales=scales,
    )
    (path / _META_FILE).write_text(meta.to_json(), encoding="utf-8")
    report("完了")


__all__ = [
    "CANDIDATE_SPACES",
    "FORMAT_VERSION",
    "SPACES",
    "PhoneticStore",
    "StoreMeta",
    "default_store_path",
    "write_store",
]

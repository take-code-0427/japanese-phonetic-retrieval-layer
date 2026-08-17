"""音韻インデックスの永続化 (Phonetic RAG のオフライン側)。

保存するもの:

    meta.json          形式バージョンと件数、空間の次元
    <配列名>.npy       語彙のメタデータ (表層・読み・音素列・カテゴリ・コスト)
    vectors-<空間>.npy 各埋め込み空間の (G, D) 行列

**語彙メタデータを npz にまとめない。** npz は zip なので mmap できず、
`np.load` が全配列をヒープに展開する (full で 133MB が匿名メモリに載る)。
1 配列 1 ファイルなら mmap で開けて、同じ 133MB が回収可能なページキャッシュ
に移る。圧縮していないのでディスク上の大きさは変わらない。

**行列の行は語ではなく音素列グループ (G)。** 埋め込みも編集距離も音素列だけの
関数なので、同音異表記 (「価格」「架格」「カカク」) はまったく同じベクトルと
同じ距離を持つ。行ごとに持つと full の 202 万語のうち 146 万ぶんしか情報が
無いのに全部を保存し、内積も重複して計算することになる。グループ単位にすると
行が 28% 減り、**索引サイズ・内積の行数・rerank の候補数のすべてに同時に効く**。
行からグループへは `group_ids`、グループから行へは `group_starts` で写す。

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
from .phonology import vowel_skeleton_of

#: 索引形式のバージョン。`PhoneticStore.__init__` が不一致を検出してエラーにする。
#:
#: **上げる基準は「古い索引を読ませると黙って間違うか」。** 配列の並びやエントリの
#: 持ち方が変わったときは上げる (読めば壊れる)。読み方だけを変えたときは上げない —
#: 再構築は 10 分かかるので、動くものを動かなくする理由がない。索引の内容そのものが
#: 変わったとき (語彙の範囲など) も上げる。形式は読めても結果が説明できなくなる。
FORMAT_VERSION = 9

#: 候補生成に使う空間。
#:
#: **phonetic 1 本だけ。** `PhoneticSearcher.candidate_space` (既定 "phonetic") が
#: 索引全体と内積を取る空間で、他の空間は rerank でスコアを足すだけ。
#: どちらもベクトル行列を mmap で読むので、空間を増やしてもディスクだけが
#: 増える (匿名メモリは増えない)。
CANDIDATE_SPACES = ("phonetic",)

#: 索引に保存する空間。
#:
#: `embedding.SPACES` の全空間ではなく、**検索が実際に読む 2 つだけ**。
#: 候補生成が `phonetic`、rerank が `coda` を引く (`search._score_candidates`)。
#: rerank の母音軸はベクトル空間ではなく母音骨格 CSR (`vowel_csr`) を引く —
#: 母音の類似は列の照合であって、プーリングした内積では長さの違いが消える
#: (v8、`distance.vowel_skeleton_similarity` の項を参照)。
#:
#: `consonant` と `rhythm` は書いても読まれない。この 2 つを使うのは
#: `compare` (`search.compare_pronunciations`) だけで、あちらは索引を引かず
#: **入力 2 語をその場で `embed()` する**ので行列が要らない。full の実測で
#: consonant 91MB + rhythm 16MB = 107MB、core で 58MB が死蔵されていた。
#:
#: **空間を rerank に足すときはここに加える。** 索引の再構築が要る
#: (`FORMAT_VERSION` を上げる)。
INDEXED_SPACES = ("phonetic", "coda")

_META_FILE = "meta.json"

#: 語彙メタデータの配列。1 つずつ .npy として置き、mmap で開く
#: (`PhoneticStore.__init__` の項を参照)。
#:
#: 語彙 (`*_vocabulary`) は小さく、開いた時点で Python のリストに起こすので
#: mmap の対象ではないが、置き場所は同じ。
_ENTRY_ARRAYS = (
    "surface_blob",
    "surface_bounds",
    "reading_blob",
    "reading_bounds",
    "pos_vocabulary",
    "pos_ids",
    "category_vocabulary",
    "category_ids",
    "costs",
    "familiarities",
    "mora_counts",
    "phoneme_vocabulary",
    "phoneme_ids",
    "phoneme_bounds",
    "vowel_ids",
    "vowel_bounds",
    "group_ids",
)


def default_store_path() -> Path:
    """既定の索引ディレクトリ。"""
    return Path.home() / ".cache" / "jpr" / "index"


@dataclass(frozen=True)
class StoreMeta:
    version: int
    count: int
    dims: dict[str, int]
    #: 空間ごとの量子化スケール。int8 の値にこれを掛けると元の float32 に戻る。
    scales: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "count": self.count,
                "dims": self.dims,
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
            scales=payload["scales"],
        )


def _quantize(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """(N, D) の float32 行列を int8 とスケールに落とす。

    **スケールは空間ごとに 1 つ。** 行ごとに持つ案も測ったが、`phonetic` は
    L2 正規化済みで行の最大値が 0.24〜0.49 に収まるため再構成誤差が変わらない
    (どちらも 0.0026)。行ごとのスケール配列を持つ複雑さに見合わない。
    `rhythm` だけは正規化しないので絶対量が大きいが (ノルム最大 2.24)、
    空間ごとに最大値を取るこの形なら自動的に追従する。

    量子化の誤差は内積で最大 0.012 (全空間の実測)。順位を分ける
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


def _byte_view(array: np.ndarray) -> memoryview:
    """mmap 配列をバイトの `memoryview` として見る (コピーしない)。

    **NumPy の添字を経由しない**ための入口。`blob[start:end]` は memmap の
    スライスなので、1 回ごとに memmap オブジェクトを構築する。文字列の復号は
    1 検索で数万回走るため、実測で `memmap.__getitem__` が 20 万回・
    2.4ms/クエリを占めていた。`memoryview` ならバッファプロトコルで直接
    読めて **5.5 倍速い** (20000 件で 146ms -> 26ms)。

    ページは mmap のまま参照するので、**匿名メモリは増えない** (実測 +0kB)。
    境界配列を `np.asarray` で実体化すると 3 本で 24MB がヒープに載るので、
    そちらは採らない — この索引は匿名メモリを削るために mmap にしてある。
    """
    return memoryview(array).cast("B")


def _decode_string(blob: memoryview, boundaries: memoryview, row: int) -> str:
    start, end = boundaries[row], boundaries[row + 1]
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

    **音素列はグループごとに 1 本だけ持つ** (v5)。同じ (モーラ数, 音素列) の
    行は定義上まったく同じ音素列なので、行ごとに複製する必要がない。行から
    音素列を引くときは `group_ids` を挟む。full の実測で 21.7MB -> 17.5MB。
    """
    surface_blob, surface_bounds = _encode_strings([e.surface for e in entries])
    reading_blob, reading_bounds = _encode_strings([e.reading for e in entries])

    # 品詞とカテゴリは値の種類が少ないので、辞書 + ID で持つ。
    pos_values = sorted({e.pos for e in entries})
    pos_ids = {value: index for index, value in enumerate(pos_values)}
    category_values = sorted({e.category.value for e in entries})
    category_ids = {value: index for index, value in enumerate(category_values)}

    # 音素列を ID 化して連結する。同じ (モーラ数, 音素列) が続く区間へ同じ
    # グループ ID を振り (ソート済みなので隣接比較で足りる)、音素列そのものは
    # グループが変わったときだけ書き出す。
    #
    # 母音骨格も同じグループ単位で持つ (v8)。骨格は音素列だけの関数なので
    # (`phonology.vowel_skeleton_of`)、同音異表記はまったく同じ骨格を持つ。
    # rerank の母音軸がこの CSR を `edit_distance_csr` に渡す。
    phoneme_ids: list[int] = []
    phoneme_offsets = [0]
    vowel_ids: list[int] = []
    vowel_offsets = [0]
    group_ids = np.zeros(len(entries), dtype=np.int32)
    symbols: dict[str, int] = {}
    group = -1
    previous_key: tuple[int, tuple[str, ...]] | None = None
    for row, entry in enumerate(entries):
        key = (entry.mora_count, entry.phonemes)
        if key != previous_key:
            group += 1
            previous_key = key
            for symbol in entry.phonemes:
                phoneme_ids.append(symbols.setdefault(symbol, len(symbols)))
            phoneme_offsets.append(len(phoneme_ids))
            for symbol in vowel_skeleton_of(entry.phonemes):
                vowel_ids.append(symbols.setdefault(symbol, len(symbols)))
            vowel_offsets.append(len(vowel_ids))
        group_ids[row] = group

    _check_blob_fits("音素 blob", len(phoneme_ids))
    _check_blob_fits("母音骨格 blob", len(vowel_ids))

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
        "familiarities": np.array([e.familiarity for e in entries], dtype=np.float32),
        "mora_counts": np.array([e.mora_count for e in entries], dtype=np.int16),
        "phoneme_vocabulary": np.array(phoneme_vocabulary, dtype=np.str_),
        "phoneme_ids": np.array(phoneme_ids, dtype=np.uint8),
        "phoneme_bounds": np.asarray(phoneme_offsets, dtype=np.int32),
        "vowel_ids": np.array(vowel_ids, dtype=np.uint8),
        "vowel_bounds": np.asarray(vowel_offsets, dtype=np.int32),
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

        # 語彙メタデータも mmap で開く。
        #
        # **npz にまとめてはいけない。** npz は zip なので mmap できず、
        # `np.load` が全配列をヒープに展開する。full の実測で 133MB がまるごと
        # 匿名メモリ (回収不可) に載り、ベクトル行列を mmap にした意味を
        # 相殺していた。個別の .npy に分けると同じ 133MB がページキャッシュ
        # (回収可能) に移る — 圧縮していないので**ディスク上の大きさは変わらない**。
        self._data = {
            name: np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in _ENTRY_ARRAYS
        }

        # 行がモーラ数順であることは v3 の不変条件で、モーラ範囲の連続スライス
        # (`mora_range`) がこれに依存する。壊れた索引が黙って誤った母集団を
        # 返すより、開いた時点で落ちるほうがいい。検証は 2ms 程度。
        moras = self._data["mora_counts"]
        if moras.size > 1 and np.any(np.diff(moras) < 0):
            raise ValueError(
                f"索引の行がモーラ数順に並んでいません: {self.path}\n"
                "`jpr build-index --force` で再構築してください。"
            )

        # ベクトルと音素列はグループ単位で持つので (v5)、行数と行列の行数が
        # 一致しない。グループ ID が行数を超えていないことだけ確かめておく
        # (壊れた索引が別の語のベクトルを黙って返すのを防ぐ)。
        groups = self._data["group_ids"]
        self._group_count = int(groups[-1]) + 1 if groups.size else 0
        if self._data["phoneme_bounds"].size != self._group_count + 1:
            raise ValueError(
                f"音素列の本数がグループ数と一致しません: {self.path}\n"
                "`jpr build-index --force` で再構築してください。"
            )
        if self._data["vowel_bounds"].size != self._group_count + 1:
            raise ValueError(
                f"母音骨格の本数がグループ数と一致しません: {self.path}\n"
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

        # 文字列の復号は NumPy の添字を通さず memoryview で読む
        # (`_byte_view` の項を参照)。境界は int32 なのでバイト経由で写す。
        self._surface_blob = _byte_view(self._data["surface_blob"])
        self._surface_bounds = _byte_view(self._data["surface_bounds"]).cast("i")
        self._reading_blob = _byte_view(self._data["reading_blob"])
        self._reading_bounds = _byte_view(self._data["reading_bounds"]).cast("i")

        # 1 行ずつ引く列も同じ理由で memoryview を通す (`entry` / `phonemes`)。
        # 配列演算で読む経路は NumPy のまま (`self._data`) — こちらはスカラー
        # 引きの Python 呼び出しが支配的な経路だけを置き換える。
        self._row_group_ids = _byte_view(self._data["group_ids"]).cast("i")
        self._phoneme_bounds_view = _byte_view(self._data["phoneme_bounds"]).cast("i")
        self._phoneme_ids_view = _byte_view(self._data["phoneme_ids"])
        self._mora_counts_view = _byte_view(self._data["mora_counts"]).cast("h")
        self._pos_ids_view = _byte_view(self._data["pos_ids"]).cast("h")
        self._category_ids_view = _byte_view(self._data["category_ids"]).cast("b")
        self._costs_view = _byte_view(self._data["costs"]).cast("i")
        self._familiarities_view = _byte_view(self._data["familiarities"]).cast("f")

        self._vectors: dict[str, np.ndarray] = {}
        self._group_starts: np.ndarray | None = None
        self._mora_edge_cache: np.ndarray | None = None

    def __len__(self) -> int:
        return self.meta.count

    # --- 語彙メタデータ ---------------------------------------------------

    def surface(self, row: int) -> str:
        return _decode_string(self._surface_blob, self._surface_bounds, row)

    def reading(self, row: int) -> str:
        return _decode_string(self._reading_blob, self._reading_bounds, row)

    def phonemes(self, row: int) -> tuple[str, ...]:
        group = self._row_group_ids[row]
        bounds = self._phoneme_bounds_view
        start, end = bounds[group], bounds[group + 1]
        vocabulary = self._phoneme_vocabulary
        return tuple(vocabulary[i] for i in self._phoneme_ids_view[start:end])

    def phoneme_id_array(self, row: int) -> np.ndarray:
        """音素列を距離計算用の ID 配列として返す。

        索引が持つ ID は語彙内で振ったものなので、距離テーブルの ID に写し直す。
        記号のタプルを経由しないので rerank の内側で使える。
        """
        group = self._row_group_ids[row]
        bounds = self._phoneme_bounds_view
        start, end = bounds[group], bounds[group + 1]
        return self._distance_ids[self._data["phoneme_ids"][start:end]]

    def phoneme_id_matrix(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """複数行の音素列を (C, L) のパディング行列と長さ配列で返す。

        `edit_distance_batch` に渡して距離をまとめて計算するための形。
        行ごとに `phoneme_id_array` を呼ぶと 2000 件で 15ms かかるが、
        開始位置 + 列番号の外積で一度に引けば 2ms で済む (実測 7 倍)。
        """
        bounds = self._data["phoneme_bounds"]
        groups = self._data["group_ids"][rows]
        starts = bounds[groups]
        lengths = (bounds[groups + 1] - starts).astype(np.int64)
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
        groups = self._data["group_ids"][rows]
        return (bounds[groups + 1] - bounds[groups]).astype(np.int64)

    @property
    def phoneme_csr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """音素列の CSR 表現 (連結 ID, 境界, 距離テーブルへの写像)。

        `distance.edit_distance_csr` に渡してパディング行列を経由せずに
        編集距離を計算するための入口。行列を組む処理 (`phoneme_id_matrix`) は
        53 万候補で 102ms かかるので、Rust 側で CSR を直接読めるならその分が消える。

        **境界は行ではなくグループで引く** (v5)。渡す添字は `group_ids[rows]`
        であって行番号ではない。編集距離は音素列だけで決まるので、同音異表記に
        同じ計算を繰り返す理由がない。
        """
        return self._data["phoneme_ids"], self._data["phoneme_bounds"], self._distance_ids

    def vowel_lengths(self, rows: np.ndarray) -> np.ndarray:
        """指定した行の母音骨格の長さ。類似度の正規化に使う。"""
        bounds = self._data["vowel_bounds"]
        groups = self._data["group_ids"][rows]
        return (bounds[groups + 1] - bounds[groups]).astype(np.int64)

    @property
    def vowel_csr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """母音骨格の CSR 表現 (連結 ID, 境界, 距離テーブルへの写像)。

        rerank の母音軸がここを `distance.edit_distance_csr` に渡す (v8)。
        骨格の記号 (母音 + 促音・撥音) はすべて音素なので、ID の語彙も距離
        テーブルへの写像も音素列の CSR とそのまま共有する。添字がグループで
        あることも `phoneme_csr` と同じ。
        """
        return self._data["vowel_ids"], self._data["vowel_bounds"], self._distance_ids

    def entry(self, row: int) -> IndexEntry:
        return IndexEntry(
            surface=self.surface(row),
            reading=self.reading(row),
            phonemes=self.phonemes(row),
            mora_count=self._mora_counts_view[row],
            pos=self._pos_vocabulary[self._pos_ids_view[row]],
            category=self._category_vocabulary[self._category_ids_view[row]],
            cost=self._costs_view[row],
            familiarity=self._familiarities_view[row],
        )

    def category_of(self, row: int) -> Category:
        return self._category_vocabulary[self._category_ids_view[row]]

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
    def familiarities(self) -> np.ndarray:
        """行ごとの一般性 (0.0〜1.0)。rerank が候補全体に対して引く。"""
        return self._data["familiarities"]

    @property
    def mora_counts(self) -> np.ndarray:
        return self._data["mora_counts"]

    @property
    def group_ids(self) -> np.ndarray:
        """行ごとの音素列グループ ID。同じ (モーラ数, 音素列) の行が同じ値を持つ。

        行は構築時にグループが隣接するよう並べてあるので (`_locality_order`)、
        この配列は全体で単調非減少。**ベクトルと音素列はこの ID で引く** (v5)。
        rerank が同音異表記の編集距離を代表 1 件に畳むのにも使う。
        """
        return self._data["group_ids"]

    @property
    def group_count(self) -> int:
        """音素列グループの数。ベクトル行列と音素 CSR の行数。"""
        return self._group_count

    @property
    def group_starts(self) -> np.ndarray:
        """グループごとの先頭行。グループ -> 行 の展開に使う。

        行はグループが隣接するよう並んでいるので (`_locality_order`)、
        グループ `g` の行は `[group_starts[g], group_starts[g + 1])` の連続区間。
        末尾に総行数を置いた長さ `group_count + 1` の配列。
        """
        if self._group_starts is None:
            groups = self._data["group_ids"]
            starts = np.empty(self._group_count + 1, dtype=np.int64)
            starts[self._group_count] = groups.size
            # 各グループの先頭行 = そのグループ ID が初めて現れる位置。
            # 逆順に書くと同じ ID の中で最も小さい行が最後に残る。
            starts[groups[::-1]] = np.arange(groups.size - 1, -1, -1, dtype=np.int64)
            self._group_starts = starts
        return self._group_starts

    def group_mora_range_of_rows(self, start: int, end: int) -> tuple[int, int]:
        """行区間 [start, end) を覆うグループの連続区間 [start, end) を返す。

        グループもモーラ数の昇順に並ぶ (行がそう並んでおり、グループは行の
        連続区間なので)、行区間をグループ ID に写すだけで済む。

        **モーラ範囲の境界はグループを割らない。** グループは同じモーラ数の
        行だけで構成されるので (キーが (モーラ数, 音素列))、モーラ数で切った
        行区間はグループ境界にちょうど揃う。
        """
        groups = self._data["group_ids"]
        if start >= end:
            first = int(groups[start]) if start < groups.size else self._group_count
            return first, first
        return int(groups[start]), int(groups[end - 1]) + 1

    @property
    def _mora_edges(self) -> np.ndarray:
        """モーラ数ごとの行の開始位置。`_mora_edges[n]` が「n モーラ未満」の行数。

        **毎回 `searchsorted` を呼んではいけない。** 探索する `mora_counts` は
        mmap 上の 202 万要素なので、二分探索がページフォールトを踏んで 1 回
        1.2ms かかる。既定の検索は候補生成のたびに帯の両端を引くので、実測で
        **1 クエリあたり 3.2ms** — 内積 (4ms) に並ぶ第 2 の項目になっていた。

        モーラ数の種類は 2〜12 の 11 個しかないので、全境界を一度に求めて
        持てば以降は配列引きで済む (前計算は 3.1ms、1 回だけ)。
        """
        if self._mora_edge_cache is None:
            moras = self._data["mora_counts"]
            largest = int(moras[-1]) if moras.size else 0
            # 添字 n をそのまま使えるよう 0..largest+1 の全整数で切る。
            self._mora_edge_cache = np.searchsorted(
                moras, np.arange(largest + 2), side="left"
            ).astype(np.int64)
        return self._mora_edge_cache

    def mora_range(self, min_mora: int | None, max_mora: int | None) -> tuple[int, int]:
        """モーラ数が範囲に入る行の連続区間 [start, end) を返す。

        行はモーラ数の昇順に並んでいるので (`_locality_order`)、該当行は必ず
        連続する。マスク + fancy indexing で選抜する必要がなく、ベクトル行列や
        列配列をスライスで直接読める。

        境界は前計算した表から引く (`_mora_edges`)。mmap 上の二分探索は
        1 回 1.2ms かかり、毎クエリ払うには重すぎる。
        """
        edges = self._mora_edges
        total = self._data["mora_counts"].size
        # 表は「n モーラ未満の行数」なので、上端は max_mora + 1 の位置を引く。
        start = 0 if min_mora is None else int(edges[min(max(min_mora, 0), edges.size - 1)])
        end = total if max_mora is None else int(edges[min(max(max_mora + 1, 0), edges.size - 1)])
        return start, max(start, end)

    # --- ベクトル ---------------------------------------------------------

    def vectors(self, space: str) -> np.ndarray:
        """指定した空間の (G, D) int8 行列を mmap で返す。

        **行は語ではなく音素列グループ** (v5)。埋め込みは音素列だけの関数なので
        (`embedding.embed` は `Pronunciation` しか見ない)、同音異表記は
        まったく同じベクトルになる。行ごとに持つと full の 202 万行のうち
        146 万行ぶんしか情報が無いのに全部を保存し、内積も重複して計算する
        ことになる。実測で全空間・全 202 万行がグループ先頭と完全一致した。

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

    def top_groups(
        self,
        space: str,
        query: np.ndarray,
        k: int,
        start: int = 0,
        end: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """グループ区間 [start, end) と内積を取り、上位 `k` 件とコサイン類似度を返す。

        **返るのは行ではなくグループ** (v5)。同音異表記はベクトルが同一なので、
        行単位で Top-K を取ると上位が同じ音の異表記で埋まる。呼び出し側が
        `group_starts` で行へ展開する (`search._top_candidates`)。

        **区間はモーラ帯** (v6)。既定の検索は候補生成の直後にモーラ差で候補を
        削るので (`search._MAX_MORA_GAP`)、帯の外に内積を取っても捨てるだけに
        なる。グループもモーラ数順に並ぶので区間はスライスで切れ、mmap の
        連続読みのまま行数だけが減る。返すグループ番号は `start` を足して
        索引全体の番号に戻す — 呼び出し側は区間を意識しなくてよい。

        内積と Top-K の両方を Rust が担う。NumPy に int8 の GEMV 経路が無い
        ためで、`astype(np.int32)` を挟むと索引ぶんの中間配列を実体化してしまう。
        """
        vectors = self.vectors(space)
        if end is None:
            end = vectors.shape[0]
        quantized = self.quantize_query(space, query)
        scale = self.scale(space)
        # mmap のスライスはビューなので、ここでコピーは起きない。
        groups, scores = _rust.top_candidates(vectors[start:end], quantized, k, scale * scale)
        if start:
            groups += start
        return groups, scores

    def dot_groups(self, space: str, query: np.ndarray, start: int, end: int) -> np.ndarray:
        """連続区間 [start, end) の全**グループ**とのコサイン類似度。

        母集団をスライスで渡すので、mmap 上の連続読みになる。
        """
        vectors = self.vectors(space)
        quantized = self.quantize_query(space, query)
        scale = self.scale(space)
        return _rust.dot_all(vectors[start:end], quantized, scale * scale)

    def dot_selected_groups(self, space: str, query: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """指定したグループだけとのコサイン類似度。

        候補が母集団のごく一部のときだけ使う経路 (`search._space_scores` が
        件数で切り替える)。`vectors[groups]` が飛び飛びの行を実体化するので、
        件数が増えるとコストが急伸する — 多いときは連続読みの `dot_groups` が
        速い。
        """
        vectors = self.vectors(space)
        quantized = self.quantize_query(space, query)
        scale = self.scale(space)
        return _rust.dot_all(np.ascontiguousarray(vectors[groups]), quantized, scale * scale)


def write_store(
    path: Path | str,
    entries: Sequence[IndexEntry],
    vectors: dict[str, np.ndarray],
    *,
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

    report("語彙メタデータを書き出し中")
    encoded = _encode_entries(entries)
    # npz にまとめず 1 配列 1 ファイルで置く。読み手が mmap で開けるように
    # するため (`PhoneticStore.__init__` の項を参照)。
    for name in _ENTRY_ARRAYS:
        np.save(path / f"{name}.npy", encoded[name])

    # ベクトルはグループごとに 1 行だけ書く (v5)。埋め込みは音素列だけの関数
    # なので同音異表記は同一のベクトルになり、行ごとに持つと full の 202 万行
    # のうち 146 万行ぶんしか情報が無いのに全部を保存することになる。
    # グループの先頭行を選べば代表になる (行はグループが隣接するよう並ぶ)。
    group_ids = encoded["group_ids"]
    group_count = int(group_ids[-1]) + 1 if group_ids.size else 0
    leaders = np.empty(group_count, dtype=np.int64)
    leaders[group_ids[::-1]] = np.arange(group_ids.size - 1, -1, -1, dtype=np.int64)
    # 並べ替えと代表の選抜を 1 回の fancy indexing にまとめる。
    representative = permutation[leaders]

    # ベクトルは int8 に量子化して保存する。float32 のままだと full で 1.47GB
    # あり、そのままイメージに焼くとコンテナが膨らむ (`_quantize` の項を参照)。
    # 保存するのは検索が読む空間だけ (`INDEXED_SPACES`)。`consonant` と
    # `rhythm` は `compare` がその場で埋め込むので索引に要らない。
    scales: dict[str, float] = {}
    dims: dict[str, int] = {}
    for space in INDEXED_SPACES:
        if space not in vectors:
            raise ValueError(f"索引に必要な空間が渡されていません: {space}")
        report(f"ベクトルを量子化して書き出し中: {space}")
        quantized, scale = _quantize(vectors[space][representative])
        scales[space] = scale
        dims[space] = int(quantized.shape[1])
        np.save(path / f"vectors-{space}.npy", quantized)

    meta = StoreMeta(
        version=FORMAT_VERSION,
        count=len(entries),
        dims=dims,
        scales=scales,
    )
    (path / _META_FILE).write_text(meta.to_json(), encoding="utf-8")
    report("完了")


__all__ = [
    "CANDIDATE_SPACES",
    "FORMAT_VERSION",
    "INDEXED_SPACES",
    "SPACES",
    "PhoneticStore",
    "StoreMeta",
    "default_store_path",
    "write_store",
]

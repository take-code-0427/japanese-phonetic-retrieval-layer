"""SudachiDict のバイナリ辞書 (system.dic) から全語彙を列挙する。

SudachiPy には語彙を列挙する API が無く (`lookup` は完全一致のみ) 、Java 版の
`dump` サブコマンドも提供されない。そのため音韻インデックスの構築には
system.dic を直接読む必要がある。

ファイル構造 (すべて little endian):

    header      : version(u64) createdAt(u64) description(256B)
    grammar     : posSize(u16) POS[posSize] (各 6 個の可変長文字列)
                  leftIdSize(i16) rightIdSize(i16) matrix(i16 * left * right)
    lexicon     : trieSize(i32) trie(u32 * trieSize)
                  wordIdTableSize(i32) wordIdTable(bytes)
                  wordParamsSize(i32) wordParams(i16 * 3 * size)
                  wordInfoOffsets(i32 * size) wordInfos(...)

可変長文字列は「長さ(1〜2B) + UTF-16LE 本体」。長さの最上位ビットが立って
いれば 2 バイト長。
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_HEADER_SIZE = 8 + 8 + 256
_POS_FIELD_COUNT = 6


@dataclass(frozen=True)
class DictionaryEntry:
    """辞書 1 語。"""

    surface: str
    reading: str
    normalized: str
    pos: tuple[str, ...]
    #: Sudachi の連接コスト。小さいほど解析上出現しやすく、語の一般性の弱い指標になる。
    cost: int

    @property
    def is_noun(self) -> bool:
        return bool(self.pos) and self.pos[0] == "名詞"


def find_system_dic() -> Path:
    """インストール済み SudachiDict (full) の system.dic のパスを返す。"""
    try:
        import sudachidict_full as module
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise FileNotFoundError(
            "sudachidict_full がインストールされていません。`uv sync` を実行してください。"
        ) from exc

    path = Path(module.__file__).parent / "resources" / "system.dic"
    if not path.exists():  # pragma: no cover - 環境依存
        raise FileNotFoundError(f"system.dic が見つかりません: {path}")
    return path


class _Reader:
    """バイト列を前方向に読み進めるカーソル。"""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.pos = pos

    def u16(self) -> int:
        (value,) = struct.unpack_from("<H", self.data, self.pos)
        self.pos += 2
        return value

    def i16(self) -> int:
        (value,) = struct.unpack_from("<h", self.data, self.pos)
        self.pos += 2
        return value

    def i32(self) -> int:
        (value,) = struct.unpack_from("<i", self.data, self.pos)
        self.pos += 4
        return value

    def u8(self) -> int:
        value = self.data[self.pos]
        self.pos += 1
        return value

    def string(self) -> str:
        length = self.u8()
        if length >= 0x80:
            length = ((length & 0x7F) << 8) | self.u8()
        start = self.pos
        self.pos += length * 2
        return self.data[start : self.pos].decode("utf-16-le")

    def skip(self, count: int) -> None:
        self.pos += count


class SystemDictionary:
    """system.dic を読み込み、全語彙を列挙できるようにする。"""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._data = self.path.read_bytes()
        self._pos_table: tuple[tuple[str, ...], ...] = ()
        self._offset_table_pos = 0
        self._word_count = 0
        self._parse_layout()

    def _parse_layout(self) -> None:
        reader = _Reader(self._data, _HEADER_SIZE)

        pos_size = reader.u16()
        pos_table: list[tuple[str, ...]] = []
        for _ in range(pos_size):
            pos_table.append(tuple(reader.string() for _ in range(_POS_FIELD_COUNT)))
        self._pos_table = tuple(pos_table)

        left_id_size = reader.i16()
        right_id_size = reader.i16()
        reader.skip(left_id_size * right_id_size * 2)

        trie_size = reader.i32()
        reader.skip(trie_size * 4)

        word_id_table_size = reader.i32()
        reader.skip(word_id_table_size)

        word_count = reader.i32()
        self._word_count = word_count
        # wordParams は 1 語あたり leftId/rightId/cost の i16 × 3。
        self._word_params_pos = reader.pos
        reader.skip(word_count * 6)

        self._offset_table_pos = reader.pos
        self._validate_offset_table()

    def _validate_offset_table(self) -> None:
        """オフセットテーブルの先頭値が WordInfo 領域の先頭を指すことを確認する。

        構造の読み違いを黙って通さないためのガード。辞書フォーマットが変わった
        場合はここで失敗する。
        """
        expected = self._offset_table_pos + self._word_count * 4
        (first,) = struct.unpack_from("<i", self._data, self._offset_table_pos)
        if first != expected:
            raise ValueError(
                "system.dic の構造を解釈できません "
                f"(WordInfo 先頭の推定値 {expected} に対し実値 {first})。"
                "SudachiDict のフォーマットが変更された可能性があります。"
            )

    def __len__(self) -> int:
        return self._word_count

    def entry(self, word_id: int) -> DictionaryEntry:
        (offset,) = struct.unpack_from("<i", self._data, self._offset_table_pos + word_id * 4)
        reader = _Reader(self._data, offset)

        surface = reader.string()
        # headwordLength (可変長整数)。音韻情報には使わないので読み捨てる。
        if reader.u8() >= 0x80:
            reader.skip(1)
        pos_id = reader.u16()
        normalized = reader.string()
        reader.i32()  # dictionaryFormWordId
        reading = reader.string()

        # wordParams の 3 番目が語のコスト。
        (cost,) = struct.unpack_from("<h", self._data, self._word_params_pos + word_id * 6 + 4)

        # Sudachi は表層と一致する読み・正規化形を空文字で省略する。
        return DictionaryEntry(
            surface=surface,
            reading=reading or surface,
            normalized=normalized or surface,
            pos=self._pos_table[pos_id] if pos_id < len(self._pos_table) else (),
            cost=cost,
        )

    def entries(self) -> Iterator[DictionaryEntry]:
        for word_id in range(self._word_count):
            yield self.entry(word_id)

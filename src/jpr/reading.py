"""任意の日本語テキストから読み (カタカナ) を取り出す。

Sudachi の形態素解析を使い、複合語やフレーズも語ごとの読みを連結して扱う。
辞書に無い語 (OOV) は Sudachi が表層をそのまま返すため、かな入力はそのまま
カタカナ化され、漢字だけの未知語は読みが取れず空になる。
"""

from __future__ import annotations

import threading

from sudachipy import Dictionary, SplitMode

from .phonology import to_katakana

#: 読み・正規化形のキャッシュ上限。
_CACHE_LIMIT = 4096


class ReadingExtractor:
    """Sudachi による読み取得。

    Sudachi の辞書ロードは重いので、インスタンスを使い回すこと。
    """

    def __init__(self, dict_type: str = "full", split_mode: SplitMode = SplitMode.C) -> None:
        self._dict_type = dict_type
        self._split_mode = split_mode
        self._dictionary = Dictionary(dict=dict_type)
        self._tokenizer = self._dictionary.create()
        # Sudachi の Tokenizer は解析用バッファを内部に持つので複数スレッドから
        # 同時に呼べない (Rust 側が `RuntimeError: Already borrowed` を投げる)。
        # web.py の非 async なエンドポイントは starlette のスレッドプールで動くため
        # 同時リクエストがそのまま衝突する — 実際に本番の /api/similar が 500 を
        # 返した。tokenize は 1 語あたり 1ms 未満でキャッシュも効くので、
        # スレッドごとに tokenizer を持つ (Sudachi 辞書を重複して抱える) より
        # 直列化するほうが安い。
        self._lock = threading.Lock()
        # 解析結果はプロセス内で不変なのでキャッシュする。メソッドに lru_cache を
        # 付けるとクラス単位のキャッシュが self を握り続けインスタンスが解放され
        # なくなるため、インスタンスごとに持つ。
        self._reading_cache: dict[str, str] = {}
        self._normalized_cache: dict[str, str] = {}

    def reading_of(self, text: str) -> str:
        """テキスト全体の読みをカタカナで返す。

        入力が既にかなだけなら形態素解析を経ずにカタカナ化する。
        """
        cached = self._reading_cache.get(text)
        if cached is not None:
            return cached

        reading = self._compute_reading(text)
        self._remember(self._reading_cache, text, reading)
        return reading

    def _compute_reading(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""

        katakana = to_katakana(stripped)
        if _is_all_kana(katakana):
            return katakana

        parts: list[str] = []
        # Morpheme は解析器のバッファを参照するので、読み出しまでロック内で終える。
        with self._lock:
            for morpheme in self._tokenizer.tokenize(stripped, self._split_mode):
                reading = morpheme.reading_form()
                # OOV では読みが空になることがあるため表層で代替する。
                parts.append(to_katakana(reading or morpheme.surface()))
        return "".join(parts)

    def normalize(self, text: str) -> str:
        """正規化表記を返す。表層の揺れを吸収して自己一致を判定するのに使う。"""
        cached = self._normalized_cache.get(text)
        if cached is not None:
            return cached

        stripped = text.strip()
        if not stripped:
            normalized = ""
        else:
            # 生成子のままロックを出ると解析器のバッファを外で触るので、
            # ロック内で文字列に確定させる。
            with self._lock:
                morphemes = self._tokenizer.tokenize(stripped, self._split_mode)
                normalized = "".join(m.normalized_form() for m in morphemes)

        self._remember(self._normalized_cache, text, normalized)
        return normalized

    @staticmethod
    def _remember(cache: dict[str, str], key: str, value: str) -> None:
        """上限付きでキャッシュに入れる。上限に達したら丸ごと捨てる。

        検索のクエリは繰り返し現れるので、厳密な LRU は要らない。
        """
        if len(cache) >= _CACHE_LIMIT:
            cache.clear()
        cache[key] = value


_KANA_RANGES = (
    (0x3041, 0x309F),  # ひらがな
    (0x30A1, 0x30FF),  # カタカナ
)


def _is_all_kana(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if not any(start <= code <= end for start, end in _KANA_RANGES):
            return False
    return bool(text)


__all__ = ["ReadingExtractor"]

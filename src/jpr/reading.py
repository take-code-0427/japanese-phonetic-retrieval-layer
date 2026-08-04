"""任意の日本語テキストから読み (カタカナ) を取り出す。

Sudachi の形態素解析を使い、複合語やフレーズも語ごとの読みを連結して扱う。
辞書に無い語 (OOV) は Sudachi が表層をそのまま返すため、かな入力はそのまま
カタカナ化され、漢字だけの未知語は読みが取れず空になる。
"""

from __future__ import annotations

from functools import lru_cache

from sudachipy import Dictionary, SplitMode

from .phonology import to_katakana


class ReadingExtractor:
    """Sudachi による読み取得。

    Sudachi の辞書ロードは重いので、インスタンスを使い回すこと。
    """

    def __init__(self, dict_type: str = "full", split_mode: SplitMode = SplitMode.C) -> None:
        self._dict_type = dict_type
        self._split_mode = split_mode
        self._dictionary = Dictionary(dict=dict_type)
        self._tokenizer = self._dictionary.create()

    @lru_cache(maxsize=4096)  # noqa: B019 - 解析結果はプロセス内で不変
    def reading_of(self, text: str) -> str:
        """テキスト全体の読みをカタカナで返す。

        入力が既にかなだけなら形態素解析を経ずにカタカナ化する。
        """
        stripped = text.strip()
        if not stripped:
            return ""

        katakana = to_katakana(stripped)
        if _is_all_kana(katakana):
            return katakana

        parts: list[str] = []
        for morpheme in self._tokenizer.tokenize(stripped, self._split_mode):
            reading = morpheme.reading_form()
            # OOV では読みが空になることがあるため表層で代替する。
            parts.append(to_katakana(reading or morpheme.surface()))
        return "".join(parts)

    @lru_cache(maxsize=4096)  # noqa: B019 - 同上
    def normalize(self, text: str) -> str:
        """正規化表記を返す。表層の揺れを吸収して自己一致を判定するのに使う。"""
        stripped = text.strip()
        if not stripped:
            return ""
        morphemes = self._tokenizer.tokenize(stripped, self._split_mode)
        return "".join(m.normalized_form() for m in morphemes)


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

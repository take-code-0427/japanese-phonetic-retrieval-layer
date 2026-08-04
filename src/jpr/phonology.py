"""かな読みを音素列・モーラ列に変換する。

音素表記は日本語音韻論の慣用に従い、拗音を口蓋化子音 1 音素として扱う
(キャ -> ky a)。長音・促音・撥音は独立した特殊モーラとして保持する。
"""

from __future__ import annotations

from dataclasses import dataclass

# 特殊モーラ。音素としても同じ記号を使う。
LONG = "R"  # 長音 (ー)
GEMINATE = "Q"  # 促音 (ッ)
MORAIC_N = "N"  # 撥音 (ン)

SPECIAL_PHONEMES = frozenset({LONG, GEMINATE, MORAIC_N})

VOWELS = frozenset({"a", "i", "u", "e", "o"})

# 直音・拗音のかな 1〜2 文字を (子音, 母音) に対応させる表。
# 子音が空文字列のものは母音単独モーラ。
_KANA_TABLE: dict[str, tuple[str, str]] = {}


def _register(mapping: dict[str, str], consonant: str) -> None:
    for kana, vowel in mapping.items():
        _KANA_TABLE[kana] = (consonant, vowel)


_register({"ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o"}, "")
_register({"カ": "a", "キ": "i", "ク": "u", "ケ": "e", "コ": "o"}, "k")
_register({"ガ": "a", "ギ": "i", "グ": "u", "ゲ": "e", "ゴ": "o"}, "g")
_register({"サ": "a", "ス": "u", "セ": "e", "ソ": "o"}, "s")
_register({"ザ": "a", "ズ": "u", "ゼ": "e", "ゾ": "o"}, "z")
_register({"タ": "a", "テ": "e", "ト": "o"}, "t")
_register({"ダ": "a", "デ": "e", "ド": "o"}, "d")
_register({"ナ": "a", "ニ": "i", "ヌ": "u", "ネ": "e", "ノ": "o"}, "n")
_register({"ハ": "a", "ヘ": "e", "ホ": "o"}, "h")
_register({"バ": "a", "ビ": "i", "ブ": "u", "ベ": "e", "ボ": "o"}, "b")
_register({"パ": "a", "ピ": "i", "プ": "u", "ペ": "e", "ポ": "o"}, "p")
_register({"マ": "a", "ミ": "i", "ム": "u", "メ": "e", "モ": "o"}, "m")
_register({"ヤ": "a", "ユ": "u", "ヨ": "o"}, "y")
_register({"ラ": "a", "リ": "i", "ル": "u", "レ": "e", "ロ": "o"}, "r")
_register({"ワ": "a"}, "w")

# 音韻的に子音が交替するもの。
_KANA_TABLE["シ"] = ("sh", "i")
_KANA_TABLE["ジ"] = ("j", "i")
_KANA_TABLE["チ"] = ("ch", "i")
_KANA_TABLE["ツ"] = ("ts", "u")
_KANA_TABLE["ヂ"] = ("j", "i")
_KANA_TABLE["ヅ"] = ("z", "u")
_KANA_TABLE["フ"] = ("f", "u")
_KANA_TABLE["ヒ"] = ("hy", "i")

# 歴史的仮名。現代語では発音上 オ/エ に合流する。
_KANA_TABLE["ヲ"] = ("", "o")
_KANA_TABLE["ヰ"] = ("", "i")
_KANA_TABLE["ヱ"] = ("", "e")

# 拗音 (小書きのヤ行・ワ行を伴う 2 文字)。
_YOUON_SMALL = {"ャ": "a", "ュ": "u", "ョ": "o", "ヮ": "a"}
_PALATALIZED = {
    "キ": "ky",
    "ギ": "gy",
    "シ": "sh",
    "ジ": "j",
    "チ": "ch",
    "ニ": "ny",
    "ヒ": "hy",
    "ビ": "by",
    "ピ": "py",
    "ミ": "my",
    "リ": "ry",
    "フ": "fy",
}
for _base, _cons in _PALATALIZED.items():
    for _small, _vowel in _YOUON_SMALL.items():
        _KANA_TABLE[_base + _small] = (_cons, _vowel)

# 外来語表記の拗音 (小書きのア行を伴う 2 文字)。
_SMALL_VOWELS = {"ァ": "a", "ィ": "i", "ゥ": "u", "ェ": "e", "ォ": "o"}
_FOREIGN = {
    "フ": "f",
    "ヴ": "v",
    "ウ": "w",
    "ツ": "ts",
    "テ": "t",
    "デ": "d",
    "シ": "sh",
    "ジ": "j",
    "チ": "ch",
    "ク": "k",
    "グ": "g",
    "ズ": "z",
    "ス": "s",
    "ト": "t",
    "ド": "d",
}
for _base, _cons in _FOREIGN.items():
    for _small, _vowel in _SMALL_VOWELS.items():
        _KANA_TABLE.setdefault(_base + _small, (_cons, _vowel))

# ヴ の直音。
_register({"ヴ": "u"}, "v")

# 小書き文字が単独で現れた場合 (前接が拗音を作れないとき) は母音単独として拾う。
for _small, _vowel in _SMALL_VOWELS.items():
    _KANA_TABLE.setdefault(_small, ("", _vowel))
for _small, _vowel in _YOUON_SMALL.items():
    _KANA_TABLE.setdefault(_small, ("y", _vowel))

_LONG_MARKS = frozenset({"ー", "〜", "–", "—", "ｰ"})
_SMALL_TSU = frozenset({"ッ", "ｯ"})

# ひらがな -> カタカナ の差分。
_HIRA_TO_KATA_OFFSET = ord("ア") - ord("あ")


def to_katakana(text: str) -> str:
    """ひらがなをカタカナに寄せる。それ以外の文字はそのまま返す。"""
    out: list[str] = []
    for ch in text:
        if "ぁ" <= ch <= "ゖ":
            out.append(chr(ord(ch) + _HIRA_TO_KATA_OFFSET))
        else:
            out.append(ch)
    return "".join(out)


@dataclass(frozen=True)
class Mora:
    """1 モーラ。特殊モーラでは consonant/vowel が空になる。"""

    kana: str
    consonant: str
    vowel: str
    special: str = ""

    @property
    def phonemes(self) -> tuple[str, ...]:
        if self.special:
            return (self.special,)
        if self.consonant:
            return (self.consonant, self.vowel)
        return (self.vowel,)


@dataclass(frozen=True)
class Pronunciation:
    """読みの音韻表現。"""

    reading: str
    moras: tuple[Mora, ...]

    @property
    def phonemes(self) -> tuple[str, ...]:
        return tuple(p for m in self.moras for p in m.phonemes)

    @property
    def mora_count(self) -> int:
        return len(self.moras)

    @property
    def vowel_skeleton(self) -> tuple[str, ...]:
        """母音・特殊モーラのみの列。韻の判定に使う。"""
        out: list[str] = []
        for m in self.moras:
            if m.special == LONG and out:
                out.append(out[-1])
            elif m.special:
                out.append(m.special)
            else:
                out.append(m.vowel)
        return tuple(out)

    def phoneme_string(self) -> str:
        return " ".join(self.phonemes)


def analyze_reading(reading: str) -> Pronunciation:
    """カタカナ/ひらがな読みを音韻表現に変換する。

    解釈できない文字 (漢字・記号・ラテン文字など) は黙って読み飛ばす。
    """
    kana = to_katakana(reading)
    moras: list[Mora] = []
    i = 0
    n = len(kana)
    while i < n:
        ch = kana[i]

        if ch in _LONG_MARKS:
            moras.append(Mora(kana=ch, consonant="", vowel="", special=LONG))
            i += 1
            continue
        if ch in _SMALL_TSU:
            moras.append(Mora(kana=ch, consonant="", vowel="", special=GEMINATE))
            i += 1
            continue
        if ch == "ン":
            moras.append(Mora(kana=ch, consonant="", vowel="", special=MORAIC_N))
            i += 1
            continue

        # 2 文字の拗音を優先して照合する。
        if i + 1 < n:
            pair = kana[i : i + 2]
            hit = _KANA_TABLE.get(pair)
            if hit is not None:
                moras.append(Mora(kana=pair, consonant=hit[0], vowel=hit[1]))
                i += 2
                continue

        hit = _KANA_TABLE.get(ch)
        if hit is not None:
            moras.append(Mora(kana=ch, consonant=hit[0], vowel=hit[1]))
            i += 1
            continue

        # 未知の文字はスキップ。
        i += 1

    return Pronunciation(reading=kana, moras=tuple(moras))

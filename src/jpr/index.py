"""索引に載せる語彙の表現と、辞書エントリを索引向けに選別する規則。

実際の永続化と検索は `store` と `search` が担う。ここは「どの語を索引に載せ、
どう分類するか」だけを持つ。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# 検索結果として意味を持たない品詞。助詞・記号などを索引に入れても
# ダジャレ候補にならずノイズになるだけなので落とす。
EXCLUDED_POS = frozenset({"助詞", "助動詞", "補助記号", "記号", "空白", "接続詞"})

_KATAKANA = frozenset(chr(code) for code in range(0x30A1, 0x30FB)) | {"ー"}

# Sudachi のコストを一般性に写すための境界。full 辞書の実測分布 (中央値 11000、
# 第 1 十分位 5000) をもとに、下位 1 割以下を「よく知られた語」、
# 上位 1/4 以上を「稀な語」とみなす。
COST_FAMILIAR = 5000
COST_RARE = 15000


class Category(StrEnum):
    """語彙カテゴリ。

    SudachiDict full は地名 49 万・人名 24 万を含み、これが索引の 7 割を占める。
    人名の異表記 (「ココミ」に対する心々美・湖々美・瑚々海…) は音韻的に密集する
    ため、区別せずに検索すると一般語や商品名が上位から押し出される。
    カテゴリを保持して検索時に選べるようにする。
    """

    COMMON = "common"  # 普通名詞・動詞・形容詞など一般語
    PRODUCT = "product"  # 商品名・作品名などの固有名詞 (一般)
    PERSON = "person"  # 人名
    PLACE = "place"  # 地名
    OTHER = "other"


#: 既定で検索対象にするカテゴリ。人名・地名は明示的に指定したときだけ引く。
DEFAULT_CATEGORIES = frozenset({Category.COMMON, Category.PRODUCT, Category.OTHER})


def classify(pos: tuple[str, ...]) -> Category:
    """品詞列から語彙カテゴリを決める。"""
    if len(pos) < 2:
        return Category.OTHER
    if pos[1] != "固有名詞":
        return Category.COMMON

    subtype = pos[2] if len(pos) > 2 else ""
    if subtype == "人名":
        return Category.PERSON
    if subtype == "地名":
        return Category.PLACE
    # 固有名詞かつ人名・地名でないもの: 商品名・作品名・組織名など。
    return Category.PRODUCT


@dataclass(frozen=True)
class IndexEntry:
    """索引 1 件。"""

    surface: str
    reading: str
    phonemes: tuple[str, ...]
    mora_count: int
    pos: str
    category: Category
    #: Sudachi のコスト。小さいほど一般的な語である傾向がある。
    cost: int

    @property
    def is_proper_noun(self) -> bool:
        return self.category in (Category.PRODUCT, Category.PERSON, Category.PLACE)

    @property
    def familiarity(self) -> float:
        """語の一般性を 0.0〜1.0 で近似する。

        Sudachi のコストは言語モデル上の出現しにくさなので、これを反転して
        「知られている語らしさ」の弱い指標として使う。厳密な頻度ではないため
        順位の同点をほどく程度の重みでしか使わないこと。
        """
        return familiarity_of(self.cost)


def familiarity_of(cost: int) -> float:
    """Sudachi のコストを 0.0〜1.0 の一般性に写す。"""
    if cost <= COST_FAMILIAR:
        return 1.0
    if cost >= COST_RARE:
        return 0.0
    return 1.0 - (cost - COST_FAMILIAR) / (COST_RARE - COST_FAMILIAR)


def has_pronounceable_reading(reading: str) -> bool:
    """読みがカタカナのみで構成されているか。"""
    return bool(reading) and all(ch in _KATAKANA for ch in reading)


def is_searchable_surface(surface: str) -> bool:
    """表層が音韻検索の結果として提示に値するかを判定する。

    SudachiDict full には ASCII 見出しの外来語 (`cicli`, `kikugi`) が多数含まれる。
    これらは読みを持つため音韻的には正しくヒットするが、「音が似た日本語」を
    求める用途では雑音になるので落とす。
    """
    if not surface:
        return False
    return any("ぁ" <= ch <= "ヿ" or "一" <= ch <= "鿿" or ch == "ー" for ch in surface)


__all__ = [
    "COST_FAMILIAR",
    "COST_RARE",
    "DEFAULT_CATEGORIES",
    "EXCLUDED_POS",
    "Category",
    "IndexEntry",
    "classify",
    "familiarity_of",
    "has_pronounceable_reading",
    "is_searchable_surface",
]

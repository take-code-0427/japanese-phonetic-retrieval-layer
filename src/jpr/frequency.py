"""語の一般性を Wikipedia の出現記事数から測る。

`index.familiarity_of` (Sudachi の連接コストの反転) の置き換え。**連接コストは
頻度ではないので、知名度の指標として部分的に逆転していた** — full 辞書の実測で
「電話」(cost 6389) が「デンダ」(cost 4187) より低い一般性を与えられ、
`familiarity` の重みを上げるほど結果が悪化した:

    weights.familiarity  0.15 -> 0.50 での「電話」上位
    0.15  前夜 / 田野 / 原話 / 談話 / 禅話
    0.50  前夜 / デンダ / レンガ / ダンガ / テンダ / ギンガ

連接コストが小さいのは「短く品詞的に単純な語」であって「知られた語」ではない。
加えて full の既定カテゴリ 92 万行のうち 8.9% が `COST_FAMILIAR` (5000) 以下で
1.0 に飽和し、0.8% は cost <= 0 で値を持たない。

出典は jawiki (2022-10-20) を UniDic 3.1.0 の短単位で分割した頻度表
(adno/wikipedia-word-frequency-clean, BSD-3-Clause)。`data/` に同梱する。
"""

from __future__ import annotations

import lzma
import math
from collections.abc import Callable
from functools import lru_cache
from importlib.resources import files

#: 同梱する頻度表。表層形 (NFKC 正規化済み) と出現記事数のタブ区切り。
#: 最終行の `[TOTAL]` はコーパス全体の記事数。
_DATA_FILE = "jawiki-documents.tsv.xz"

_TOTAL_KEY = "[TOTAL]"

#: 生の出現数ではなく**出現記事数**を使う。1 つの記事に大量に出る固有名詞
#: (作品名・人名) が生カウントでは過大に評価されるため。実測で「東京」は
#: 43.6 万回 / 18.5 万記事、対して稀な専門語は回数だけが偏る。

#: 一般性は百万記事あたりの出現記事数の常用対数で測る。**線形では使えない** —
#: 出現記事数は 3 (収録の下限) から 18.5 万 (「東京」) まで 5 桁に渡るので、
#: 線形に写すと上位数語以外が 0 に潰れる。
#:
#: 境界は収録語の分布から決める。実測の z 分位は
#: 1% 0.376 / 25% 0.453 / 50% 0.710 / 75% 1.244 / 90% 1.959 / 99% 3.398。
#: `Z_MAX` 4.5 は「東京」(4.93) を 1.0 に飽和させ、`Z_MIN` 0.0 は下端を潰さない
#: (0.3 に上げると収録語の 1 割が 0.0 で同点になり、下端の順序が消える)。
Z_MIN = 0.0
Z_MAX = 4.5

#: 頻度表に無い語の一般性。
#:
#: **0 にはしない。** 表は UniDic 短単位なので、SudachiDict full の主力である
#: 複合語は分割されて収録されていない (実測で「テレビ電話」「ノートパソコン」
#: 「エア電話」がいずれも未収録)。0 にするとこれらが最下位に沈み、包含検索
#: (`search._containment`) が拾った良い候補を潰す。
#:
#: **構成語から合成する案は採れない。** 未収録語を最長一致で分割して構成語の
#: 一般性を取る案を実測したが、1 文字の高頻度語が拾われて逆効果だった:
#:
#:     かけく -> かけ + く    min z=3.636   (沈めたい語が高くなる)
#:     化薬   -> 化 + 薬      min z=3.729
#:     ダンガ -> ダン + ガ    min z=3.389
#:     ラリンゴ -> ラリン + ゴ min z=0.897
#:
#: 沈めたい語 (かけく・化薬・ダンガ) と拾いたい語 (テレビ電話) が分離しない。
#: 分割は語の切れ目を知らないので、区別できるだけの情報が無い。
#:
#: 0.25 は収録語の第 1 四分位 (0.101) と中央値 (0.158) の上、第 3 四分位
#: (0.276) の下。「収録されている稀語より上、平均的な語より下」に置く。
UNKNOWN_FAMILIARITY = 0.25


def _data_path() -> str:
    return str(files("jpr.data") / _DATA_FILE)


@lru_cache(maxsize=1)
def document_counts() -> tuple[dict[str, int], int]:
    """頻度表を読む。戻り値は (表層 -> 出現記事数, コーパス全体の記事数)。

    索引構築でしか読まないので遅延で足りる (実測 0.9 秒 / 56 万語)。検索側は
    索引の `familiarities` 配列を引くだけでこの表に触れない。
    """
    counts: dict[str, int] = {}
    total = 0
    with lzma.open(_data_path(), "rt", encoding="utf-8") as handle:
        for line in handle:
            word, _, count = line.rstrip("\n").partition("\t")
            if word == _TOTAL_KEY:
                total = int(count)
                continue
            counts[word] = int(count)
    if not total:
        raise ValueError(f"頻度表に {_TOTAL_KEY} 行がありません: {_data_path()}")
    return counts, total


def familiarity_of_documents(documents: int, total: int) -> float:
    """出現記事数を 0.0〜1.0 の一般性に写す。"""
    if documents <= 0:
        return UNKNOWN_FAMILIARITY
    z = math.log10(documents / total * 1e6 + 1.0)
    return max(0.0, min(1.0, (z - Z_MIN) / (Z_MAX - Z_MIN)))


def familiarity_lookup() -> Callable[[str], float]:
    """表層を一般性に写す関数を返す。未収録語は `UNKNOWN_FAMILIARITY`。"""
    counts, total = document_counts()

    def lookup(surface: str) -> float:
        documents = counts.get(surface)
        if documents is None:
            return UNKNOWN_FAMILIARITY
        return familiarity_of_documents(documents, total)

    return lookup


__all__ = [
    "UNKNOWN_FAMILIARITY",
    "Z_MAX",
    "Z_MIN",
    "document_counts",
    "familiarity_lookup",
    "familiarity_of_documents",
]

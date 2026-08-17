"""MCP サーバ。LLM に音韻検索を tool として提供する。

LLM 単体では「音が似ている」という関係を確実には扱えない。この tool は
LLM が外部の音韻空間を引くための窓で、意味空間 (LLM 自身が持つ) と
音韻空間 (ここ) を分離したまま組み合わせられるようにする。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from .index import Category, parse_categories
from .phrase import DEFAULT_MAX_CHUNK_MORAS, DEFAULT_MIN_CHUNK_SCORE
from .search import DEFAULT_CANDIDATES, PRESETS, PhoneticSearcher
from .serialize import (
    comparison_payload,
    phrase_candidate_payload,
    pronunciation_payload,
    search_result_payload,
)
from .store import PhoneticStore, default_store_path

_SEARCH_DESCRIPTION = """\
日本語の単語・フレーズと「音が似ている」語を検索する。

意味ではなく発音の近さで引く。次のような場合に使う:

- なぞなぞ・ダジャレの答えを探す (「乳首みたいなお菓子」→ チョコビ)
- 韻を踏む語を探す (preset="rhyme")
- 聞き間違い・音声認識の誤りを補正する (preset="mishearing")
- 「〜みたいに聞こえる語」を問われた

意味的な絞り込みが必要な場合 (「お菓子の中から」など) は、音韻的に近い語が
数百件あるため、まず limit を大きめにして候補を取り、意味の判断は呼び出し側で
行うこと。

「もっと長い語で」「4 モーラ以上で」のようにモーラ数 (拍) を指定された場合は
min_mora / max_mora を使う。通常の検索は音韻空間の近傍を引くので、モーラ数の
違う語は近傍に入らず出てこない。

結果の `containment` は「クエリの音がその語に完全な形で入っている」度合い
(0 なら入っていない)。「りんご」に対する「ラリンゴ」「五輪後」のような語が
これに当たり、値はクエリが語全体に占める割合なので余分が多いほど低い。
「〜が入っている語」「〜を含む語」を問われたときはこのフィールドで絞る。

`score` の内訳も返すので、順位の理由を検算できる: `phonetic_similarity`
(音素列全体)、`coda_similarity` (語尾)、`vowel_similarity` (母音列 = 韻)、
`familiarity` (語の一般性)。発音を人に示すときは `ipa` を使う
(`phonemes` は内部表記)。"""

_COMPOSE_DESCRIPTION = """\
長い入力を「複数の語 + 助詞」の連なりに置き換える (空耳・替え歌)。

**長いフレーズには search_phonetically を使わない。** あちらは 1 語を 1 語に
写すので、「ワタシノナマエハ」のような長い入力に音が近い単一の語は辞書に
存在せず、答えが返らない。こちらは入力をモーラ境界で区間に切り、区間ごとに
別の語を当てて繋ぐ。

    ワタシノナマエハ -> 私 | の | 名前 | は

次のような場合に使う:

- 空耳を作る (「〜に聞こえる日本語の文」)
- 替え歌・歌詞の音合わせ
- 長いフレーズを別の語の連なりで言い換える
- 外国語の音を日本語で写す

`segments` が「入力のどこが何になったか」を持つので、意味が通る候補を
そこから選び直すこと。**音韻スコアが最上位の候補が意味として最良とは限らない** —
むしろ上位は音が完全に一致する無意味な列になりやすい。意味の判断は
呼び出し側 (LLM) の仕事。"""

_COMPARE_DESCRIPTION = """\
2 つの日本語表現の音韻類似度を計算する。

意味的類似度とは独立した軸として返す。空間別の内訳 (子音・母音・語尾・リズム)
も返すので、「どこが似ているのか」を判断できる。"""

_PRONOUNCE_DESCRIPTION = """\
日本語テキストの読み・音素列・IPA (国際音声記号)・モーラ構造を返す。索引を引かずに
音韻表現だけを得る。

`phonemes` は内部の音素記号 (ヘボン式寄りの ASCII)、`ipa` は同じものの IPA 表記。
発音を人に示すときや他言語の音と比べるときは `ipa` を使う。"""

#: MCP が 1 回に返す件数の上限。LLM のコンテキストを溢れさせないため、
#: CLI や Web と違って無制限は露出しない。
_MAX_LIMIT = 200


_INSTRUCTIONS = """\
日本語の音韻空間を引くためのサーバ。意味的な類似性ではなく「音の近さ」を扱う。

音韻類似度は意味とは独立した軸なので、意味的な制約のある問い (「乳首みたいな
お菓子」など) では、返ってきた候補から意味で選び直す必要がある。音韻スコアが
最も高い語が答えとは限らない。

**入力の長さで使う tool が変わる。** 1 語に対して音の近い語を引くなら
search_phonetically、長いフレーズを語の連なりに置き換えるなら compose_phrase。
長い入力に音が近い単一の語は辞書に存在しないので、search_phonetically に
フレーズを渡しても答えは返らない。"""


def create_server(index_path: Path | str | None = None) -> MCPServer:
    """MCP サーバを組み立てる。"""
    path = Path(index_path) if index_path is not None else default_store_path()
    server = MCPServer("jpr", instructions=_INSTRUCTIONS)

    # 索引のロードは最初の呼び出しまで遅らせる。mmap なので実質即座に開くが、
    # 索引が無い環境でもサーバ自体は起動できるようにしておく。
    state: dict[str, PhoneticSearcher] = {}

    def searcher() -> PhoneticSearcher:
        if "searcher" not in state:
            state["searcher"] = PhoneticSearcher(PhoneticStore(path))
        return state["searcher"]

    @server.tool(name="search_phonetically", description=_SEARCH_DESCRIPTION)
    def search_phonetically(
        query: str,
        limit: int = 10,
        preset: str = "pun",
        categories: list[str] | None = None,
        candidates: int = DEFAULT_CANDIDATES,
        min_mora: int | None = None,
        max_mora: int | None = None,
    ) -> str:
        """音が近い語を検索する。

        Args:
            query: 検索語。漢字・ひらがな・カタカナのいずれでもよい。
            limit: 返す件数。
            preset: "pun" (ダジャレ) / "rhyme" (韻) / "mishearing" (聞き間違い)。
            categories: 絞り込むカテゴリ。common (一般語) / product (商品名・作品名)
                / person (人名) / place (地名)。省略すると人名と地名を除いて検索する。
            candidates: ANN から取る候補数。増やすと再現率が上がる。
            min_mora: 結果のモーラ数 (拍) の下限。
            max_mora: 結果のモーラ数 (拍) の上限。min_mora と併せて範囲を切る。
                **指定すると通常の近傍検索とは別の経路になり、その範囲の語を
                全件走査する。** 通常の検索は音韻空間の近傍を引くので、モーラ数の
                違う語 (「乳首」3 モーラに対する「筑前煮」5 モーラ) は近傍に
                入らず出てこない。数秒かかる。
        """
        if preset not in PRESETS:
            return json.dumps(
                {"error": f"未知のプリセット: {preset}", "available": sorted(PRESETS)},
                ensure_ascii=False,
            )

        try:
            parsed = parse_categories(categories or ())
        except ValueError as exc:
            return json.dumps(
                {"error": str(exc), "available": [c.value for c in Category]},
                ensure_ascii=False,
            )

        engine = searcher()
        scanned: int | None = None
        if min_mora is not None or max_mora is not None:
            try:
                scanned = engine.mora_range_size(min_mora, max_mora)
            except ValueError as exc:
                return json.dumps({"error": str(exc)}, ensure_ascii=False)

        pronunciation, results = engine.search(
            query,
            # 上限は必ず掛ける。無制限を許すと数万件が JSON-RPC に載り、
            # LLM のコンテキストを溢れさせる。全件が要るなら CLI か Web を使う。
            limit=min(max(limit, 1), _MAX_LIMIT),
            preset=preset,
            categories=parsed,
            candidates=candidates,
            min_mora=min_mora,
            max_mora=max_mora,
        )
        payload: dict[str, Any] = {
            "query": query,
            **pronunciation_payload(pronunciation),
            "preset": preset,
            "min_mora": min_mora,
            "max_mora": max_mora,
            "scanned": scanned,
            "note": (
                "score は音韻的な近さのみを表し、意味は考慮していない。"
                "意味的な制約がある問いでは、この候補から意味で選び直すこと。"
            ),
            "results": [search_result_payload(r) for r in results],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @server.tool(name="compose_phrase", description=_COMPOSE_DESCRIPTION)
    def compose_phrase(
        text: str,
        limit: int = 10,
        max_chunk_moras: int = DEFAULT_MAX_CHUNK_MORAS,
        min_chunk_score: float = DEFAULT_MIN_CHUNK_SCORE,
        allow_particles: bool = True,
    ) -> str:
        """入力を複数の語 + 助詞の連なりに合成する。

        Args:
            text: 入力。漢字・ひらがな・カタカナのいずれでもよい。長いフレーズを想定する。
            limit: 返す候補数。
            max_chunk_moras: 1 区間に許すモーラ数の上限。上げると 1 区間に長い語を
                当てられるが、区間の数が増えて遅くなる。
            min_chunk_score: 区間ごとの音韻類似度の下限。上げると音の一致が
                厳しくなり、候補が減る。
            allow_particles: 助詞・助動詞を繋ぎに使うか。**索引に 1 モーラの語が
                無いので、false にすると 1 モーラの区間が埋まらなくなる。**
        """
        engine = searcher()
        pronunciation, candidates = engine.compose(
            text,
            limit=min(max(limit, 1), _MAX_LIMIT),
            max_chunk_moras=max_chunk_moras,
            min_chunk_score=min_chunk_score,
            allow_particles=allow_particles,
        )
        return json.dumps(
            {
                "text": text,
                **pronunciation_payload(pronunciation),
                "note": (
                    "score は音韻的な近さのみを表し、意味は考慮していない。"
                    "上位は音が合うだけの無意味な列になりやすいので、"
                    "segments の対応を見て意味が通る候補を選び直すこと。"
                ),
                "results": [phrase_candidate_payload(c) for c in candidates],
            },
            ensure_ascii=False,
            indent=2,
        )

    @server.tool(name="compare_phonetically", description=_COMPARE_DESCRIPTION)
    def compare_phonetically(a: str, b: str) -> str:
        """2 語の音韻類似度を計算する。

        Args:
            a: 比較する表現。
            b: 比較する表現。
        """
        comparison = searcher().compare(a, b)
        return json.dumps(comparison_payload(comparison), ensure_ascii=False, indent=2)

    @server.tool(name="pronounce", description=_PRONOUNCE_DESCRIPTION)
    def pronounce(text: str) -> str:
        """読みと音素列を返す。

        Args:
            text: 日本語テキスト。
        """
        pronunciation = searcher().pronounce(text)
        return json.dumps(
            {
                "text": text,
                **pronunciation_payload(pronunciation),
                "vowel_skeleton": list(pronunciation.vowel_skeleton),
            },
            ensure_ascii=False,
            indent=2,
        )

    return server


def serve(index_path: Path | str | None = None) -> None:
    """stdio で MCP サーバを動かす。"""
    create_server(index_path).run()


__all__ = ["create_server", "serve"]

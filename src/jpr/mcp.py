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

from .index import Category
from .search import DEFAULT_CANDIDATES, PRESETS, PhoneticSearcher
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
行うこと。"""

_COMPARE_DESCRIPTION = """\
2 つの日本語表現の音韻類似度を計算する。

意味的類似度とは独立した軸として返す。空間別の内訳 (子音・母音・語尾・リズム)
も返すので、「どこが似ているのか」を判断できる。"""

_PRONOUNCE_DESCRIPTION = """\
日本語テキストの読み・音素列・モーラ構造を返す。索引を引かずに音韻表現だけを得る。"""


_INSTRUCTIONS = """\
日本語の音韻空間を引くためのサーバ。意味的な類似性ではなく「音の近さ」を扱う。

音韻類似度は意味とは独立した軸なので、意味的な制約のある問い (「乳首みたいな
お菓子」など) では、返ってきた候補から意味で選び直す必要がある。音韻スコアが
最も高い語が答えとは限らない。"""


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
    ) -> str:
        """音が近い語を検索する。

        Args:
            query: 検索語。漢字・ひらがな・カタカナのいずれでもよい。
            limit: 返す件数。
            preset: "pun" (ダジャレ) / "rhyme" (韻) / "mishearing" (聞き間違い)。
            categories: 絞り込むカテゴリ。common (一般語) / product (商品名・作品名)
                / person (人名) / place (地名)。省略すると人名と地名を除いて検索する。
            candidates: ANN から取る候補数。増やすと再現率が上がる。
        """
        if preset not in PRESETS:
            return json.dumps(
                {"error": f"未知のプリセット: {preset}", "available": sorted(PRESETS)},
                ensure_ascii=False,
            )

        parsed: list[Category] | None = None
        if categories:
            try:
                parsed = [Category(name) for name in categories]
            except ValueError as exc:
                return json.dumps(
                    {"error": str(exc), "available": [c.value for c in Category]},
                    ensure_ascii=False,
                )

        pronunciation, results = searcher().search(
            query,
            limit=limit,
            preset=preset,
            categories=parsed,
            candidates=candidates,
        )
        payload: dict[str, Any] = {
            "query": query,
            "reading": pronunciation.reading,
            "phonemes": list(pronunciation.phonemes),
            "mora_count": pronunciation.mora_count,
            "preset": preset,
            "note": (
                "score は音韻的な近さのみを表し、意味は考慮していない。"
                "意味的な制約がある問いでは、この候補から意味で選び直すこと。"
            ),
            "results": [
                {
                    "word": r.surface,
                    "reading": r.reading,
                    "score": r.score,
                    "phonetic_similarity": r.phonetic_similarity,
                    "coda_similarity": r.coda_similarity,
                    "mora_count": r.mora_count,
                    "category": r.category.value,
                    "familiarity": r.familiarity,
                }
                for r in results
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    @server.tool(name="compare_phonetically", description=_COMPARE_DESCRIPTION)
    def compare_phonetically(a: str, b: str) -> str:
        """2 語の音韻類似度を計算する。

        Args:
            a: 比較する表現。
            b: 比較する表現。
        """
        comparison = searcher().compare(a, b)
        return json.dumps(
            {
                "a": {
                    "text": comparison.a_text,
                    "reading": comparison.a_reading,
                    "phonemes": list(comparison.a_phonemes),
                },
                "b": {
                    "text": comparison.b_text,
                    "reading": comparison.b_reading,
                    "phonemes": list(comparison.b_phonemes),
                },
                "phonetic_similarity": comparison.similarity,
                "phonetic_distance": comparison.distance,
                "spaces": comparison.spaces,
            },
            ensure_ascii=False,
            indent=2,
        )

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
                "reading": pronunciation.reading,
                "phonemes": list(pronunciation.phonemes),
                "mora_count": pronunciation.mora_count,
                "moras": [m.kana or m.special for m in pronunciation.moras],
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

"""音韻検索の Web フロントエンド (HTTP API + 静的ページ)。

MCP が LLM 向けの窓なのに対し、こちらは人が音韻空間を直接覗くための窓。
索引を 1 度だけ mmap して常駐させるので、CLI のようにプロセス起動ごとの
Sudachi ロード (数百 ms) と索引オープンを払わずに済む。

API は CLI の `--json` 出力と同じ構造を返す。フロント側の JS が CLI 出力を
そのまま読める形にしておくと、curl でのデバッグと画面表示が食い違わない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .distance import CONSONANTS, VOWELS, align_phonemes
from .index import Category
from .phonology import GEMINATE, LONG, MORAIC_N, analyze_reading
from .search import (
    DEFAULT_CANDIDATES,
    DEFAULT_PRESET,
    PRESETS,
    PhoneticSearcher,
    compare_pronunciations,
)
from .store import INNER_PRODUCT_SPACES, PhoneticStore, default_store_path

STATIC_DIR = Path(__file__).parent / "static"

#: 1 リクエストで返す件数の上限。索引には音が近い語が数百件あるため、
#: limit を無制限にすると rerank の打ち切り線が効かず素直に遅くなる。
MAX_LIMIT = 200

#: ANN から取る候補数の上限。ここを大きくすると再現率と引き換えに
#: rerank のコストが線形に増える (DEFAULT_CANDIDATES の項も参照)。
MAX_CANDIDATES = 50_000


def _parse_categories(value: str | None) -> list[Category] | None:
    """カンマ区切りのカテゴリ名を Category に変換する。"""
    if not value:
        return None
    result: list[Category] = []
    for name in value.split(","):
        name = name.strip()
        if not name:
            continue
        try:
            result.append(Category(name))
        except ValueError:
            valid = ", ".join(c.value for c in Category)
            raise HTTPException(
                status_code=400,
                detail=f"未知のカテゴリ '{name}' (利用可能: {valid})",
            ) from None
    return result or None


def create_app(index_path: Path | str | None = None) -> FastAPI:
    """Web アプリを組み立てる。"""
    path = Path(index_path) if index_path is not None else default_store_path()
    app = FastAPI(
        title="jpr",
        description="日本語の音韻検索レイヤー (Japanese Phonetic Retrieval)",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    # 索引 (mmap) と Sudachi のロードは最初のリクエストまで遅らせる。索引が
    # 無い環境でもサーバは起動でき、/api/info が原因を 503 で返せるようにする。
    state: dict[str, PhoneticSearcher] = {}

    def searcher() -> PhoneticSearcher:
        if "searcher" not in state:
            try:
                state["searcher"] = PhoneticSearcher(PhoneticStore(path))
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"索引を開けません: {exc} (`jpr build-index` で構築してください)",
                ) from exc
        return state["searcher"]

    @app.get("/api/similar")
    def api_similar(
        q: str = Query(..., min_length=1, description="検索語 (漢字・かな・カタカナ)"),
        limit: int = Query(20, ge=1, le=MAX_LIMIT),
        preset: str = Query(DEFAULT_PRESET),
        candidates: int = Query(DEFAULT_CANDIDATES, ge=1, le=MAX_CANDIDATES),
        min_score: float = Query(0.0, ge=0.0, le=1.0),
        categories: str | None = Query(None, description="カンマ区切り"),
    ) -> dict[str, Any]:
        """音が近い語を返す。"""
        if preset not in PRESETS:
            raise HTTPException(
                status_code=400,
                detail=f"未知のプリセット '{preset}' (利用可能: {', '.join(sorted(PRESETS))})",
            )

        engine = searcher()
        pronunciation, results = engine.search(
            q,
            limit=limit,
            preset=preset,
            candidates=candidates,
            min_score=min_score,
            categories=_parse_categories(categories),
        )
        return {
            "query": q,
            "reading": pronunciation.reading,
            "phonemes": list(pronunciation.phonemes),
            "mora_count": pronunciation.mora_count,
            "preset": preset,
            "results": [
                {
                    "word": r.surface,
                    "reading": r.reading,
                    "score": r.score,
                    "phonetic_similarity": r.phonetic_similarity,
                    "embedding_similarity": r.embedding_similarity,
                    "coda_similarity": r.coda_similarity,
                    "vowel_similarity": r.vowel_similarity,
                    "mora_count": r.mora_count,
                    "category": r.category.value,
                    "pos": r.pos,
                    "familiarity": r.familiarity,
                    "phonemes": list(r.phonemes),
                }
                for r in results
            ],
        }

    @app.get("/api/pronounce")
    def api_pronounce(
        text: str = Query(..., min_length=1, description="日本語テキスト"),
    ) -> dict[str, Any]:
        """読み・音素列・モーラ構造を返す。"""
        pronunciation = searcher().pronounce(text)
        return {
            "text": text,
            "reading": pronunciation.reading,
            "phonemes": list(pronunciation.phonemes),
            "mora_count": pronunciation.mora_count,
            "moras": [m.kana or m.special for m in pronunciation.moras],
            "vowel_skeleton": list(pronunciation.vowel_skeleton),
        }

    @app.get("/api/compare")
    def api_compare(
        a: str = Query(..., min_length=1),
        b: str = Query(..., min_length=1),
    ) -> dict[str, Any]:
        """2 語の音韻類似度を返す。索引は不要で、読み取得器だけで完結する。"""
        extractor = searcher().extractor
        comparison = compare_pronunciations(
            a,
            b,
            analyze_reading(extractor.reading_of(a)),
            analyze_reading(extractor.reading_of(b)),
        )
        return {
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
            "similarity": comparison.similarity,
            "distance": comparison.distance,
            "spaces": comparison.spaces,
        }

    @app.get("/api/info")
    def api_info() -> dict[str, Any]:
        """索引のメタ情報。フロントの起動時にプリセットとカテゴリを引くのにも使う。"""
        store = searcher().store
        counts = store.category_counts()
        return {
            "path": str(store.path),
            "format_version": store.meta.version,
            "dict_type": store.meta.dict_type,
            "count": len(store),
            "spaces": [
                {
                    "name": name,
                    "dim": dim,
                    # 候補生成に使う空間のみ ANN を張る。他は rerank でベクトルを直接引く。
                    "role": "ANN + rerank" if name in INNER_PRODUCT_SPACES else "rerank のみ",
                }
                for name, dim in store.meta.dims.items()
            ],
            "categories": [
                {"name": category.value, "count": count}
                for category, count in sorted(counts.items(), key=lambda item: -item[1])
            ],
            "presets": sorted(PRESETS),
            "default_preset": DEFAULT_PRESET,
            "default_candidates": DEFAULT_CANDIDATES,
        }

    @app.get("/api/phonemes")
    def api_phonemes() -> dict[str, Any]:
        """音素の素性表を返す。

        フロントはこれを使って音素チップの色を決める。色を UI 側の固定表に
        持たせると素性表を変えたときに黙ってずれるので、素性そのものを配る。
        """
        return {
            "consonants": {
                symbol: {
                    "place": c.place,
                    "manner": c.manner,
                    "voiced": c.voiced,
                    "palatalized": c.palatalized,
                }
                for symbol, c in CONSONANTS.items()
            },
            "vowels": {
                symbol: {
                    "height": v.height,
                    "backness": v.backness,
                    "rounded": v.rounded,
                }
                for symbol, v in VOWELS.items()
            },
            "special": {LONG: "長音", GEMINATE: "促音", MORAIC_N: "撥音"},
        }

    @app.get("/api/align")
    def api_align(
        a: str = Query(..., min_length=1, description="音素列 (空白区切り)"),
        b: str = Query(..., min_length=1, description="音素列 (空白区切り)"),
    ) -> dict[str, Any]:
        """2 つの音素列を対応付け、各対の素性距離を返す。

        「なぜ近いのか」を数値 1 つに畳まず、どの音素がどれに対応して
        どれだけ離れているかまで見せるために使う。
        """
        pairs = align_phonemes(tuple(a.split()), tuple(b.split()))
        return {
            "pairs": [
                {"a": left, "b": right, "distance": round(dist, 4), "op": op}
                for left, right, dist, op in pairs
            ],
            "total": round(sum(p[2] for p in pairs), 4),
        }

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def serve_web(
    index_path: Path | str | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> None:
    """開発用の HTTP サーバを立てる。"""
    import uvicorn

    uvicorn.run(create_app(index_path), host=host, port=port)


__all__ = ["create_app", "serve_web"]

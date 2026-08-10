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

from .distance import CONSONANTS, VOWELS, align_phonemes, ipa_transcription, phoneme_ipa
from .index import Category
from .phonology import GEMINATE, LONG, MORAIC_N, analyze_reading
from .phrase import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_CHUNK_CANDIDATES,
    DEFAULT_MAX_CHUNK_MORAS,
    DEFAULT_MAX_NODES_PER_SPAN,
    DEFAULT_MIN_CHUNK_SCORE,
    DEFAULT_NODE_BUDGET,
)
from .search import (
    DEFAULT_CANDIDATES,
    DEFAULT_PRESET,
    PRESETS,
    PhoneticSearcher,
    compare_pronunciations,
)
from .store import INNER_PRODUCT_SPACES, PhoneticStore, default_store_path

STATIC_DIR = Path(__file__).parent / "static"

#: 1 リクエストで返す件数の上限。件数に比例して `store.entry()` の呼び出しと
#: JSON のシリアライズが増える (2000 件で 40ms)。limit=0 で無制限にでき、
#: そのときは MAX_UNLIMITED で切る。
MAX_LIMIT = 200

#: limit=0 (無制限) のときに実際に返す件数の上限。5 モーラの全走査は
#: 30 万語が母集団になるので、min_score を付けずに無制限を要求されると
#: レスポンスが数百 MB になる。ブラウザが受け取れる量で切り、切ったことは
#: `truncated` で伝える。本当に全件が要るなら CLI を使う。
MAX_UNLIMITED = 5_000

#: ANN から取る候補数の上限。ここを大きくすると再現率と引き換えに
#: rerank のコストが線形に増える (DEFAULT_CANDIDATES の項も参照)。
MAX_CANDIDATES = 50_000

#: 分割合成が 1 回に返す候補数の上限。
#:
#: 候補 1 件が区間ごとの内訳を持つので、JSON が件数の数倍の速さで膨らむ。
#: 合成は 1 件あたり数百 ms かかる経路でもあり、無制限を露出する意味がない。
MAX_PHRASE_LIMIT = 50

#: ラティス表示のノード数の上限。
#:
#: 予算に届くまでビーム幅を広げるので、大きくすると探索が伸びる
#: (`_LATTICE_MAX_BEAM` の 512 で打ち止まる)。200 は 14 モーラの入力でも
#: 画面に収まる限界の目安。
MAX_NODE_BUDGET = 200


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
        limit: int = Query(20, ge=0, le=MAX_LIMIT, description="0 で無制限"),
        preset: str = Query(DEFAULT_PRESET),
        candidates: int = Query(DEFAULT_CANDIDATES, ge=1, le=MAX_CANDIDATES),
        min_score: float = Query(0.0, ge=0.0, le=1.0),
        categories: str | None = Query(None, description="カンマ区切り"),
        min_mora: int | None = Query(None, ge=1, le=32, description="モーラ数の下限"),
        max_mora: int | None = Query(None, ge=1, le=32, description="モーラ数の上限"),
    ) -> dict[str, Any]:
        """音が近い語を返す。"""
        if preset not in PRESETS:
            raise HTTPException(
                status_code=400,
                detail=f"未知のプリセット '{preset}' (利用可能: {', '.join(sorted(PRESETS))})",
            )

        engine = searcher()
        # モーラ範囲を指定すると ANN を迂回して全走査するので、母集団の
        # 規模をレスポンスに含めて「なぜ遅いのか」を画面から読めるようにする。
        scanned: int | None = None
        if min_mora is not None or max_mora is not None:
            try:
                scanned = engine.mora_range_size(min_mora, max_mora)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None

        pronunciation, results = engine.search(
            q,
            limit=None if limit == 0 else limit,
            preset=preset,
            candidates=candidates,
            min_score=min_score,
            categories=_parse_categories(categories),
            min_mora=min_mora,
            max_mora=max_mora,
        )
        total = len(results)
        truncated = total > MAX_UNLIMITED
        if truncated:
            results = results[:MAX_UNLIMITED]
        return {
            "query": q,
            "reading": pronunciation.reading,
            "phonemes": list(pronunciation.phonemes),
            "ipa": ipa_transcription(pronunciation.phonemes),
            "mora_count": pronunciation.mora_count,
            "preset": preset,
            "scanned": scanned,
            "total": total,
            "truncated": truncated,
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
                    # 促音の重複は後続音素を見ないと書けないので、連続表記は
                    # JS に組ませずここで作る (`ipa_transcription` の docstring)。
                    "ipa": ipa_transcription(r.phonemes),
                }
                for r in results
            ],
        }

    @app.get("/api/phrase")
    def api_phrase(
        text: str = Query(..., min_length=1, description="入力 (漢字・かな・カタカナ)"),
        limit: int = Query(10, ge=1, le=MAX_PHRASE_LIMIT),
        max_chunk_moras: int = Query(DEFAULT_MAX_CHUNK_MORAS, ge=1, le=8),
        chunk_candidates: int = Query(DEFAULT_CHUNK_CANDIDATES, ge=1, le=64),
        beam_width: int = Query(DEFAULT_BEAM_WIDTH, ge=1, le=256),
        min_chunk_score: float = Query(DEFAULT_MIN_CHUNK_SCORE, ge=0.0, le=1.0),
        allow_particles: bool = Query(True),
    ) -> dict[str, Any]:
        """入力を「複数の語 + 助詞」の連なりに合成して返す (空耳の経路)。

        通常の `/api/similar` は 1 語を 1 語に写すので、長い入力には答えが
        返らない (音が近い単一の語が辞書に無い)。こちらは入力をモーラ境界で
        区間に切って繋ぐ (`phrase.py` 参照)。
        """
        pronunciation, candidates = searcher().compose(
            text,
            limit=limit,
            max_chunk_moras=max_chunk_moras,
            chunk_candidates=chunk_candidates,
            beam_width=beam_width,
            min_chunk_score=min_chunk_score,
            allow_particles=allow_particles,
        )
        return {
            "text": text,
            "reading": pronunciation.reading,
            "phonemes": list(pronunciation.phonemes),
            "ipa": ipa_transcription(pronunciation.phonemes),
            "mora_count": pronunciation.mora_count,
            # 入力側のモーラ列。区間の対応を画面に描くのに要る。
            "moras": [m.kana or m.special for m in pronunciation.moras],
            "total": len(candidates),
            "results": [
                {
                    "text": c.text,
                    "reading": c.reading,
                    "score": c.score,
                    "phonetic_similarity": c.phonetic_similarity,
                    "segment_count": c.segment_count,
                    "segments": [
                        {
                            "surface": s.surface,
                            "reading": s.reading,
                            "source_reading": s.source_reading,
                            "start": s.start,
                            "end": s.end,
                            "mora_count": s.mora_count,
                            "similarity": s.similarity,
                            "is_particle": s.is_particle,
                            "phonemes": list(s.phonemes),
                            # 促音の重複は後続音素を見ないと書けないのでここで作る。
                            "ipa": ipa_transcription(s.phonemes),
                        }
                        for s in c.segments
                    ],
                }
                for c in candidates
            ],
        }

    @app.get("/api/phrase/lattice")
    def api_phrase_lattice(
        text: str = Query(..., min_length=1, description="入力 (漢字・かな・カタカナ)"),
        node_budget: int = Query(DEFAULT_NODE_BUDGET, ge=2, le=MAX_NODE_BUDGET),
        max_nodes_per_span: int = Query(DEFAULT_MAX_NODES_PER_SPAN, ge=1, le=32),
        max_chunk_moras: int = Query(DEFAULT_MAX_CHUNK_MORAS, ge=1, le=8),
        chunk_candidates: int = Query(DEFAULT_CHUNK_CANDIDATES, ge=1, le=64),
        beam_width: int = Query(DEFAULT_BEAM_WIDTH, ge=1, le=256),
        min_chunk_score: float = Query(DEFAULT_MIN_CHUNK_SCORE, ge=0.0, le=1.0),
        allow_particles: bool = Query(True),
    ) -> dict[str, Any]:
        """合成の候補群を 1 枚の DAG (ラティス) に畳んで返す。

        候補を並べると同じ語が何度も出る (実測で区間の 65〜77% が重複し、
        「名前」「は」は上位 10 件の全部に現れた)。ノードに畳めば 1 度しか
        描かれず、分岐だけが見える。

        `node_budget` はノード数の予算。**届くまでビーム幅を広げる** ので、
        上位数件だけでは薄く全候補では埋まるという両極を避けられる
        (`phrase.PhraseComposer.lattice`)。

        一覧表示は `/api/phrase`。同じ経路集合の別の見せ方なので、
        画面はこの 2 つを切り替える。
        """
        pronunciation, lattice = searcher().lattice(
            text,
            node_budget=node_budget,
            max_nodes_per_span=max_nodes_per_span,
            max_chunk_moras=max_chunk_moras,
            chunk_candidates=chunk_candidates,
            beam_width=beam_width,
            min_chunk_score=min_chunk_score,
            allow_particles=allow_particles,
        )
        return {
            "text": text,
            "reading": pronunciation.reading,
            "ipa": ipa_transcription(pronunciation.phonemes),
            "mora_count": pronunciation.mora_count,
            # ノードをモーラ位置に並べるので、入力側のモーラ列が要る。
            "moras": [m.kana or m.special for m in pronunciation.moras],
            "node_count": lattice.node_count,
            "path_count": lattice.path_count,
            "beam_width": lattice.beam_width,
            # 予算のために経路を削ったか。画面に「もっとある」ことを出す。
            "truncated": lattice.truncated,
            "nodes": [
                {
                    "id": n.id,
                    "surface": n.surface,
                    "reading": n.reading,
                    "start": n.start,
                    "end": n.end,
                    "mora_count": n.mora_count,
                    "source_reading": n.source_reading,
                    "similarity": n.similarity,
                    "is_particle": n.is_particle,
                    "path_count": n.path_count,
                    "best_score": n.best_score,
                }
                for n in lattice.nodes
            ],
            "edges": [
                {"source": e.source, "target": e.target, "path_count": e.path_count}
                for e in lattice.edges
            ],
            # 図から一覧に戻れるように経路も返す。ノードを選んで絞り込むとき、
            # どの経路が残るかをフロントが計算できる。
            "paths": [
                {
                    "text": c.text,
                    "reading": c.reading,
                    "score": c.score,
                    "nodes": [f"{s.start}:{s.end}:{s.surface}" for s in c.segments],
                }
                for c in lattice.paths
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
            "ipa": ipa_transcription(pronunciation.phonemes),
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
                "ipa": ipa_transcription(comparison.a_phonemes),
            },
            "b": {
                "text": comparison.b_text,
                "reading": comparison.b_reading,
                "phonemes": list(comparison.b_phonemes),
                "ipa": ipa_transcription(comparison.b_phonemes),
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
            # 分割合成の既定値。フロントに固定表を持たせると phrase.py を
            # 変えたときに黙ってずれるので、サーバ側の値を配る。
            "phrase": {
                "max_chunk_moras": DEFAULT_MAX_CHUNK_MORAS,
                "chunk_candidates": DEFAULT_CHUNK_CANDIDATES,
                "beam_width": DEFAULT_BEAM_WIDTH,
                "min_chunk_score": DEFAULT_MIN_CHUNK_SCORE,
                "max_limit": MAX_PHRASE_LIMIT,
                "node_budget": DEFAULT_NODE_BUDGET,
                "max_node_budget": MAX_NODE_BUDGET,
            },
        }

    @app.get("/api/phonemes")
    def api_phonemes() -> dict[str, Any]:
        """音素の素性表と IPA 表記を返す。

        フロントはこれを使って音素チップの色と IPA を決める。どちらも UI 側の
        固定表に持たせると distance.py を変えたときに黙ってずれるので、
        素性と対応表そのものを配って JS に写させる。
        """
        return {
            "consonants": {
                symbol: {
                    "place": c.place,
                    "manner": c.manner,
                    "voiced": c.voiced,
                    "palatalized": c.palatalized,
                    "ipa": phoneme_ipa(symbol),
                }
                for symbol, c in CONSONANTS.items()
            },
            "vowels": {
                symbol: {
                    "height": v.height,
                    "backness": v.backness,
                    "rounded": v.rounded,
                    "ipa": phoneme_ipa(symbol),
                }
                for symbol, v in VOWELS.items()
            },
            "special": {
                symbol: {"label": label, "ipa": phoneme_ipa(symbol)}
                for symbol, label in (
                    (LONG, "長音"),
                    (GEMINATE, "促音"),
                    (MORAIC_N, "撥音"),
                )
            },
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

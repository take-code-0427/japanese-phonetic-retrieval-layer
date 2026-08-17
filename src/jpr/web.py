"""音韻検索の Web フロントエンド (HTTP API + 静的ページ)。

MCP が LLM 向けの窓なのに対し、こちらは人が音韻空間を直接覗くための窓。
索引を 1 度だけ mmap して常駐させるので、CLI のようにプロセス起動ごとの
Sudachi ロード (数百 ms) と索引オープンを払わずに済む。

API は CLI の `--json` 出力と同じ構造を返す。フロント側の JS が CLI 出力を
そのまま読める形にしておくと、curl でのデバッグと画面表示が食い違わない。
"""

from __future__ import annotations

import threading
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .distance import CONSONANTS, VOWELS, align_phonemes, phoneme_ipa
from .index import Category, parse_category_list
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
from .serialize import (
    comparison_payload,
    lattice_edge_payload,
    lattice_node_payload,
    phrase_candidate_payload,
    pronunciation_payload,
    search_result_payload,
)
from .store import CANDIDATE_SPACES, PhoneticStore, default_store_path

STATIC_DIR = Path(__file__).parent / "static"

#: 件数に上限を持たない。母集団を切るのは `min_score` の役目で、既定の
#: 画面 (スコア下限 0.8 / 件数無制限) は「基準を満たす語を全部見る」ための
#: ものなので、件数で切ると基準を満たした語が黙って消える。
#:
#: 既定経路では母集団が候補生成の Top-K (グループ) に限られるので、返る件数は
#: 行に展開した 1 万件程度が上界になる。**歯止めが効かないのは全走査経路のほう**
#: (5 モーラで 30 万語)。そこは `min_score` を 0 にすると数百 MB になり得るが、
#: 範囲指定は呼び出し側が意図を持って入れるものなので、件数で勝手に切らない。

#: 検索結果を返すスコアの下限。
#:
#: 件数ではなくスコアで切るのが既定。上位 N 件で切ると「N 件目と N+1 件目の
#: スコアが同じ」ときに片方だけが消え、順位の読み比べができなくなる。基準を
#: 決めて満たすものを全部出すほうが、何を見ているかが画面から読める。
#:
#: 0.7 は rerank 後のスコアで「音が近いと言える」線。裾まで含む緩めの基準。
#:
#: **v9 で 0.8 から下げた。** 一般性を Wikipedia の出現記事数に変えたとき
#: (`frequency`)、指標の尺度そのものが動いてスコアの絶対値が 0.05〜0.09 沈んだ。
#: 旧指標 (連接コストの反転) は既定カテゴリの平均が 0.8 付近だったのに対し、
#: 新指標は 0.26 — 頻度表に無い語が 8 割あり、そこが `UNKNOWN_FAMILIARITY`
#: (0.25) に寄るため。**順位は改善しているがスコアの絶対値は下がる**ので、
#: 0.8 のままだと「ラーメン」「電話」「パソコン」が 0 件になり、常に
#: フォールバック経路 (下の `FALLBACK_LIMIT`) に落ちていた。
#:
#: 実測の該当件数 (>= 0.7): ラーメン 70 / 乳首 39 / 科学 186 / 電話 42 /
#: りんご 63 / 東京 279 / パソコン 22 / 明日 291。
#:
#: **スコアの重み構成を変えたらここも測り直す。** 下限は絶対値の基準なので、
#: 成分の尺度が動くと意味が変わる。
DEFAULT_MIN_SCORE = 0.7

#: 下限を満たす語が 1 件も無いときに、下限を外して返す件数。
#:
#: ここだけは件数で切る。下限を外した以上「基準を満たす全部」という意味が
#: 無いので、切る位置に根拠を求められない。20 は最も近い語がどれかを読むのに
#: 足りる数。
FALLBACK_LIMIT = 20

#: 候補生成から取る件数の上限。ここを大きくすると rerank のコストが線形に
#: 増える (DEFAULT_CANDIDATES の項も参照)。内積は全行に対して取るので、
#: 増やしても候補生成側のコストは変わらない。
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

#: モーラ範囲の全走査を同時に走らせる本数。
#:
#: 全走査は母集団ぶんの一時配列を並行数だけ抱えるので、匿名メモリが並行度に
#: 比例して伸びる。full (202 万語) の実測ピークは 2 並行 +190MB・5 並行
#: +357MB・10 並行 +613MB。2 なら常駐 260MB と合わせて 451MB で収まる。
#:
#: 通常検索は門を通さない (10 並行 x30 回でも匿名メモリは +60MB 程度)。
#: 待たせるほうが落とすよりましだが、待たせる相手は重い経路だけに限る。
MAX_CONCURRENT_SCANS = 2

#: 全走査の順番待ちを諦める秒数。
#:
#: 待ち行列が伸びたときに無制限に待たせるとクライアントのタイムアウトと
#: 二重待ちになるので、503 で明示的に断る。
SCAN_QUEUE_TIMEOUT = 30.0


def _parse_categories(value: str | None) -> list[Category] | None:
    """カンマ区切りのカテゴリ名を Category に変換する。"""
    try:
        return parse_category_list(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


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

    # 全走査の同時実行を絞る門 (`MAX_CONCURRENT_SCANS` 参照)。エンドポイントは
    # 非 async なのでスレッドプールで動く。threading の semaphore で足りる。
    scan_gate = threading.Semaphore(MAX_CONCURRENT_SCANS)

    @contextmanager
    def limit_scans():
        """モーラ範囲の全走査を `MAX_CONCURRENT_SCANS` 本までに絞る。

        取れなければ 503 で断る。待ち続けるとクライアント側のタイムアウトと
        二重待ちになり、どちらが原因か画面から読めなくなる。
        """
        if not scan_gate.acquire(timeout=SCAN_QUEUE_TIMEOUT):
            raise HTTPException(
                status_code=503,
                detail=(
                    "モーラ範囲の全走査が混み合っています "
                    f"(同時 {MAX_CONCURRENT_SCANS} 本まで)。時間をおいて再試行してください。"
                ),
            )
        try:
            yield
        finally:
            scan_gate.release()

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
        limit: int = Query(0, ge=0, description="0 で無制限"),
        preset: str = Query(DEFAULT_PRESET),
        candidates: int = Query(DEFAULT_CANDIDATES, ge=1, le=MAX_CANDIDATES),
        min_score: float = Query(DEFAULT_MIN_SCORE, ge=0.0, le=1.0),
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
        # モーラ範囲を指定すると Top-K ではなく範囲の全行を見るので、母集団の
        # 規模をレスポンスに含めて「なぜ遅いのか」を画面から読めるようにする。
        scanned: int | None = None
        scanning = min_mora is not None or max_mora is not None
        if scanning:
            try:
                scanned = engine.mora_range_size(min_mora, max_mora)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from None

        # 全走査は母集団ぶんの一時配列を作るので、並行で走らせるとメモリが
        # 溢れる (`MAX_CONCURRENT_SCANS` 参照)。Top-K の経路は軽いので
        # 門を通さない。
        parsed_categories = _parse_categories(categories)
        with limit_scans() if scanning else nullcontext():
            pronunciation, results = engine.search(
                q,
                limit=None if limit == 0 else limit,
                preset=preset,
                candidates=candidates,
                min_score=min_score,
                categories=parsed_categories,
                min_mora=min_mora,
                max_mora=max_mora,
            )
            # スコアの下限は絶対的な基準として使えない。**スコアはクエリの
            # 長さに依存する** — 長い語ほど完全一致でない限り編集距離の減点が
            # 積み上がるので、既定の 0.8 に最近傍すら届かないクエリが実在する
            # (実測で「わたしのなまえ」7 拍の最高が 0.791、「ありがとうござい
            # ます」10 拍で 0.807)。件数で切らない以上そのまま空の画面になる。
            #
            # 下限を長さで動かす案は取らない — 「0.8 以上」の意味が
            # クエリごとに変わると、画面に出た数値が何を表すのか読めなくなる。
            # 基準は固定したまま、満たす語が無いときだけ下限を外して近い順に
            # 見せ、外したことを `below_floor` で伝える。
            fell_back = not results and min_score > 0.0
            if fell_back:
                # 件数は呼び出し側の `limit` を優先する。`FALLBACK_LIMIT` は
                # 「無制限 (limit=0) で来たときに何件で切るか」であって、
                # 明示された上限を上書きする理由がない。
                _, results = engine.search(
                    q,
                    limit=FALLBACK_LIMIT if limit == 0 else min(limit, FALLBACK_LIMIT),
                    preset=preset,
                    candidates=candidates,
                    categories=parsed_categories,
                    min_mora=min_mora,
                    max_mora=max_mora,
                )
        return {
            "below_floor": fell_back,
            "query": q,
            **pronunciation_payload(pronunciation),
            "preset": preset,
            "scanned": scanned,
            "results": [search_result_payload(r) for r in results],
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
            **pronunciation_payload(pronunciation),
            "total": len(candidates),
            "results": [phrase_candidate_payload(c) for c in candidates],
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
            **pronunciation_payload(pronunciation),
            "node_count": lattice.node_count,
            "path_count": lattice.path_count,
            "beam_width": lattice.beam_width,
            # 予算のために経路を削ったか。画面に「もっとある」ことを出す。
            "truncated": lattice.truncated,
            "nodes": [lattice_node_payload(n) for n in lattice.nodes],
            "edges": [lattice_edge_payload(e) for e in lattice.edges],
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
            **pronunciation_payload(pronunciation),
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
        return comparison_payload(comparison)

    @app.get("/api/info")
    def api_info() -> dict[str, Any]:
        """索引のメタ情報。フロントの起動時にプリセットとカテゴリを引くのにも使う。"""
        store = searcher().store
        counts = store.category_counts()
        return {
            "path": str(store.path),
            "format_version": store.meta.version,
            "count": len(store),
            "spaces": [
                {
                    "name": name,
                    "dim": dim,
                    # 候補生成は 1 空間との内積で行う。他は rerank でスコアを足すだけ。
                    "role": "候補生成 + rerank" if name in CANDIDATE_SPACES else "rerank のみ",
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
            "default_min_score": DEFAULT_MIN_SCORE,
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

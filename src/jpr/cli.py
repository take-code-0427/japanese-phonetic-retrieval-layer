"""jpr のコマンドラインインターフェース。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .distance import ipa_transcription
from .index import Category
from .phonology import analyze_reading
from .search import DEFAULT_CANDIDATES, DEFAULT_PRESET, PRESETS, PhoneticSearcher
from .store import INNER_PRODUCT_SPACES, PhoneticStore, default_store_path


def _add_store_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--index",
        type=Path,
        default=default_store_path(),
        metavar="DIR",
        help="索引ディレクトリ (既定: %(default)s)",
    )


def _open_searcher(path: Path) -> PhoneticSearcher:
    try:
        store = PhoneticStore(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    return PhoneticSearcher(store)


def _print_json(payload: object) -> None:
    """CLI の JSON 出力。日本語をエスケープせずに整形して出す。"""
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_categories(value: str | None) -> list[Category] | None:
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
            print(f"エラー: 未知のカテゴリ '{name}' (利用可能: {valid})", file=sys.stderr)
            raise SystemExit(1) from None
    return result or None


def cmd_similar(args: argparse.Namespace) -> int:
    searcher = _open_searcher(args.index)
    limit = None if args.limit == 0 else args.limit

    # モーラ範囲を指定すると ANN を使わず全走査する。実測で 1 モーラ長あたり
    # 1 秒以上かかるので、待たせる前に規模を伝える。
    scanned: int | None = None
    if args.min_mora is not None or args.max_mora is not None:
        try:
            scanned = searcher.mora_range_size(args.min_mora, args.max_mora)
        except ValueError as exc:
            print(f"エラー: {exc}", file=sys.stderr)
            return 1
        if not args.json:
            print(f"モーラ範囲を指定 — {scanned:,} 語を全走査します", file=sys.stderr)

    started = time.perf_counter()
    try:
        pronunciation, results = searcher.search(
            args.query,
            limit=limit,
            preset=args.preset,
            candidates=args.candidates,
            min_score=args.min_score,
            categories=_parse_categories(args.categories),
            min_mora=args.min_mora,
            max_mora=args.max_mora,
        )
    except ValueError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    elapsed = (time.perf_counter() - started) * 1000

    if args.json:
        _print_json(
            {
                "query": args.query,
                "reading": pronunciation.reading,
                "phonemes": list(pronunciation.phonemes),
                "ipa": ipa_transcription(pronunciation.phonemes),
                "mora_count": pronunciation.mora_count,
                "preset": args.preset,
                "elapsed_ms": round(elapsed, 1),
                # 全走査したときだけ、母集団の規模を添える。
                "scanned": scanned,
                "result_count": len(results),
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
                        "ipa": ipa_transcription(r.phonemes),
                    }
                    for r in results
                ],
            }
        )
        return 0

    # IPA は角括弧に入れる (音声表記の慣用)。既存の `[音素列]` と衝突するので、
    # 抽象的な音素列のほうを本来の音素表記の記法 // に移す。
    print(
        f"{args.query} -> {pronunciation.reading} "
        f"/{pronunciation.phoneme_string()}/ "
        f"[{ipa_transcription(pronunciation.phonemes)}] "
        f"{pronunciation.mora_count} モーラ  ({elapsed:.0f}ms, preset={args.preset})"
    )
    if not results:
        print("該当なし")
        return 0

    # モーラ範囲で絞った結果を読むにはモーラ数が要る。既定の検索でも
    # 候補がどの長さに寄っているかが見えるので、常に出す。
    print(f"\n{'score':>6}  {'音韻':>5}  {'語尾':>5}  {'拍':>3}  {'語':<16} {'読み':<14} カテゴリ")
    for result in results:
        print(
            f"{result.score:6.3f}  "
            f"{result.phonetic_similarity:5.3f}  "
            f"{result.coda_similarity:5.3f}  "
            f"{result.mora_count:3d}  "
            f"{result.surface:<16} {result.reading:<14} {result.category.value}"
        )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """2 語を比較する。索引を必要とせず、辞書だけで完結する。"""
    from .reading import ReadingExtractor
    from .search import compare_pronunciations

    extractor = ReadingExtractor(dict_type=args.dict)
    comparison = compare_pronunciations(
        args.a,
        args.b,
        analyze_reading(extractor.reading_of(args.a)),
        analyze_reading(extractor.reading_of(args.b)),
    )

    if args.json:
        _print_json(
            {
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
        )
        return 0

    for text, reading, phonemes in (
        (comparison.a_text, comparison.a_reading, comparison.a_phonemes),
        (comparison.b_text, comparison.b_reading, comparison.b_phonemes),
    ):
        print(f"{text} -> {reading} /{' '.join(phonemes)}/ [{ipa_transcription(phonemes)}]")
    print(f"\n音韻類似度: {comparison.similarity:.4f}  (編集距離 {comparison.distance:.3f})")
    print("\n空間別:")
    for name, value in comparison.spaces.items():
        print(f"  {name:<10} {value:.4f}")
    return 0


def cmd_pronounce(args: argparse.Namespace) -> int:
    """読みと音素列だけを表示する。索引を必要としない。"""
    from .reading import ReadingExtractor

    extractor = ReadingExtractor(dict_type=args.dict)
    for text in args.text:
        pronunciation = analyze_reading(extractor.reading_of(text))
        moras = " ".join(m.kana or m.special for m in pronunciation.moras)
        print(
            f"{text} -> {pronunciation.reading} "
            f"/{pronunciation.phoneme_string()}/ "
            f"[{ipa_transcription(pronunciation.phonemes)}] "
            f"モーラ: {moras} ({pronunciation.mora_count})"
        )
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    from .build import build_index

    path = args.index
    if path.exists() and not args.force:
        print(
            f"エラー: 索引が既に存在します: {path}\n上書きするには --force を指定してください。",
            file=sys.stderr,
        )
        return 1

    started = time.perf_counter()

    def report(message: str) -> None:
        print(f"[{time.perf_counter() - started:6.0f}s] {message}", flush=True)

    count = build_index(
        path,
        dict_type=args.dict,
        min_mora=args.min_mora,
        max_mora=args.max_mora,
        progress=report,
    )
    print(f"\n{count:,} 語を索引しました: {path}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    try:
        store = PhoneticStore(args.index)
    except (FileNotFoundError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    print(f"索引: {store.path}")
    print(f"形式バージョン: {store.meta.version}")
    print(f"辞書: SudachiDict {store.meta.dict_type}")
    print(f"語数: {len(store):,}")
    print("\n埋め込み空間:")
    for name, dim in store.meta.dims.items():
        # 候補生成に使う空間のみ ANN を張る。他は rerank でベクトルを直接引く。
        role = "ANN + rerank" if name in INNER_PRODUCT_SPACES else "rerank のみ"
        print(f"  {name:<10} {dim:>4} 次元  ({role})")

    print("\nカテゴリ:")
    counts = store.category_counts()
    for category, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {category.value:<10} {count:>10,}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .mcp import serve

    serve(args.index)
    return 0


def cmd_serve_web(args: argparse.Namespace) -> int:
    from .web import serve_web

    print(f"http://{args.host}:{args.port}  (索引: {args.index})")
    serve_web(args.index, host=args.host, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jpr",
        description="日本語の音韻検索レイヤー (Japanese Phonetic Retrieval)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    similar = subparsers.add_parser("similar", help="音が近い語を検索する")
    similar.add_argument("query", help="検索語 (漢字・かな・カタカナ)")
    similar.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        help="結果数。0 で無制限 (--min-score と併せて使う) (既定: %(default)s)",
    )
    similar.add_argument(
        "-p",
        "--preset",
        default=DEFAULT_PRESET,
        choices=sorted(PRESETS),
        help="スコアのプリセット (既定: %(default)s)",
    )
    similar.add_argument(
        "-k",
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATES,
        help="ANN から取る候補数 (既定: %(default)s)",
    )
    similar.add_argument("--min-score", type=float, default=0.0, help="スコアの下限")
    similar.add_argument(
        "-c",
        "--categories",
        help="カテゴリを絞る (カンマ区切り: common,product,person,place,other)",
    )
    similar.add_argument(
        "--min-mora",
        type=int,
        help="検索するモーラ数の下限。指定すると ANN を使わず全走査する (数秒かかる)",
    )
    similar.add_argument(
        "--max-mora",
        type=int,
        help="検索するモーラ数の上限。指定すると ANN を使わず全走査する (数秒かかる)",
    )
    similar.add_argument("--json", action="store_true", help="JSON で出力する")
    _add_store_argument(similar)
    similar.set_defaults(func=cmd_similar)

    compare = subparsers.add_parser("compare", help="2 語の音韻類似度を計算する")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.add_argument("--dict", default="full", help="SudachiDict の種類 (既定: %(default)s)")
    compare.add_argument("--json", action="store_true", help="JSON で出力する")
    compare.set_defaults(func=cmd_compare)

    pronounce = subparsers.add_parser("pronounce", help="読みと音素列を表示する")
    pronounce.add_argument("text", nargs="+")
    pronounce.add_argument("--dict", default="full", help="SudachiDict の種類 (既定: %(default)s)")
    pronounce.set_defaults(func=cmd_pronounce)

    build = subparsers.add_parser("build-index", help="索引を構築する")
    build.add_argument("--dict", default="full", help="SudachiDict の種類 (既定: %(default)s)")
    build.add_argument("--min-mora", type=int, default=2, help="索引する最小モーラ数")
    build.add_argument("--max-mora", type=int, default=12, help="索引する最大モーラ数")
    build.add_argument("--force", action="store_true", help="既存の索引を上書きする")
    _add_store_argument(build)
    build.set_defaults(func=cmd_build_index)

    info = subparsers.add_parser("info", help="索引の情報を表示する")
    _add_store_argument(info)
    info.set_defaults(func=cmd_info)

    serve = subparsers.add_parser("serve", help="MCP サーバとして起動する")
    _add_store_argument(serve)
    serve.set_defaults(func=cmd_serve)

    serve_web = subparsers.add_parser("serve-web", help="Web フロントを起動する")
    serve_web.add_argument(
        "--host", default="127.0.0.1", help="待ち受けアドレス (既定: %(default)s)"
    )
    serve_web.add_argument("--port", type=int, default=8000, help="ポート (既定: %(default)s)")
    _add_store_argument(serve_web)
    serve_web.set_defaults(func=cmd_serve_web)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

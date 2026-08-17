"""検索結果を JSON に載る形へ写す。

**窓 (`cli.py --json` / `mcp.py` / `web.py`) が結果の形を決めない。** 3 者とも
同じ `SearchResult` / `PhraseCandidate` を返しているのに、以前はそれぞれの
出口が辞書リテラルを書いていた。結果として MCP だけ `ipa` と `pos` が落ち、
区間の位置が `mora_range: [start, end]` と `start` / `end` に分かれ、比較の
キーが `phonetic_similarity` と `similarity` に割れていた — **同じ検索を
違う窓から引くと違うフィールドが返っていた**。

窓ごとに違ってよいのは「1 回に何件返すか」(MCP は `_MAX_LIMIT` で切る)
や、その窓にしか意味のない項目 (CLI の `elapsed_ms`、Web の `below_floor`、
MCP の `note`) のような**呼び出しの制約**であって、1 件をどう表すかではない。
前者は窓が決め、後者はここが決める。

`tests/test_serialize.py` が窓どうしを突き合わせるので、片方にだけ
フィールドを足すと落ちる。

`ipa` をここで作るのは、促音の重複が後続音素を見ないと書けないため
(`distance.ipa_transcription` の docstring)。JS や LLM に組ませると
音素チップの色と同じ問題が起きる — 素性表を変えたときに表記だけが黙って
古いまま残る。
"""

from __future__ import annotations

from typing import Any

from .distance import ipa_transcription
from .phonology import Pronunciation
from .phrase import LatticeEdge, LatticeNode, PhraseCandidate, PhraseSegment
from .search import ComparisonResult, SearchResult


def pronunciation_payload(pronunciation: Pronunciation) -> dict[str, Any]:
    """クエリ側の音韻表現。全エンドポイントの応答に共通で載る。

    `moras` は入力側のモーラ列。区間の対応を画面に描くのに要る
    (`static/app.js` の分割合成ビュー) が、LLM が読んでも区間の
    `source_reading` と重複しないので両方の窓に出す。
    """
    return {
        "reading": pronunciation.reading,
        "phonemes": list(pronunciation.phonemes),
        "ipa": ipa_transcription(pronunciation.phonemes),
        "mora_count": pronunciation.mora_count,
        "moras": [m.kana or m.special for m in pronunciation.moras],
    }


def search_result_payload(result: SearchResult) -> dict[str, Any]:
    """検索結果 1 件。

    スコアの内訳を全部返すのは「なぜ近いと判断したか」を呼び出し側が検算
    できるようにするため (`SearchResult` の docstring)。LLM も画面と同じ
    根拠を読めるべきなので、窓で削らない。
    """
    return {
        "word": result.surface,
        "reading": result.reading,
        "score": result.score,
        "phonetic_similarity": result.phonetic_similarity,
        "embedding_similarity": result.embedding_similarity,
        "coda_similarity": result.coda_similarity,
        "vowel_similarity": result.vowel_similarity,
        # クエリの音が完全な形で入っているか (0 なら入っていない)。
        # 値はクエリが候補の音素列に占める割合で、余分が多いほど低い。
        "containment": result.containment,
        "mora_count": result.mora_count,
        "category": result.category.value,
        "pos": result.pos,
        "familiarity": result.familiarity,
        "phonemes": list(result.phonemes),
        "ipa": ipa_transcription(result.phonemes),
    }


def segment_payload(segment: PhraseSegment) -> dict[str, Any]:
    """合成結果の 1 区間。

    位置は `start` / `end` の 2 つのキーで返す。MCP だけ
    `mora_range: [start, end]` という別の形にしていたが、同じ値を 2 通りに
    書き分ける理由がない。
    """
    return {
        "surface": segment.surface,
        "reading": segment.reading,
        # 入力側のこの区間の読み。「どこが」を示すのに要る。
        "source_reading": segment.source_reading,
        "start": segment.start,
        "end": segment.end,
        "mora_count": segment.mora_count,
        "similarity": segment.similarity,
        "is_particle": segment.is_particle,
        "phonemes": list(segment.phonemes),
        "ipa": ipa_transcription(segment.phonemes),
    }


def phrase_candidate_payload(candidate: PhraseCandidate) -> dict[str, Any]:
    """合成結果 1 件 (語 + 助詞の連なり)。"""
    return {
        "text": candidate.text,
        "reading": candidate.reading,
        "score": candidate.score,
        "phonetic_similarity": candidate.phonetic_similarity,
        "segment_count": candidate.segment_count,
        "segments": [segment_payload(s) for s in candidate.segments],
    }


def comparison_payload(comparison: ComparisonResult) -> dict[str, Any]:
    """2 語の音韻比較。

    `spaces` は空間ごとの内訳で、「どこが似ているのか」を判断するのに要る。
    `rhythm` だけは正規化していないので距離から作った値が入る
    (`search.compare_pronunciations`)。
    """
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


def lattice_node_payload(node: LatticeNode) -> dict[str, Any]:
    """ラティスの 1 ノード。"""
    return {
        "id": node.id,
        "surface": node.surface,
        "reading": node.reading,
        "start": node.start,
        "end": node.end,
        "mora_count": node.mora_count,
        "source_reading": node.source_reading,
        "similarity": node.similarity,
        "is_particle": node.is_particle,
        "path_count": node.path_count,
        "best_score": node.best_score,
    }


def lattice_edge_payload(edge: LatticeEdge) -> dict[str, Any]:
    """ラティスの 1 辺。"""
    return {"source": edge.source, "target": edge.target, "path_count": edge.path_count}


__all__ = [
    "comparison_payload",
    "lattice_edge_payload",
    "lattice_node_payload",
    "phrase_candidate_payload",
    "pronunciation_payload",
    "search_result_payload",
    "segment_payload",
]

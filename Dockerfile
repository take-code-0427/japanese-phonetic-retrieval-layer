# 索引をイメージに焼き込む。Fly の Volume に置く案も採れるが、それだと
# 「初回に索引を用意する」手順がデプロイと別に要る。ビルド時に構築すれば
# デプロイした時点で動く状態になる。
#
# 辞書は core を使う。実測で品質は保たれる
# (「乳首」-> 手首 / 「ワタシノナマエハ」-> 私の名前は)。
# ANN のグラフを作らなくなったので構築は 496s -> 163s に縮んでいる
# (`store.py` の CANDIDATE_SPACES と `search.py` の _top_candidates を参照)。
#
# 索引サイズの推移 (core / full):
#   855MB / 1.64GB  ->  int8 量子化 (`store.py` の `_quantize`)
#   274MB / 508MB   ->  音素列グループ化 + 索引 3 空間化 (v5)
#   148MB / 322MB
#
# **`sudachidict-full` を入れない。** 実行時にどちらの辞書を読むかは索引の
# `dict_type` が決めるので (`PhoneticSearcher.extractor`)、core の索引で動かす
# このイメージでは full が一度も読まれない。それでいて 344MB あり、索引より
# 大きい死荷重になっていた。任意の依存にしてあるので `--extra full` を
# 付けなければ入らない (`pyproject.toml`)。
#
# イメージは 641MB (実測)。内訳は venv 363MB (うち sudachidict-core 208MB) +
# 索引 142MB + Python の基底。**次に大きいのは Sudachi の core 辞書**で、
# 読みの取得に要るので外せない。

FROM python:3.12-slim AS builder

# Rust ツールチェイン。`jpr.distance` が読む `jpr_distance` を maturin が
# ビルドするので、拡張の依存として必須 (無いと import に失敗する)。
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 依存の解決を先に済ませる。ソースだけを後から入れると、コードを触った
# ときに Rust と wheel の再ビルドを避けられる。
# LICENSE は `project.license-files` が参照するので、無いとビルドが
# メタデータ検証で落ちる。
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY rust/ ./rust/
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/
RUN uv sync --frozen --no-dev

# 索引の構築。sudachidict-full も依存に入っているが、ここでは core だけを
# 使う (上のコメント参照)。
ENV JPR_INDEX=/opt/jpr-index
RUN uv run jpr build-index --dict core --index "$JPR_INDEX"


FROM python:3.12-slim

# 実行時は Rust ツールチェインが要らない (ビルド済みの .so を持ち込む)。
# libgomp は numpy が参照する。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/jpr-index /opt/jpr-index
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    JPR_INDEX=/opt/jpr-index

EXPOSE 8080

# 0.0.0.0 で待つ (既定の 127.0.0.1 だとコンテナ外から届かない)。
CMD ["jpr", "serve-web", "--host", "0.0.0.0", "--port", "8080", "--index", "/opt/jpr-index"]

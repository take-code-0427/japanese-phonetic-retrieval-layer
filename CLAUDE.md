# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクトの目的

日本語の「音韻空間」を検索レイヤーとして提供する (Phonetic RAG)。semantic embedding が
意味の近さを引くのに対し、ここは **発音の近さ** だけを引く。意味と音は独立した軸として
扱い、混ぜない。音韻検索は候補を出すところまでを担い、意味による最終選択は呼び出し側
(LLM) が行う — この分担が設計上の中心的な前提。詳細な背景と実測値は `README.md`。

## コマンド

```bash
uv sync
uv run pytest                              # テスト
uv run pytest tests/test_search.py -k dedupe  # 単体テスト 1 つ
uv run ruff check .                        # lint (line-length 100, RUF001-003 は無効)
uv run jpr build-index                     # 索引構築。full 辞書で約 10 分 / 2.6GB
uv run jpr build-index --dict core --force # 小さい辞書で素早く作り直す
uv run jpr info                            # 索引の語数・空間・カテゴリ内訳
uv run jpr serve                           # MCP サーバ (stdio)
uv run jpr serve-web                       # Web フロント (http://127.0.0.1:8000)
```

索引は `~/.cache/jpr/index` (`default_store_path()`)。`--index DIR` で切り替えられる。

`tests/test_acceptance.py` は実辞書の索引を要求し、未構築なら自動スキップされる
(`conftest.py` の `real_store` フィクスチャ)。それ以外のテストは `sample_store` の
小さな語彙 (18 語) で完結するので索引なしで走る。**受入テストがスキップされている状態を
「通った」と見なさないこと** — 検索品質の回帰はここでしか捕まらない。

`compare` と `pronounce` は索引を必要とせず、Sudachi 辞書だけで動く。

## アーキテクチャ

オフライン (索引構築) とオンライン (検索) の 2 経路があり、両者は `store.py` の
ファイル形式と `embedding.py` の `SPACES` 定義だけで繋がる。

```
オフライン: dictionary.py → build.py → embedding.py → store.py(write_store)
            system.dic を直接パース  語彙選別   5 空間ベクトル  npz + npy + HNSW

オンライン: reading.py → phonology.py → embedding.py → search.py
            Sudachi で読み  音素/モーラ列  クエリを埋め込み  ANN → rerank
```

検索エンジンには 3 つの窓がある。いずれも `PhoneticSearcher` を呼ぶだけで、
検索のロジックは持たない。

- `cli.py` — 人が端末から。プロセス起動ごとに Sudachi と索引を開く。
- `mcp.py` — LLM から (stdio)。
- `web.py` — 人がブラウザから。HTTP API + `static/` の素の HTML/JS。
  索引を 1 度だけ mmap して常駐するので CLI のコールドスタートを払わない。

### 2 段検索 (`search.py`)

1. **ANN 候補生成** — HNSW (`phonetic` 空間) で Top-K。既定 K=5000。速度をここで稼ぐ。
2. **rerank** — weighted phonetic edit distance + 語尾・母音・モーラ数・一般性の
   重み付き和。精度をここで稼ぐ。

embedding 単独に任せないのは「なぜ近いのか」が曖昧になるため。最終スコアの重みは
用途別プリセット (`pun` / `rhyme` / `mishearing`) で切り替える。

rerank は編集距離を除いた成分でスコアの上限 (`reachable`) を先に出し、確定済み上位の
最低スコアを打ち切り線にして見込みのない候補には距離を計算しない。この最適化があるため、
**スコアの重み構成を変えるときは上限計算 (`weights.phoneme` を足す部分) の妥当性も
併せて確認する必要がある**。

### 5 つの埋め込み空間 (`embedding.py`)

`phonetic` / `consonant` / `vowel` / `coda` / `rhythm`。用途で「近い」の定義が変わるので
単一ベクトルに混ぜない。ニューラルな学習は行わず、音声学的素性から決定的に構成する。

- `phonetic`, `consonant`, `vowel`, `coda` は L2 正規化済み → 内積 = コサイン類似度。
- `rhythm` だけは正規化しない (モーラ数の絶対量が意味を持つ)。比較はユークリッド距離。
  `compare_pronunciations` がこの空間を特別扱いしているのはそのため。
- ANN 索引を張るのは `INNER_PRODUCT_SPACES` = `("phonetic", "coda")` のみ。1 空間で
  1.2GB (202 万語) かかるので、rerank だけに使う空間はベクトル行列で足りる。

位置情報は等間隔ビンのマルチスケールプーリング (`POSITION_SCALES = (1, 2)`) で保つ。
**ここを細かくしてはいけない** — 4 ビンにすると語頭 1 音素の差でビンが丸ごと外れ、
「チクビ」と「テクビ」の類似度が 0.68 まで落ちて ANN 候補にすら入らなくなった。
細かい位置合わせは rerank 段の編集距離が担うので、候補生成側は粗くてよい。

### 音素距離 (`distance.py`)

音素を記号ではなく素性ベクトル (調音位置・方法・有声性・口蓋化 / 母音の高さ・前後・円唇)
として扱い、素性差から距離を導く。同じ距離関数に 2 つの実装がある:

- `weighted_edit_distance(tuple[str], tuple[str])` — 記号ベース。可読性優先。
- `edit_distance_ids(np.ndarray, np.ndarray)` — ID ベース。rerank の内側で使う高速経路。
  置換コストを `SUBSTITUTION_COSTS` (P, P) 行列に事前計算し、DP の行を NumPy 演算にする。

**両者は同じ値を返さなければならない** (`tests/test_distance.py` が対応を検証する)。
距離の重みや素性表を変えるときは両方に効くことを確認する。`_worst_substitution_cost()`
は素性表から実測するので、重みを変えても類似度の正規化は自動追従する。

### 索引形式 (`store.py`)

`FORMAT_VERSION` を上げる変更 (エントリの持ち方・空間の追加削除) をしたら、既存の索引は
`jpr build-index --force` で作り直す必要がある。`PhoneticStore.__init__` がバージョン
不一致を検出してエラーにする。

pickle を使わない理由は 2 つ — ロードに 12〜18 秒かかり MCP サーバの起動コストとして
重すぎたこと、任意コード実行を招く形式を配布物に使いたくないこと。NumPy の mmap なら
実質 0 秒。文字列は固定幅 unicode 配列 (`<U27`) を避け、UTF-8 バイト列 + 境界インデックス
(CSR 風) で持つ (`<U27` は最長要素の幅で全行を埋め、202 万語の表層だけで 218MB 消費した)。

200 万件を起動時に materialize しない。`entry(row)` は必要になった行だけ Python
オブジェクトに起こす。フィルタは `category_ids` / `mora_counts` / `costs` の配列を
直接引いてベクトル化する (`_apply_cheap_filters`)。

### 辞書パース (`dictionary.py`)

SudachiPy には語彙列挙 API が無く (`lookup` は完全一致のみ)、Java 版の `dump` も
提供されないため `system.dic` のバイナリを直接パースしている。`_validate_offset_table()`
がオフセットテーブルの整合性を起動時に検証するので、SudachiDict のフォーマットが
変わればそこで失敗する。**このガードを外さないこと** — 読み違いが黙って通ると
索引全体が静かに壊れる。

### カテゴリ (`index.py`)

SudachiDict full は地名 49 万・人名 24 万を含み索引の 7 割を占める。人名の異表記は
音韻的に密集するので、`DEFAULT_CATEGORIES` は `person` / `place` を除く。

`familiarity` は Sudachi の連接コストを反転した「知られた語らしさ」の弱い指標。厳密な
頻度ではないので、順位の同点をほどく程度の重みしか掛けない (`PRESETS` の `familiarity`)。

## 実装上の注意

- **同音異表記の代表選び** (`_representative_rank`): SudachiDict では読みをそのまま
  書いたカタカナ見出し (「カカク」cost 3269) が漢字表記 (「価格」cost 5496) より低コスト
  なことがある。コストだけで選ぶと「科学」の近傍が「カカク」になり語として何を指すのか
  読み取れないので、見出しの情報量 (`_surface_informativeness`) をスコアより先に見る。
- **`DEFAULT_CANDIDATES = 5000`**: 400 に絞れば 16ms まで落ちるが、202 万語では
  コサイン 0.91 以上の語が 400 件を超えるため「乳首」に対する「手首」のような明らかな
  近傍が漏れる。品質を優先している。
- **`ReadingExtractor` のキャッシュはインスタンス単位**: メソッドに `lru_cache` を付けると
  クラス単位のキャッシュが `self` を握り続けインスタンスが解放されなくなる。
- **Sudachi のロードは重い**: `PhoneticSearcher.extractor` は初回参照まで遅延させている。
  索引 (mmap) も MCP サーバ・Web サーバでは最初のリクエストまで開かない。実測で
  初回 12.5 秒、2 回目以降は 34ms〜1 秒。

### Web フロント (`web.py` + `static/`)

- **ビルド工程を持たない**: 素の HTML/CSS/JS を `static/` に置き、FastAPI が配る。
  Node の依存とビルド手順を増やすほどの UI ではない。
- **音素の色は素性表から決める** (`/api/phonemes`): 子音は調音位置、母音は舌の位置で
  色相が決まる。UI 側に色の固定表を持つと `distance.py` の素性表を変えたときに黙って
  ずれるので、素性そのものを配って JS に写させる。
- **`align_phonemes()` は表示専用ではない**: `weighted_edit_distance` と同じコスト定義で
  DP を回して経路を復元する。**対ごとのコストの総和は編集距離に一致しなければならない**
  (`tests/test_distance.py::test_alignment_costs_sum_to_edit_distance` が検証する)。
  距離の重みを変えたらここも追従する。
- **スコア棒は結果内で正規化する**: 音韻スコアは上位が 0.90〜0.98 に密集するので、
  0〜1 をそのまま幅に写すと全部が満杯に見えて順位差が読めない。棒は相対差、
  数値が絶対値という二重表現にしている。

## 言語とスタイル

コード内のドキュメンテーション文字列・コメントは日本語で書く。既存コードは「なぜこの
選択をしたのか」「何を試して駄目だったのか」を実測値つきで残すスタイルなので、これに
合わせる。ruff の `RUF001-003` (全角記号を ASCII の誤入力として警告) は無効にしてある。

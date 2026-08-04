# Japanese Phonetic Retrieval Layer

LLM が弱い「音韻空間」を、外部の検索システムとして提供する。

LLM は意味的類似性には強いが、「音として似ている」という関係を明示的に扱うのが
苦手だ。たとえば **乳首** と **チョコビ** は意味的にはほぼ無関係だが、発音上は
一定の類似性がある。人間はなぞなぞ・ダジャレ・空耳・聞き間違いの文脈でこの
音韻的類似を自然に利用するが、semantic embedding ではこの関係を検索できない。

このプロジェクトは意味検索に対する semantic embedding、画像に対する vision
embedding と同様に、**音韻に対する retrieval** を提供する。

```
Semantic RAG : 意味の近い知識を取ってくる
Phonetic RAG : 音の近い語を取ってくる      <- これ
```

## 仕組み

オフラインで辞書全体を音韻ベクトル化しておき、実行時は入力語だけを
ベクトル化して近傍検索する。

```
オフライン:
  SudachiDict (288 万語)
      -> 読み (カタカナ)
      -> 音素列 / モーラ列
      -> phonetic embedding (5 空間)
      -> NumPy 行列 + HNSW 索引

オンライン:
  入力語 -> 読み -> 音素列 -> embedding
      -> ANN (HNSW) で候補 Top-K を高速取得         ... 速度をここで稼ぐ
      -> weighted phonetic edit distance で rerank  ... 精度をここで稼ぐ
      -> 最終 Top-N
```

embedding だけに任せないのは「なぜ近いのか」が曖昧になるためだ。ダジャレ用途
ではモーラ数が同じ・語尾が一致・子音だけ違う・母音列が似ているといった局所的な
構造が効くので、最終スコアはそれらを明示的に重み付けして決める。

### 音素距離

音素を記号ではなく音声学的素性のベクトルとして扱う。

```
/k/  [-voiced, velar, stop]
/g/  [+voiced, velar, stop]     -> 有声性のみの差なので距離は小さい (0.14)
/m/  [+voiced, bilabial, nasal] -> 調音位置と方法まで違うので大きい (0.69)
```

子音は調音位置・調音方法・有声性・口蓋化、母音は舌の高さ・前後・円唇性から
距離を導き、weighted Levenshtein で音素列間の距離を計算する。

### 5 つの埋め込み空間

用途によって「近い」の定義が変わるので、単一のベクトルに混ぜず別々に持つ。

| 空間 | 内容 | 効く場面 |
|---|---|---|
| `phonetic` | 全体の音韻 | 既定の候補生成 |
| `consonant` | 子音の骨格のみ | 母音が違っても子音が揃う語 |
| `vowel` | 母音の骨格のみ | 韻を踏む語 |
| `coda` | 語尾 2 モーラ | 韻、語尾の一致 |
| `rhythm` | モーラ数と特殊モーラの配置 | リズムの近さ |

素性は位置ビンに分けて畳み込むので語順が保たれる (「チクビ」と「ビクチ」は
別のベクトルになる)。

### 用途別プリセット

```
pun         ダジャレ・なぞなぞ。音韻全体の近さと、知られた語であることを重視
rhyme       韻を踏む語。語尾と母音列を重視し、語頭の一致は求めない
mishearing  聞き間違い・ASR 補正。全体の音韻とリズムの一致を最重視
```

## インストール

```bash
uv sync
uv run jpr build-index      # 初回のみ。10 分ほどかかる
```

索引は `~/.cache/jpr/index` に置かれる。

## 使い方

### 音が近い語を探す

```bash
uv run jpr similar 乳首
uv run jpr similar ラーメン --preset rhyme
uv run jpr similar 田中 --categories person
uv run jpr similar 乳首 --json
```

### 2 語を比較する

```console
$ uv run jpr compare 乳首 チョコビ
乳首 -> チクビ [ch i k u b i]
チョコビ -> チョコビ [ch o k o b i]

音韻類似度: 0.8083  (編集距離 1.150)

空間別:
  phonetic   0.9055
  consonant  1.0000
  vowel      0.3851
  coda       0.9428
  rhythm     1.0000
```

意味は無関係でも子音が完全一致していることが読み取れる。

### 読みと音素列を見る

```console
$ uv run jpr pronounce 東京特許許可局
東京特許許可局 -> トウキョウトッキョキョカキョク [t o u ky o u t o Q ky o ky o k a ky o k u] モーラ: ト ウ キョ ウ ト ッ キョ キョ カ キョ ク (11)
```

### Python から

```python
from jpr import PhoneticSearcher, PhoneticStore, default_store_path

searcher = PhoneticSearcher(PhoneticStore(default_store_path()))
pronunciation, results = searcher.search("乳首", preset="pun")

for result in results:
    print(result.surface, result.score, result.phonetic_similarity)
```

意味的な制約を掛けたい場合は `candidate_filter` を使う。

```python
snacks = {"チョコビ", "チョコボール", "ハイチュウ", "ウマイボウ"}
_, results = searcher.search(
    "乳首", candidate_filter=lambda entry: entry.reading in snacks
)
# -> チョコビ (0.808) が 1 位
```

## MCP サーバとして使う

```bash
uv run jpr serve
```

Claude Code に登録する場合:

```bash
claude mcp add jpr -- uv run --project /path/to/this/repo jpr serve
```

提供する tool:

| tool | 用途 |
|---|---|
| `search_phonetically` | 音が近い語を検索する |
| `compare_phonetically` | 2 語の音韻類似度を計算する |
| `pronounce` | 読み・音素列・モーラ構造を返す |

## 意味と音は別の軸

これが設計上もっとも重要な点だ。両者を混ぜない。

```
乳首 <-> バスト     semantic: 高い   phonetic: 0.05
乳首 <-> チョコビ   semantic: 低い   phonetic: 0.81
```

`semantic 低 + phonetic 高` はダジャレ候補のシグネチャになる。

ただし音韻空間だけでは答えが決まらないことに注意が必要だ。202 万語の辞書では
「乳首」に音韻類似度 0.82 以上の語が数百件ある。「乳首みたいなお菓子」の答えが
「チョコビ」なのは、それが音韻的に最も近いからではなく、**お菓子である**という
意味的制約が効くからだ。音韻検索は候補を出すところまでを担い、意味による選択は
呼び出し側 (LLM) が行う。

## 語彙カテゴリ

SudachiDict full は地名 49 万・人名 24 万を含み、索引の 7 割を占める。人名の
異表記 (「ココミ」に対する心々美・湖々美・瑚々海…) は音韻的に密集するので、
区別せずに検索すると一般語や商品名が上位から押し出される。

```
common   一般語 (普通名詞・動詞・形容詞)
product  商品名・作品名・組織名
person   人名     <- 既定では検索しない
place    地名     <- 既定では検索しない
```

## 実装メモ

- **辞書の列挙**: SudachiPy には語彙を列挙する API が無く (`lookup` は完全一致
  のみ)、Java 版の `dump` サブコマンドも提供されない。そのため `system.dic` の
  バイナリを直接パースしている (`dictionary.py`)。フォーマットの読み違いを黙って
  通さないよう、オフセットテーブルの整合性を起動時に検証する。
- **語の一般性**: 辞書には音が近いだけの稀語が大量にあるため、Sudachi の連接
  コストを反転させて「知られた語らしさ」の弱い指標として使っている。厳密な頻度
  ではないので、順位の同点をほどく程度の重みしか掛けていない。
- **索引形式**: pickle はロードに 12〜18 秒かかり MCP サーバの起動コストとして
  重すぎたため、NumPy 配列 (mmap) に移行した。文字列は固定幅 unicode 配列を避けて
  UTF-8 の可変長で持つ (`<U27` は最長要素の幅で全行を埋めるため、202 万語の表層
  だけで 218MB を消費していた)。

## 今後

- アクセント (高低) を距離とベクトルに取り込む
- ニューラルな phonetic embedding の学習 (現在は素性からの決定的な構成)
- 語より長いフレーズ単位の検索 (空耳・歌詞検索)

## 開発

```bash
uv run pytest        # テスト
uv run ruff check .  # lint
```

実辞書を必要とするテストは、索引が未構築なら自動的にスキップされる。

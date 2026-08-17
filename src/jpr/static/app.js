/* jpr の Web フロント。ビルド工程を持たない素の JS。
 *
 * 音素の色は素性表 (/api/phonemes) から決める。UI 側に色の固定表を持つと
 * distance.py の素性表を変えたときに黙ってずれるので、素性を引いて写す。 */

"use strict";

const CATEGORY_LABELS = {
  common: "一般語",
  product: "商品・作品名",
  person: "人名",
  place: "地名",
  other: "その他",
};

// 既定の検索対象から外れるカテゴリ。index.py の DEFAULT_CATEGORIES に対応。
const NON_DEFAULT_CATEGORIES = new Set(["person", "place"]);

const PLACE_LABELS = {
  bilabial: "両唇",
  labiodental: "唇歯",
  alveolar: "歯茎",
  postalveolar: "後部歯茎",
  palatal: "口蓋",
  velar: "軟口蓋",
  glottal: "声門",
};

const VOWEL_LABELS = { i: "前舌・狭", e: "前舌・半狭", a: "中舌・広", o: "後舌・半狭", u: "後舌・狭" };

const SPECIAL_LABELS = { R: "長音", Q: "促音", N: "撥音" };

/** 音素の素性表と IPA。/api/phonemes で埋める。 */
let features = { consonants: {}, vowels: {}, special: {} };

/** スコア下限の既定。/api/info で埋める。URL に載せるかの判定に使うので、
 *  フロントに固定値を持たせず web.py の DEFAULT_MIN_SCORE を写す。 */
let defaultMinScore = null;

/** 音素チップに IPA を併記するか。チップは色・title で既に情報を持っており、
 *  結果行ごとに並ぶので、常時併記すると密度が上がりすぎる。既定は off。 */
let showIpa = false;

const $ = (id) => document.getElementById(id);

async function getJSON(path, params) {
  const url = new URL(path, location.origin);
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== null && value !== undefined && value !== "") url.searchParams.set(key, value);
  }
  const response = await fetch(url);
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.detail || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function setStatus(el, message, isError) {
  el.textContent = message;
  el.classList.toggle("is-error", Boolean(isError));
}

/* ---------- 音素チップ ---------- */

/** 音素の表示色を素性から決める。子音は調音位置、母音は舌の位置。 */
function phonemeColor(symbol) {
  const consonant = features.consonants[symbol];
  if (consonant) return `var(--p-${consonant.place})`;
  if (features.vowels[symbol]) return `var(--v-${symbol})`;
  return "var(--special)";
}

/** 音素 1 つの IPA。表は API から来るので UI 側に固定表を持たない。 */
function phonemeIpa(symbol) {
  return (
    features.consonants[symbol]?.ipa ||
    features.vowels[symbol]?.ipa ||
    features.special[symbol]?.ipa ||
    ""
  );
}

/** 音素記号と IPA が別物として読めるか。
 *
 * 単純な !== では足りない。IPA の [ɡ] (U+0261 一本足の g) は ASCII の "g" と
 * 別のコードポイントだが画面上ほぼ同形で、併記しても情報が増えず幅だけ増える。
 * 同じことが起きる対をここで潰す。 */
const _LOOKALIKE = { g: "ɡ" };

/** 促音。チップ単体では IPA を出さない。
 *
 * 促音の実現は後続子音の重複 (キッテ = [kitte]) で、音素 1 つに対応する記号を
 * 持たない。API が返す [ʔ] は語末促音の実現だが、チップは後続を見ないので
 * 語中でもそれを出してしまい、連続表記 (サーバ側で重複に書き換える) と食い違う。
 * 促音の音価は連続表記のほうで読ませる。 */
const _NO_STANDALONE_IPA = new Set(["Q"]);

function ipaDiffers(symbol) {
  if (_NO_STANDALONE_IPA.has(symbol)) return false;
  const ipa = phonemeIpa(symbol);
  if (!ipa || ipa === symbol) return false;
  return _LOOKALIKE[symbol] !== ipa;
}

/** 音素の読み方の説明。チップの title に入れる。 */
function phonemeTitle(symbol) {
  const ipa = phonemeIpa(symbol);
  // IPA を先頭に置く。併記が off のときも title からは読めるようにする。
  // title は幅の制約が無いので、見分けのつかない対 (g/ɡ) も含めて出す。
  let head = ipa && ipa !== symbol ? `${symbol} [${ipa}]` : symbol;
  // 促音だけは単独の記号を持たない。[ʔ] と書くと語中でも声門閉鎖に見えるので、
  // 何が起きるかを言葉で説明する (`_NO_STANDALONE_IPA`)。
  if (_NO_STANDALONE_IPA.has(symbol)) head = `${symbol} (後続子音の重複、語末では [ʔ])`;
  const consonant = features.consonants[symbol];
  if (consonant) {
    const parts = [PLACE_LABELS[consonant.place] || consonant.place, consonant.manner];
    parts.push(consonant.voiced ? "有声" : "無声");
    if (consonant.palatalized) parts.push("口蓋化");
    return `${head} — ${parts.join(" / ")}`;
  }
  if (features.vowels[symbol]) return `${head} — 母音 ${VOWEL_LABELS[symbol] || ""}`.trim();
  if (SPECIAL_LABELS[symbol]) return `${head} — ${SPECIAL_LABELS[symbol]}`;
  return head;
}

function phonemeChip(symbol) {
  const el = document.createElement("span");
  el.className = "ph";
  if (symbol === null) {
    el.classList.add("is-gap");
    el.textContent = "–";
    el.title = "対応する音素がない (挿入・削除)";
    return el;
  }
  el.textContent = symbol;
  el.style.color = phonemeColor(symbol);
  if (features.consonants[symbol]?.voiced || features.vowels[symbol]) {
    el.classList.add("is-voiced");
  }
  el.title = phonemeTitle(symbol);

  // IPA が記号と見分けのつかない字なら併記しても情報が増えないので出さない。
  if (showIpa && ipaDiffers(symbol)) {
    const ipa = phonemeIpa(symbol);
    const sub = document.createElement("span");
    sub.className = "ph-ipa";
    sub.textContent = ipa;
    el.append(sub);
  }
  return el;
}

function renderPhonemes(container, symbols) {
  // 再描画のために元の音素列を持たせる。IPA の併記を切り替えたとき、
  // 画面に出ている全チップを引き直さずに組み直せる。
  container.dataset.phonemes = JSON.stringify(symbols);
  container.replaceChildren(...symbols.map(phonemeChip));
}

/** 描画済みの音素チップを組み直す。IPA 併記の切り替えで使う。 */
function rerenderAllPhonemes() {
  for (const container of document.querySelectorAll(".phonemes[data-phonemes]")) {
    renderPhonemes(container, JSON.parse(container.dataset.phonemes));
  }
}

/* ---------- 検索 ---------- */

let activePreset = "pun";
const activeCategories = new Set();

/** 直近の検索クエリの IPA。アライメントパネルが候補側と並べて出す。
 *  renderResults から 4 段渡すことになるので引数にせず、ここに置く。 */
let queryIpa = "";

function familiarityMeter(value) {
  // 4 段。Sudachi のコスト由来の弱い指標なので連続値のバーにはしない。
  const steps = Math.round(Math.max(0, Math.min(1, value)) * 4);
  const wrap = document.createElement("span");
  wrap.className = "fam";
  wrap.title = `一般性 ${value.toFixed(2)} (Sudachi のコストを反転した弱い指標)`;
  for (let i = 0; i < 4; i += 1) {
    const step = document.createElement("span");
    step.className = i < steps ? "fam-step is-on" : "fam-step";
    wrap.append(step);
  }
  return wrap;
}

/** 包含の印。クエリの音がこの語に完全な形で入っていることを示す。
 *
 * **占有率を数値で出さない。** 意味は「クエリが語全体の何割を占めるか」で、
 * 読み手が知りたいのは「入っているかどうか」と「余分がどれだけあるか」。
 * 後者は読み (`result.reading`) を見れば直接わかるので、数値は title に置く。 */
function containmentTag(ratio) {
  const tag = document.createElement("span");
  tag.className = "tag is-contained";
  tag.textContent = "音が丸ごと入る";
  tag.title = `クエリの音がこの語の ${Math.round(ratio * 100)}% を占める`;
  return tag;
}

/** スコア棒の長さ。結果内の最小〜最大に張り直す。
 *
 * 音韻スコアは上位が 0.90〜0.98 に密集するので、0〜1 をそのまま幅に写すと
 * 全部が満杯に見えて順位の差が読めない。棒は結果同士の相対差を示す役目に絞り、
 * 絶対値は隣の数値で読ませる。 */
function scoreBarWidth(score, min, max) {
  if (max - min < 1e-6) return 100;
  return 8 + ((score - min) / (max - min)) * 92;
}

/** 結果 1 件。クリックで音素アライメントを開く。
 *
 * モーラ列の中に入るので横幅を当てにできない。スコア・語・タグを縦に積み、
 * モーラ数のタグは持たせない (列の見出しがそれを担う)。 */
function resultRow(result, queryPhonemes, rank, range) {
  const li = document.createElement("li");
  li.className = "result";

  const button = document.createElement("button");
  button.className = "result-open";
  button.type = "button";
  button.setAttribute("aria-expanded", "false");

  const head = document.createElement("div");
  head.className = "result-head";
  const score = document.createElement("span");
  score.className = "score";
  score.textContent = result.score.toFixed(3);
  const word = document.createElement("span");
  word.className = "word";
  word.textContent = result.word;
  head.append(score, word);

  const bar = document.createElement("span");
  bar.className = "score-bar";
  bar.style.width = `${scoreBarWidth(result.score, range.min, range.max)}%`;
  bar.title = "棒はこの結果内での相対差。絶対値は数値で読む";

  const wordCell = document.createElement("div");
  wordCell.className = "result-word";

  // 読みと IPA を同じ行に並べる。カードは縦に積む要素が多いので行を増やさない。
  // **IPA は音素チップの併記トグルとは独立に常に出す** — 発音そのものを読む
  // 表記であって、音素記号の注釈ではない。
  const readingRow = document.createElement("div");
  readingRow.className = "result-reading-row";
  const reading = document.createElement("span");
  reading.className = "word-reading";
  reading.textContent = result.reading;
  readingRow.append(reading);
  if (result.ipa) {
    const ipa = document.createElement("span");
    ipa.className = "ipa word-ipa";
    ipa.textContent = `[${result.ipa}]`;
    readingRow.append(ipa);
  }

  const phonemes = document.createElement("div");
  phonemes.className = "phonemes result-phonemes";
  renderPhonemes(phonemes, result.phonemes);
  wordCell.append(readingRow, phonemes);

  const tags = document.createElement("div");
  tags.className = "result-tags";
  const category = document.createElement("span");
  category.className = "tag";
  category.textContent = CATEGORY_LABELS[result.category] || result.category;
  tags.append(category);
  // クエリの音が完全な形で入っている語には印を出す。**カードの中で一番
  // 説明が要る性質**なので、他のタグと同じ地味さでは見つけられない。
  if (result.containment) {
    tags.append(containmentTag(result.containment));
  }
  tags.append(familiarityMeter(result.familiarity));

  button.append(head, bar, wordCell, tags);
  li.append(button);

  let panel = null;
  button.addEventListener("click", async () => {
    const open = button.getAttribute("aria-expanded") === "true";
    if (open) {
      panel?.remove();
      panel = null;
      button.setAttribute("aria-expanded", "false");
      return;
    }
    button.setAttribute("aria-expanded", "true");
    panel = document.createElement("div");
    panel.className = "align";
    panel.textContent = "対応付けを計算中…";
    li.append(panel);
    try {
      const data = await getJSON("/api/align", {
        a: queryPhonemes.join(" "),
        b: result.phonemes.join(" "),
      });
      panel.replaceChildren(...alignmentNodes(data, result));
      sizePanelToTrack(panel);
    } catch (error) {
      panel.textContent = `対応付けを取得できません: ${error.message}`;
    }
  });

  li.dataset.rank = String(rank);
  return li;
}

/** パネルの幅を帯の実寸に合わせ、画面から溢れるなら左へ寄せる。
 *
 * 帯は音素 1 つに 1 列を使うので、必要な幅は列の幅ではなく対の長さで決まる。
 * CSS 側は `--align-track-px` を下限として読む (`.align` の項)。素の
 * `width: max-content` に任せると、**折り返さない脚注の 1 行が幅を決めて**
 * 6 列の帯に 556px のパネルが付いた。帯だけを測って渡す。
 *
 * 測るのは `scrollWidth` — 溢れた状態の実寸なので、パネルが狭いままでも
 * 「本当はどれだけ要るか」が取れる。
 *
 * **右端の列では左へ伸ばす。** パネルは列の左端から右へ広がるので、最後の
 * モーラ数の列で開くと見える範囲の外へ出る (実測で 7 モーラの帯が右端 1480px ·
 * ビューポート 1440px)。溢れたぶんだけ負の margin で引き戻す — 行の中の
 * 位置は変えず、描画だけを左へずらす。
 *
 * **基準はビューポートではなく `.mora-columns` の可視範囲。** あちらは
 * `overflow-x: auto` で自分がスクロールコンテナなので、画面に収まっていても
 * コンテナの外に出たぶんは読めない。ビューポートだけを見ていたとき、765px 幅で
 * 右端の列を開くと左端が -121px となり**帯の頭が隠れた** (右への溢れを
 * 640px 引き戻した結果、今度は左へ突き抜けた)。左右どちらの側も
 * はみ出させない。 */
function sizePanelToTrack(panel) {
  const track = panel.querySelector(".align-track");
  if (!track) return;
  // `box-sizing: border-box` なので width は padding と border を含む。
  // 帯が収まる内寸を確保するには、その分を足した外寸を渡す。
  const style = getComputedStyle(panel);
  const inset =
    parseFloat(style.paddingLeft) +
    parseFloat(style.paddingRight) +
    parseFloat(style.borderLeftWidth) +
    parseFloat(style.borderRightWidth);
  panel.style.setProperty("--align-track-px", `${Math.ceil(track.scrollWidth + inset)}px`);

  // 幅が確定してから溢れを測る。ずらす前を基準にしたいので、前回ぶんは戻す。
  panel.style.marginLeft = "";
  const viewport = panel.closest(".mora-columns");
  if (!viewport) return;
  const bounds = viewport.getBoundingClientRect();
  const rect = panel.getBoundingClientRect();
  // 右の溢れを引き戻す。ただしパネルのほうが可視範囲より広いときは、
  // 引き戻すと左が切れるだけなので左端を優先して揃える (帯は左から読む)。
  const shift = Math.min(rect.right - bounds.right, rect.left - bounds.left);
  if (shift > 0) panel.style.marginLeft = `${-Math.ceil(shift)}px`;
}

/** 署名要素: 音素の対応付けを縦 2 段で並べ、対ごとの素性距離を縦棒で示す。 */
function alignmentNodes(data, result) {
  const title = document.createElement("p");
  title.className = "align-title";
  title.textContent = "音素の対応 — 上がクエリ、下が候補。縦棒は素性距離";

  // クエリ側の連続表記だけを置く。候補側は同じカードの読みの隣に常時出ているので、
  // ここに並べると 1 枚のカードに同じ IPA が 2 回出る。下の対応は上段がクエリ・
  // 下段が候補なので、上段が何の発音かをここで示す。
  const ipaRow = document.createElement("p");
  ipaRow.className = "align-ipa";
  if (queryIpa) {
    const cell = document.createElement("span");
    cell.className = "align-ipa-cell";
    const tag = document.createElement("span");
    tag.className = "readout-label";
    tag.textContent = "クエリ";
    const value = document.createElement("span");
    value.className = "ipa";
    value.textContent = `[${queryIpa}]`;
    cell.append(tag, value);
    ipaRow.append(cell);
  }

  const track = document.createElement("div");
  track.className = "align-track";

  // 距離を高さに写す基準。子音と母音の置換 (1.0) が満杯になるよう固定し、
  // 行ごとに最大値で正規化しない。行間で高さが比較できなくなるため。
  const GAUGE_MAX = 1.0;
  // CSS の `.align-col` の 2 行目の高さと一致させる。ここだけ大きいと
  // 満杯の棒が次の行 (候補側のチップ) に食い込む。
  const GAUGE_PX = 26;

  // モーラの先頭に区切りを立てる。子音の直後の音素は同じモーラの母音なので
  // 境界にならない。挿入・削除では片側に音素が無いため、クエリ側の列が
  // 埋まっているときだけ境界を更新する (候補側で数えると 2 語の
  // モーラ構造が食い違ったときに区切りが二重に立つ)。
  let prevWasConsonant = false;
  for (const pair of data.pairs) {
    const column = document.createElement("div");
    const classes = ["align-col"];
    if (pair.op === "match") classes.push("is-match");
    if (pair.a === null || pair.b === null) classes.push("is-gap");

    if (pair.a !== null) {
      if (!prevWasConsonant) classes.push("is-mora-start");
      prevWasConsonant = Boolean(features.consonants[pair.a]);
    }

    column.className = classes.join(" ");

    const top = phonemeChip(pair.a ?? null);
    const gauge = document.createElement("div");
    gauge.className = "align-gauge";
    const stem = document.createElement("span");
    stem.style.height = `${Math.min(1, pair.distance / GAUGE_MAX) * GAUGE_PX}px`;
    gauge.append(stem);
    const bottom = phonemeChip(pair.b ?? null);

    // 距離は棒の長さと数値の二重表現。並んだときの差は棒で、
    // 絶対値は数値で読む。
    const num = document.createElement("span");
    num.className = "align-num";
    num.textContent = pair.distance.toFixed(2);
    column.title = `${pair.a ?? "–"} → ${pair.b ?? "–"} / 距離 ${pair.distance.toFixed(3)} (${
      { match: "一致", sub: "置換", del: "削除", ins: "挿入" }[pair.op]
    })`;

    column.append(top, gauge, bottom, num);
    track.append(column);
  }

  const foot = document.createElement("div");
  foot.className = "align-foot";
  const items = [
    ["編集距離", data.total.toFixed(3)],
    ["音韻類似度", result.phonetic_similarity.toFixed(3)],
    ["ANN 空間", result.embedding_similarity.toFixed(3)],
    ["語尾", result.coda_similarity.toFixed(3)],
    ["母音列", result.vowel_similarity.toFixed(3)],
    ["品詞", result.pos],
  ];
  for (const [label, value] of items) {
    const span = document.createElement("span");
    span.append(`${label} `);
    const strong = document.createElement("b");
    strong.textContent = value;
    span.append(strong);
    foot.append(span);
  }

  return ipaRow.childElementCount ? [title, ipaRow, track, foot] : [title, track, foot];
}

/** 1 枠の幅。結果行の中身 (スコア + 語 + 音素チップ列 + タグ) が折り返さずに
 * 読める最小値を実測で決めた値。これより狭めると音素チップが 2 行に落ちて
 * 行の高さが不揃いになり、列をまたいだ順位の読み比べができなくなる。
 *
 * **成り立つのは 4 モーラ程度まで。** 実測で 3 モーラ (6 音素) のチップ列は
 * 179px で収まるが、5〜6 モーラ (17〜18 音素) は 636px を要求し、既定の 250px
 * では 20 行すべてが折り返す。長い語の検索では前提が崩れているが、枠をそこに
 * 合わせると 1 画面に 2 列しか入らないので、多数派の短い語を基準に置いている。 */
const SLOT_PX = 250;

/** IPA を併記したときの 1 枠の幅。
 *
 * 併記でチップが広がる。3 モーラのチップ列は 179px → 268px (+50%) になり、
 * 250px の枠では行の内寸 (253px) に 15px 足りずに折り返す。**既定の検索で
 * 最も多い長さがそこなので、ここは枠を広げて合わせる**。上の SLOT_PX が
 * 「4 モーラまで」なのと同じ考え方で、比例させた値を実測で確かめてある。 */
const SLOT_PX_IPA = 340;

/** 現在の 1 枠の幅。IPA 併記の状態で切り替わる。 */
const slotPx = () => (showIpa ? SLOT_PX_IPA : SLOT_PX);

/** モーラ列に枠を配る。
 *
 * 枠は「幅 SLOT_PX の縦 1 本」で、1 モーラ数 = 最低 1 枠。まずモーラ数ごとに
 * 1 枠ずつ配り、余った枠をヒット数の多いモーラ数から順に 1 枠ずつ足す。
 * 2 枠以上を持つ列は中身を CSS multi-column で段に割るので、列の見出しは
 * 1 つのままモーラ数あたりの表示件数が増える。
 *
 * 枠が全モーラ数に行き渡らない狭い画面では 1 枠ずつのまま返す
 * (総枠数が列数を下回る)。この場合は列が SLOT_PX を保って横スクロールになる —
 * 列を潰して詰めると行が折り返して読めなくなるため。 */
function allocateSlots(counts, groups, totalSlots) {
  const slots = new Map(counts.map((count) => [count, 1]));
  let spare = totalSlots - counts.length;
  if (spare <= 0) return slots;

  // ヒット数の多い順。同数はモーラ数の小さい順で決定的に (counts は昇順)。
  const byHits = [...counts].sort((a, b) => groups.get(b).length - groups.get(a).length);

  // 1 周に 1 枠ずつ配る。多い列に一気に寄せると 3 枠目・4 枠目が
  // 段あたり数件しかない痩せた段になるので、幅は広く分け合わせる。
  while (spare > 0) {
    let placed = false;
    for (const count of byHits) {
      if (spare === 0) break;
      // 1 段あたり最低 4 件は入れる。件数の少ない列に枠を足しても
      // 空白が伸びるだけで、他の列から幅を奪う損のほうが大きい。
      if (groups.get(count).length < (slots.get(count) + 1) * 4) continue;
      slots.set(count, slots.get(count) + 1);
      spare -= 1;
      placed = true;
    }
    // どの列も枠を受け取れないなら余りは捨てる (全列が短い場合)。
    if (!placed) break;
  }
  return slots;
}

/** 結果をモーラ数ごとの列に分けて描く。
 *
 * モーラ数の違う語は phonetic 空間の近傍に入らない (「乳首」3 モーラに対する
 * 「筑前煮」5 モーラ) ので、混ぜて 1 列に並べるとスコア上位を同モーラの語が
 * 埋め、モーラ数の違う候補が下に押し出されて見えなくなる。列に分けると
 * 各モーラ数の中での順位が読めるようになる。
 *
 * 列は結果に現れたモーラ数だけ立てる。クエリと同じモーラ数の列には印を付ける。
 * 画面幅に余りがあればヒット数の多いモーラ数の列を段組にして埋める
 * (`allocateSlots`)。 */
function renderMoraColumns(container, results, queryPhonemes, queryMora, range) {
  // 順位はモーラ数をまたいだ全体順位。列に分けても「全体で何位か」は
  // 情報として残す必要があるので、束ねる前に振っておく。
  const groups = new Map();
  results.forEach((result, i) => {
    if (!groups.has(result.mora_count)) groups.set(result.mora_count, []);
    groups.get(result.mora_count).push({ result, rank: i + 1 });
  });

  const counts = [...groups.keys()].sort((a, b) => a - b);

  // 入る枠数は container の実幅から出す。列間の 1px は無視できる誤差。
  // clientWidth が 0 になるのは検索ビューが非表示のときだけ。そのときは
  // ビューポートから main の左右 padding を引いて見積もる (実測できないので
  // 多めに配らないよう控えめに引く)。
  const available =
    container.clientWidth ||
    document.documentElement.clientWidth - 2 * 32;
  const totalSlots = Math.max(counts.length, Math.floor(available / slotPx()));
  const slots = allocateSlots(counts, groups, totalSlots);

  // grid は枠を単位に敷く。列は自分が持つ枠数ぶんの span を取る。
  const gridSlots = [...slots.values()].reduce((a, b) => a + b, 0);
  container.style.setProperty("--slots", String(gridSlots));
  // 枠の下限も IPA の状態に連動させる。CSS 側に固定値を残すと、枠数だけ
  // 減って 1 枠が広がらず、結局チップが折り返す。
  container.style.setProperty("--slot-px", `${slotPx()}px`);
  container.replaceChildren(
    ...counts.map((count) => {
      const items = groups.get(count);
      const span = slots.get(count);
      const column = document.createElement("section");
      column.className = "mora-column";
      column.style.setProperty("--span", String(span));
      if (count === queryMora) column.classList.add("is-query-mora");

      const head = document.createElement("h3");
      head.className = "mora-head";
      const label = document.createElement("span");
      label.className = "mora-head-count";
      label.textContent = `${count} モーラ`;
      head.append(label);
      if (count === queryMora) {
        const mark = document.createElement("span");
        mark.className = "mora-head-mark";
        mark.textContent = "入力と同じ";
        head.append(mark);
      } else {
        const diff = count - queryMora;
        const mark = document.createElement("span");
        mark.className = "mora-head-diff";
        mark.textContent = `${diff > 0 ? "+" : "−"}${Math.abs(diff)}`;
        mark.title = `入力より ${Math.abs(diff)} モーラ ${diff > 0 ? "長い" : "短い"}`;
        head.append(mark);
      }
      const num = document.createElement("span");
      num.className = "mora-head-num";
      num.textContent = `${items.length}`;
      head.append(num);

      const list = document.createElement("ol");
      list.className = "results";
      // 段数は枠数と一致させる。ここを CSS の column-width に任せると
      // 枠の配り方 (allocateSlots) と段数がずれて、余りを配ったはずの列が
      // 1 段のまま残る。
      list.style.setProperty("--cols", String(span));
      list.append(...items.map(({ result, rank }) => resultRow(result, queryPhonemes, rank, range)));

      column.append(head, list);
      return column;
    }),
  );
}

/** 直近の描画の入力。幅が変わったときに枠を配り直すために持つ。
 * 検索を投げ直さずに再配分できるので、全走査 (数百 ms) の結果を
 * ウィンドウ幅を変えるたびに捨てずに済む。 */
let lastRender = null;

function renderResults(results, queryPhonemes, queryMora, range) {
  lastRender = { results, queryPhonemes, queryMora, range };
  renderMoraColumns($("results"), results, queryPhonemes, queryMora, range);
}

/** 幅の変化で枠数が変わったときだけ描き直す。開いていたアライメント帯は
 * 閉じるが、これは段組の段割りが行の高さに依存するため避けられない。 */
function relayoutResults() {
  if (!lastRender?.results.length) return;
  const { results, queryPhonemes, queryMora, range } = lastRender;
  renderMoraColumns($("results"), results, queryPhonemes, queryMora, range);
}

async function runSearch(event) {
  event?.preventDefault();
  const query = $("q").value.trim();
  if (!query) return;

  const status = $("search-status");
  const button = $("search-form").querySelector(".run");
  button.disabled = true;
  setStatus(status, "検索中…");

  const started = performance.now();
  try {
    const data = await getJSON("/api/similar", {
      q: query,
      preset: activePreset,
      limit: $("limit").value,
      candidates: $("candidates").value,
      min_score: $("min-score").value,
      categories: [...activeCategories].join(","),
      // getJSON は空文字列を落とすので、未入力のモーラ範囲は送られない。
      min_mora: $("min-mora").value,
      max_mora: $("max-mora").value,
    });
    const elapsed = Math.round(performance.now() - started);

    $("rq-reading").textContent = data.reading || "(読みが取れない)";
    queryIpa = data.ipa || "";
    $("rq-ipa").textContent = queryIpa ? `[${queryIpa}]` : "";
    $("rq-mora").textContent = `${data.mora_count}`;
    renderPhonemes($("rq-phonemes"), data.phonemes);
    $("query-readout").hidden = false;

    const scores = data.results.map((r) => r.score);
    const range = { min: Math.min(...scores), max: Math.max(...scores) };
    renderResults(data.results, data.phonemes, data.mora_count, range);

    if (!data.results.length) {
      // スコア下限は助言に出さない。満たす語が無ければサーバが下限を外して
      // 近い順を返す (`below_floor`) ので、ここに来る 0 件は母集団そのものが
      // 空だということ。全走査では候補数を増やしても何も変わらない。
      const advice = data.scanned == null
        ? "候補数を増やすか、カテゴリの絞り込みを外す"
        : "モーラ範囲を広げるか、カテゴリの絞り込みを外す";
      setStatus(status, `該当なし (${elapsed}ms) — ${advice}`);
    } else {
      const parts = [`${data.results.length} 件`, `${elapsed}ms`];
      if (data.scanned != null) {
        parts.push(`${data.scanned.toLocaleString("ja-JP")} 語を全走査`);
      }
      // 下限を外して返ってきたときは、画面の件数が「基準を満たした数」では
      // ないことを言う。黙って出すと 0.8 未満のスコアが並ぶ理由が読めない。
      if (data.below_floor) {
        parts.push(`スコア ${$("min-score").value} 以上が無いので近い順`);
      }
      setStatus(status, `${parts.join(" / ")} — 行をクリックすると音素の対応が開く`);
    }

    // モーラ範囲も URL に載せて、絞り込んだ結果を共有できるようにする。
    const shared = new URLSearchParams({ q: query, preset: activePreset });
    if ($("min-mora").value) shared.set("min_mora", $("min-mora").value);
    if ($("max-mora").value) shared.set("max_mora", $("max-mora").value);
    // 既定と違う値だけ載せる。既定はサーバから配られるので、そのときの
    // 既定と突き合わせる (定数を変えたときに URL が長くなるだけで済む)。
    if ($("limit").value !== "0") shared.set("limit", $("limit").value);
    // スコア下限は母集団を決める主役なので、共有した URL で再現できないと
    // 「同じ画面」にならない。
    if ($("min-score").value !== String(defaultMinScore)) {
      shared.set("min_score", $("min-score").value);
    }
    history.replaceState(null, "", `?${shared}`);
  } catch (error) {
    setStatus(status, error.message, true);
    $("results").replaceChildren();
    $("query-readout").hidden = true;
  } finally {
    button.disabled = false;
  }
}

/* ---------- 分割合成 (空耳) ---------- */

/** 合成候補 1 件。
 *
 * 空耳は「入力のどこが何になったか」が読めないと検証できないので、
 * 表層だけでなく区間の対応を必ず並べる。区間は入力を隙間なく覆うので、
 * 上段 (入力の読み) と下段 (当てた語) を同じ幅で縦に揃えれば対応が読める。 */
function phraseCard(candidate, rank) {
  const card = document.createElement("article");
  card.className = "phrase";

  const head = document.createElement("div");
  head.className = "phrase-head";

  const rankEl = document.createElement("span");
  rankEl.className = "phrase-rank";
  rankEl.textContent = String(rank);

  const textEl = document.createElement("span");
  textEl.className = "phrase-text";
  textEl.textContent = candidate.text;

  const readingEl = document.createElement("span");
  readingEl.className = "phrase-reading";
  readingEl.textContent = candidate.reading;

  const scoreEl = document.createElement("span");
  scoreEl.className = "phrase-score";
  scoreEl.textContent = candidate.score.toFixed(3);
  scoreEl.title = `音韻類似度の平均 ${candidate.phonetic_similarity.toFixed(3)} / ${candidate.segment_count} 区間`;

  head.append(rankEl, textEl, readingEl, scoreEl);

  // 区間の対応。列 1 つが 1 区間で、モーラ数に比例した幅を持たせる
  // (入力のどこを覆っているかが幅で読めるようにする)。
  const grid = document.createElement("div");
  grid.className = "seg-grid";
  for (const segment of candidate.segments) {
    const col = document.createElement("div");
    col.className = segment.is_particle ? "seg-col is-particle" : "seg-col";
    col.style.flexGrow = String(segment.mora_count);

    const source = document.createElement("span");
    source.className = "seg-source";
    source.textContent = segment.source_reading;

    const arrow = document.createElement("span");
    arrow.className = "seg-arrow";
    arrow.setAttribute("aria-hidden", "true");
    arrow.textContent = "↓";

    const surface = document.createElement("span");
    surface.className = "seg-surface";
    surface.textContent = segment.surface;

    const meta = document.createElement("span");
    meta.className = "seg-meta";
    meta.textContent = segment.reading;

    const bar = document.createElement("span");
    bar.className = "seg-bar";
    // 区間ごとの音の一致度。どこで音が外れたかが読めるようにする。
    bar.style.setProperty("--fill", `${Math.round(segment.similarity * 100)}%`);
    bar.title = `音韻類似度 ${segment.similarity.toFixed(3)}${segment.is_particle ? " (助詞)" : ""}`;

    col.append(source, arrow, surface, meta, bar);
    grid.append(col);
  }

  card.append(head, grid);
  return card;
}

/* ---------- ラティス図 ---------- */

/** 分割合成の見せ方。"list" (候補を並べる) か "lattice" (DAG に畳む)。 */
let phraseView = "list";

/** 直近のラティス。ノードを選んで絞り込むときに使い回す。 */
let latticeData = null;

/** 選択中のノード id。空なら全経路を出す。 */
const latticeSelection = new Set();

/** 列の幅の下限・上限と列の間隔 (px)。
 *
 * モーラ位置に比例させるのをやめ、中身が要る分だけ取る。間隔を広めに取るのは
 * 辺の始点と終点を見分けるため — 詰めると複数の辺がノードの縁で重なる。 */
const LATTICE_MIN_COL_PX = 76;
const LATTICE_MAX_COL_PX = 180;
const LATTICE_COL_GAP_PX = 56;

/** ノードの高さと行間 (px)。SVG の座標計算と CSS の両方で使うので JS 側に持つ。 */
const LATTICE_ROW_PX = 46;
const LATTICE_NODE_H = 34;

/** ノードを列に割り当て、列ごとの幅と各ノードの座標を決める。
 *
 * **モーラ位置に幅を比例させない。** 以前はモーラ位置を横軸に固定していたが、
 * 1 モーラのノードが 78px しか取れない一方で隣の列が遠くに置かれ、辺が長い
 * 曲線になって「どれがどれに繋がるか」が読めなかった。
 *
 * 代わりに**開始位置で列を作り、列の幅は中身が要求する分だけ取る**。
 * 上下の位置合わせは捨てる — 位置を揃えることより、辺が短く追えることを取る。
 * どのモーラを覆っているかはノードが持つ読み (`source_reading`) で分かるので、
 * 目盛りに頼らなくてよい。
 *
 * 同じ列の中は経路数の多い順に縦に積む (よく使われる語が上)。 */
function layoutLattice(nodes, edges) {
  // 開始位置ごとに列を作る。同じ位置から始まるノードは互いに排他な選択肢
  // (どれか 1 つを選ぶ) なので、縦に並べるのが自然。
  const byStart = new Map();
  for (const node of nodes) {
    if (!byStart.has(node.start)) byStart.set(node.start, []);
    byStart.get(node.start).push(node);
  }
  const columns = [...byStart.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([start, items]) => ({
      start,
      items: items.sort(
        (a, b) =>
          b.path_count - a.path_count ||
          a.mora_count - b.mora_count ||
          a.surface.localeCompare(b.surface),
      ),
    }));

  // 列の中の縦順を、前の列の繋がり先に寄せて決める。
  //
  // 経路数だけで並べると、繋がっている 2 つのノードが離れた行に来て辺が
  // 縦に大きく動く (実測で最大 414px = 9 行ぶん)。前の列でのその語の入口
  // (直前のノードの行) の平均を目標にして並べ替えると、辺が短くなる。
  const rowOf = new Map();
  const incoming = new Map();
  for (const edge of edges) {
    if (!edge.source || !edge.target) continue;
    if (!incoming.has(edge.target)) incoming.set(edge.target, []);
    incoming.get(edge.target).push(edge.source);
  }
  for (const column of columns) {
    const target = (node) => {
      const sources = (incoming.get(node.id) || [])
        .map((id) => rowOf.get(id))
        .filter((row) => row !== undefined);
      if (!sources.length) return Number.POSITIVE_INFINITY;
      return sources.reduce((sum, row) => sum + row, 0) / sources.length;
    };
    // 目標行の昇順。入口が無いノード (先頭の列など) は経路数の順で後ろに置く。
    column.items.sort((a, b) => {
      const gap = target(a) - target(b);
      if (Number.isFinite(gap) && gap !== 0) return gap;
      return b.path_count - a.path_count || a.surface.localeCompare(b.surface);
    });
    column.items.forEach((node, depth) => rowOf.set(node.id, depth));
  }

  // 列の幅は中身の一番長い表層に合わせる。全角 1 文字あたりの実効幅で見積もり、
  // 下限と上限で挟む (長すぎる語は CSS 側で省略される)。
  const box = new Map();
  let x = 0;
  for (const column of columns) {
    const longest = Math.max(
      ...column.items.map((n) => Math.max(n.surface.length, n.source_reading.length * 0.8)),
    );
    const width = Math.min(LATTICE_MAX_COL_PX, Math.max(LATTICE_MIN_COL_PX, longest * 15 + 20));
    column.items.forEach((node, depth) => {
      box.set(node.id, { x, y: depth * LATTICE_ROW_PX, w: width });
    });
    column.width = width;
    column.x = x;
    x += width + LATTICE_COL_GAP_PX;
  }

  const rowCount = Math.max(...columns.map((c) => c.items.length));
  return {
    box,
    columns,
    width: Math.max(0, x - LATTICE_COL_GAP_PX),
    height: rowCount * LATTICE_ROW_PX + LATTICE_NODE_H,
  };
}

/** 選択中のノードをすべて通る経路だけを残す。 */
function activePaths() {
  if (!latticeData) return [];
  if (!latticeSelection.size) return latticeData.paths;
  return latticeData.paths.filter((path) => {
    const ids = new Set(path.nodes);
    for (const selected of latticeSelection) {
      if (!ids.has(selected)) return false;
    }
    return true;
  });
}

/** ラティスを描く。
 *
 * ノードは div、辺は背後に敷いた SVG。辺を DOM 要素で描くと曲線が引けず、
 * ノードを SVG に入れるとテキストの折り返しが効かないので、両方の都合を
 * 取ってこの組み合わせにしている。 */
function renderLattice() {
  const host = $("lattice");
  if (!latticeData) {
    host.replaceChildren();
    return;
  }
  const { nodes, edges } = latticeData;
  const { box, width, height } = layoutLattice(nodes, edges);

  // 絞り込みで残る経路。ノードと辺の強調に使う。
  const paths = activePaths();
  const liveNodes = new Set();
  const liveEdges = new Set();
  for (const path of paths) {
    path.nodes.forEach((id) => liveNodes.add(id));
    for (let i = 0; i < path.nodes.length - 1; i += 1) {
      liveEdges.add(`${path.nodes[i]} ${path.nodes[i + 1]}`);
    }
    liveEdges.add(` ${path.nodes[0]}`);
    liveEdges.add(`${path.nodes[path.nodes.length - 1]} `);
  }

  host.style.width = `${width}px`;
  host.style.height = `${height}px`;

  // --- 辺 (SVG) ---
  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("class", "lattice-edges");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  // 辺はノードの右端から次のノードの左端へ。列の間隔ぶんの隙間に引く。
  const anchor = (id) => {
    const geometry = box.get(id);
    return {
      left: geometry.x,
      right: geometry.x + geometry.w,
      y: geometry.y + LATTICE_NODE_H / 2,
    };
  };
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const maxEdge = Math.max(1, ...edges.map((e) => e.path_count));
  for (const edge of edges) {
    if (edge.source && !byId.has(edge.source)) continue;
    if (edge.target && !byId.has(edge.target)) continue;
    const key = `${edge.source || ""} ${edge.target || ""}`;
    const live = liveEdges.has(key);

    // 始端と終端は図の外側に小さく突き出す。どのノードが文頭・文末になり得るか。
    const from = edge.source ? anchor(edge.source) : null;
    const to = edge.target ? anchor(edge.target) : null;
    const x1 = from ? from.right : to.left - 14;
    const y1 = from ? from.y : to.y;
    const x2 = to ? to.left : from.right + 14;
    const y2 = to ? to.y : from.y;

    const line = document.createElementNS(svgNS, "path");
    // 縦の差を水平の立ち上がりで吸収するベジエ。制御点を端に寄せると辺が
    // ノードの縁から横向きに出るので、束になった辺の行き先が見分けやすい。
    const dx = Math.max(18, (x2 - x1) * 0.55);
    line.setAttribute("d", `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`);
    line.setAttribute("class", live ? "lattice-edge is-live" : "lattice-edge");
    // ノードにカーソルを当てたときにその辺だけを強調するための目印。
    // 縦に離れた辺は形だけでは追えないので、CSS 側で色を変える。
    if (edge.source) line.dataset.from = edge.source;
    if (edge.target) line.dataset.to = edge.target;
    // 経路数を太さに写す。よく通る繋がりが太くなる。
    line.setAttribute("stroke-width", String(0.8 + (edge.path_count / maxEdge) * 2.4));
    svg.append(line);
  }

  // --- ノード ---
  const layer = document.createElement("div");
  layer.className = "lattice-nodes";
  const maxNode = Math.max(1, ...nodes.map((n) => n.path_count));
  // 隣接するノード。カーソルを当てたときに繋がる相手を浮かせるのに使う。
  const neighbours = new Map();
  for (const edge of edges) {
    if (!edge.source || !edge.target) continue;
    if (!neighbours.has(edge.source)) neighbours.set(edge.source, new Set());
    if (!neighbours.has(edge.target)) neighbours.set(edge.target, new Set());
    neighbours.get(edge.source).add(edge.target);
    neighbours.get(edge.target).add(edge.source);
  }
  for (const node of nodes) {
    const geometry = box.get(node.id);
    const el = document.createElement("button");
    el.type = "button";
    el.className = "lattice-node";
    if (node.is_particle) el.classList.add("is-particle");
    if (latticeSelection.has(node.id)) el.classList.add("is-selected");
    if (!liveNodes.has(node.id)) el.classList.add("is-dim");
    el.style.left = `${geometry.x}px`;
    el.style.width = `${geometry.w}px`;
    el.style.top = `${geometry.y}px`;
    // 経路数を地の濃さに写す。よく使われる語が目に入る。
    el.style.setProperty("--weight", (node.path_count / maxNode).toFixed(3));
    el.setAttribute("aria-pressed", String(latticeSelection.has(node.id)));
    el.title =
      `${node.source_reading} → ${node.surface} (${node.reading})\n` +
      `入力の ${node.start + 1}-${node.end} 拍 / 音韻類似度 ${node.similarity.toFixed(3)}\n` +
      `${node.path_count} 経路が通る` +
      (node.is_particle ? " / 助詞" : "");

    const surface = document.createElement("span");
    surface.className = "lattice-node-surface";
    surface.textContent = node.surface;
    // 入力側の読み。モーラの目盛りを外したので、どこを覆っているかはこれで読ませる。
    const meta = document.createElement("span");
    meta.className = "lattice-node-meta";
    meta.textContent = node.source_reading;
    el.append(surface, meta);

    el.addEventListener("click", () => {
      // 同じノードをもう一度押すと外す。複数選ぶと AND で絞る。
      if (latticeSelection.has(node.id)) latticeSelection.delete(node.id);
      else latticeSelection.add(node.id);
      renderLattice();
      renderLatticePaths();
    });

    // そのノードに繋がる辺だけを強調する。縦に離れた辺は曲線の形だけでは
    // 追えないので、当てているあいだ色を変えて行き先を示す。
    const spotlight = (on) => {
      for (const path of svg.querySelectorAll("path")) {
        const touches = path.dataset.from === node.id || path.dataset.to === node.id;
        path.classList.toggle("is-hot", on && touches);
        path.classList.toggle("is-cold", on && !touches);
      }
      // 繋がる相手のノードも一緒に浮かせる。
      for (const other of layer.children) {
        if (!on) {
          other.classList.remove("is-near");
          continue;
        }
        const id = other.dataset.nodeId;
        other.classList.toggle("is-near", neighbours.get(node.id)?.has(id) || id === node.id);
      }
    };
    el.dataset.nodeId = node.id;
    el.addEventListener("mouseenter", () => spotlight(true));
    el.addEventListener("mouseleave", () => spotlight(false));
    el.addEventListener("focus", () => spotlight(true));
    el.addEventListener("blur", () => spotlight(false));
    layer.append(el);
  }

  host.replaceChildren(svg, layer);
}

/** 絞り込んだ経路を図の下に並べる。図だけでは「結局どう読むのか」が出ないため。 */
function renderLatticePaths() {
  const host = $("lattice-paths");
  const paths = activePaths();
  const clear = $("lattice-clear");
  clear.hidden = latticeSelection.size === 0;

  const note = $("lattice-note");
  if (latticeData) {
    const parts = [
      `${latticeData.node_count} ノード / ${latticeData.path_count} 経路`,
      `ビーム幅 ${latticeData.beam_width}`,
    ];
    if (latticeData.truncated) parts.push("予算のため一部の経路のみ");
    if (latticeSelection.size) parts.push(`選択中 ${latticeSelection.size} — ${paths.length} 経路`);
    else parts.push("カーソルを当てると繋がる先が光る / 押すとその語を通る経路に絞る");
    note.textContent = parts.join(" · ");
  }

  host.replaceChildren(
    ...paths.slice(0, 20).map((path) => {
      const row = document.createElement("div");
      row.className = "lattice-path";
      const score = document.createElement("span");
      score.className = "lattice-path-score";
      score.textContent = path.score.toFixed(3);
      const text = document.createElement("span");
      text.className = "lattice-path-text";
      text.textContent = path.text;
      const reading = document.createElement("span");
      reading.className = "lattice-path-reading";
      reading.textContent = path.reading;
      row.append(score, text, reading);
      return row;
    }),
  );
}

async function runPhrase(event) {
  event?.preventDefault();
  const text = $("ptext-phrase").value.trim();
  if (!text) return;

  const status = $("phrase-status");
  const button = event?.target?.querySelector?.(".run") || null;
  if (button) button.disabled = true;
  setStatus(status, "合成中…");
  // 共通のパラメータ。2 つの経路で同じ条件を使う。
  const shared = {
    text,
    max_chunk_moras: $("ph-max-chunk").value,
    chunk_candidates: $("ph-chunk-candidates").value,
    beam_width: $("ph-beam").value,
    min_chunk_score: $("ph-min-chunk").value,
    allow_particles: $("ph-particles").checked ? "true" : "false",
  };
  const lattice = phraseView === "lattice";
  try {
    const started = performance.now();
    const data = lattice
      ? await getJSON("/api/phrase/lattice", {
          ...shared,
          node_budget: $("ph-node-budget").value,
        })
      : await getJSON("/api/phrase", { ...shared, limit: $("ph-limit").value });
    const elapsed = Math.round(performance.now() - started);

    $("rp-reading").textContent = data.reading || "(読みが取れない)";
    $("rp-ipa").textContent = data.ipa ? `[${data.ipa}]` : "—";
    $("rp-mora").textContent = `${data.mora_count} モーラ`;
    $("phrase-readout").hidden = false;

    // 表示するのは片方だけ。切り替えのたびに引き直すので、もう一方は消す。
    $("phrase-results").hidden = lattice;
    $("lattice-wrap").hidden = !lattice;

    if (lattice) {
      latticeSelection.clear();
      latticeData = data.nodes.length ? data : null;
      renderLattice();
      renderLatticePaths();
      if (!data.nodes.length) {
        setStatus(status, `該当なし (${elapsed}ms) — 区間スコア下限を下げると候補が増える`);
        return;
      }
      setStatus(status, `${data.node_count} ノード / ${data.path_count} 経路 (${elapsed}ms)`);
      return;
    }

    const results = $("phrase-results");
    if (!data.results.length) {
      results.replaceChildren();
      setStatus(status, `該当なし (${elapsed}ms) — 区間スコア下限を下げると候補が増える`);
      return;
    }
    results.replaceChildren(...data.results.map((c, i) => phraseCard(c, i + 1)));
    setStatus(status, `${data.results.length} 件 (${elapsed}ms)`);
  } catch (error) {
    setStatus(status, error.message, true);
    $("phrase-results").replaceChildren();
    $("lattice-wrap").hidden = true;
    $("phrase-readout").hidden = true;
  } finally {
    if (button) button.disabled = false;
  }
}

/* ---------- 読み ---------- */

async function runPronounce(event) {
  event?.preventDefault();
  const text = $("ptext").value.trim();
  if (!text) return;

  const status = $("pronounce-status");
  setStatus(status, "解析中…");
  try {
    const data = await getJSON("/api/pronounce", { text });
    $("pa-text").textContent = data.text;
    $("pa-reading").textContent = data.reading || "(読みが取れない)";
    $("pa-count").textContent = `${data.mora_count} モーラ / ${data.phonemes.length} 音素`;
    $("pa-ipa").textContent = data.ipa ? `[${data.ipa}]` : "—";

    const moras = $("pa-moras");
    moras.replaceChildren(
      ...data.moras.map((kana) => {
        const el = document.createElement("span");
        el.className = SPECIAL_LABELS[kana] ? "mora is-special" : "mora";
        el.textContent = kana;
        if (SPECIAL_LABELS[kana]) el.title = SPECIAL_LABELS[kana];
        return el;
      }),
    );
    renderPhonemes($("pa-phonemes"), data.phonemes);
    renderPhonemes($("pa-skeleton"), data.vowel_skeleton);

    $("pronounce-out").hidden = false;
    setStatus(status, "");
  } catch (error) {
    setStatus(status, error.message, true);
    $("pronounce-out").hidden = true;
  }
}

/* ---------- 索引 ---------- */

let infoLoaded = false;

async function loadInfo() {
  if (infoLoaded) return;
  const status = $("info-status");
  setStatus(status, "読み込み中…");
  try {
    const data = await getJSON("/api/info");
    const meta = $("info-meta");
    meta.replaceChildren(
      ...[
        ["語数", data.count.toLocaleString("ja-JP")],
        ["形式バージョン", `v${data.format_version}`],
        ["場所", data.path],
      ].map(([label, value]) => {
        const wrap = document.createElement("div");
        const dt = document.createElement("dt");
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.textContent = value;
        wrap.append(dt, dd);
        return wrap;
      }),
    );

    const table = $("info-spaces");
    const head = document.createElement("tr");
    for (const label of ["空間", "次元", "役割"]) {
      const th = document.createElement("th");
      th.textContent = label;
      head.append(th);
    }
    table.replaceChildren(
      head,
      ...data.spaces.map((space) => {
        const tr = document.createElement("tr");
        for (const value of [space.name, String(space.dim), space.role]) {
          const td = document.createElement("td");
          td.textContent = value;
          tr.append(td);
        }
        return tr;
      }),
    );

    const total = Math.max(1, ...data.categories.map((c) => c.count));
    $("info-cats").replaceChildren(
      ...data.categories.map((category) => {
        const row = document.createElement("div");
        row.className = "bar-row";
        if (NON_DEFAULT_CATEGORIES.has(category.name)) row.classList.add("is-excluded");
        const label = document.createElement("span");
        label.textContent = CATEGORY_LABELS[category.name] || category.name;
        const track = document.createElement("span");
        track.className = "bar-track";
        const fill = document.createElement("span");
        fill.className = "bar-fill";
        fill.style.width = `${(category.count / total) * 100}%`;
        track.append(fill);
        const num = document.createElement("span");
        num.className = "bar-num";
        num.textContent = category.count.toLocaleString("ja-JP");
        row.append(label, track, num);
        if (NON_DEFAULT_CATEGORIES.has(category.name)) {
          row.title = "既定では検索対象から外れる (音韻的に密集して一般語を押し出すため)";
        }
        return row;
      }),
    );

    $("info-out").hidden = false;
    setStatus(status, "");
    infoLoaded = true;
  } catch (error) {
    setStatus(status, error.message, true);
  }
}

/* ---------- カテゴリのチップと凡例 ---------- */

function renderCategoryChips(categories) {
  $("cat-group").replaceChildren(
    ...categories.map((category) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.setAttribute("aria-pressed", "false");
      chip.append(CATEGORY_LABELS[category.name] || category.name);
      const count = document.createElement("span");
      count.className = "chip-count";
      count.textContent = category.count.toLocaleString("ja-JP");
      chip.append(count);
      chip.addEventListener("click", () => {
        const on = chip.classList.toggle("is-on");
        chip.setAttribute("aria-pressed", String(on));
        if (on) activeCategories.add(category.name);
        else activeCategories.delete(category.name);
      });
      return chip;
    }),
  );
}

function renderLegend() {
  const places = $("legend-places");
  places.replaceChildren(
    ...Object.entries(PLACE_LABELS).map(([place, label]) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      const dot = document.createElement("span");
      dot.className = "legend-dot";
      dot.style.background = `var(--p-${place})`;
      item.append(dot, label);
      return item;
    }),
  );

  const vowels = $("legend-vowels");
  vowels.replaceChildren(
    ...Object.keys(features.vowels).map((symbol) => {
      const item = document.createElement("span");
      item.className = "legend-item";
      const dot = document.createElement("span");
      dot.className = "legend-dot";
      dot.style.background = `var(--v-${symbol})`;
      // /u/ のように記号と IPA が食い違う母音があるので、凡例では常に併記する。
      const head = ipaDiffers(symbol) ? `${symbol} [${phonemeIpa(symbol)}]` : symbol;
      item.append(dot, `${head} ${VOWEL_LABELS[symbol] || ""}`.trim());
      return item;
    }),
  );
}

/* ---------- タブ ---------- */

function selectView(name) {
  for (const tab of document.querySelectorAll(".tab")) {
    const on = tab.dataset.view === name;
    tab.classList.toggle("is-active", on);
    tab.setAttribute("aria-selected", String(on));
  }
  for (const view of document.querySelectorAll(".view")) {
    view.classList.toggle("is-active", view.id === `view-${name}`);
  }
  if (name === "info") loadInfo();
}

/* ---------- 起動 ---------- */

async function main() {
  $("search-form").addEventListener("submit", runSearch);
  $("phrase-form").addEventListener("submit", runPhrase);
  $("pronounce-form").addEventListener("submit", runPronounce);

  // 一覧 / ラティスの切り替え。同じ経路集合の別の見せ方なので、条件は変えずに
  // 引き直す。`.seg` は検索タブのプリセットと同じクラスなので、この
  // グループの中だけを見る。
  for (const seg of $("phrase-view-group").querySelectorAll(".seg")) {
    seg.addEventListener("click", () => {
      phraseView = seg.dataset.phraseView;
      for (const other of $("phrase-view-group").querySelectorAll(".seg")) {
        const on = other === seg;
        other.classList.toggle("is-active", on);
        other.setAttribute("aria-checked", String(on));
      }
      if ($("ptext-phrase").value.trim()) runPhrase();
    });
  }

  $("lattice-clear").addEventListener("click", () => {
    latticeSelection.clear();
    renderLattice();
    renderLatticePaths();
  });

  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => selectView(tab.dataset.view));
  }

  // 幅が変わると入る枠数が変わる。枠数が変わったときだけ描き直す —
  // 1px 単位の resize ごとに 200 行を作り直すと重い。
  let lastSlots = 0;
  new ResizeObserver((entries) => {
    const width = entries[0].contentRect.width;
    const slots = Math.floor(width / slotPx());
    if (slots === lastSlots) return;
    lastSlots = slots;
    relayoutResults();
  }).observe($("results"));

  // IPA の併記。素性表を読む前に押されても困らないよう、状態だけ持って
  // 描画は下の 2 つに任せる。
  const ipaToggle = $("ipa-toggle");
  const applyIpaToggle = () => {
    ipaToggle.classList.toggle("is-on", showIpa);
    ipaToggle.setAttribute("aria-pressed", String(showIpa));
    // 併記でチップの幅が変わるので、結果は枠から配り直す (SLOT_PX_IPA)。
    // チップだけ差し替えると枠が足りずに折り返す。
    relayoutResults();
    // 結果の外 (クエリの読み・読みビュー・開いている対応帯) は枠を持たないので
    // チップだけ組み直す。
    rerenderAllPhonemes();
  };
  ipaToggle.addEventListener("click", () => {
    showIpa = !showIpa;
    // 表示の好みは検索条件と違って共有する対象ではないので URL ではなく
    // localStorage に置く。
    try {
      localStorage.setItem("jpr.showIpa", showIpa ? "1" : "");
    } catch {
      // プライベートモードでは書けない。表示は切り替わるので黙って続ける。
    }
    applyIpaToggle();
  });
  try {
    showIpa = localStorage.getItem("jpr.showIpa") === "1";
  } catch {
    showIpa = false;
  }
  applyIpaToggle();

  // **プリセットのグループの中だけを見る。** `.seg` は分割合成ビューの
  // 見せ方の切り替えにも使っているので、文書全体から引くとあちらを押したときに
  // activePreset が undefined になり、検索が壊れる。
  const presetGroup = $("preset-group");
  for (const seg of presetGroup.querySelectorAll(".seg")) {
    seg.addEventListener("click", () => {
      activePreset = seg.dataset.preset;
      for (const other of presetGroup.querySelectorAll(".seg")) {
        const on = other === seg;
        other.classList.toggle("is-active", on);
        other.setAttribute("aria-checked", String(on));
      }
      if ($("q").value.trim()) runSearch();
    });
  }

  // 素性表は音素チップの色の根拠なので、描画の前に読む。
  try {
    features = await getJSON("/api/phonemes");
    renderLegend();
  } catch (error) {
    setStatus($("search-status"), `素性表を読めません: ${error.message}`, true);
  }

  // 語数とカテゴリは索引を開いて初めて分かる。索引が無い環境では
  // 検索を試みたときに 503 の理由が出るので、ここでは静かに諦める。
  try {
    const info = await getJSON("/api/info");
    $("corpus-note").textContent = `${info.count.toLocaleString("ja-JP")} 語 / SudachiDict full`;
    $("candidates").value = String(info.default_candidates);
    defaultMinScore = info.default_min_score;
    $("min-score").value = String(defaultMinScore);
    renderCategoryChips(info.categories);
    // 分割合成の既定値もサーバから受け取る。phrase.py の定数を変えたときに
    // 画面の初期値が黙ってずれないようにする (音素の色と同じ理由)。
    if (info.phrase) {
      $("ph-max-chunk").value = String(info.phrase.max_chunk_moras);
      $("ph-chunk-candidates").value = String(info.phrase.chunk_candidates);
      $("ph-beam").value = String(info.phrase.beam_width);
      $("ph-min-chunk").value = String(info.phrase.min_chunk_score);
      $("ph-limit").max = String(info.phrase.max_limit);
      $("ph-node-budget").value = String(info.phrase.node_budget);
      $("ph-node-budget").max = String(info.phrase.max_node_budget);
    }
  } catch {
    $("corpus-note").textContent = "索引が未構築 — `jpr build-index` で構築する";
  }

  const params = new URLSearchParams(location.search);
  const preset = params.get("preset");
  if (preset) {
    const seg = presetGroup.querySelector(`.seg[data-preset="${CSS.escape(preset)}"]`);
    if (seg) seg.click();
  }
  const minMora = params.get("min_mora");
  const maxMora = params.get("max_mora");
  if (minMora) $("min-mora").value = minMora;
  if (maxMora) $("max-mora").value = maxMora;
  // 件数も URL から復元する。列の詰まり方は件数で変わるので、
  // これが効かないと共有された URL が同じ画面を再現しない。
  const limit = params.get("limit");
  if (limit) $("limit").value = limit;
  // スコア下限も同じ理由で復元する。**info の後に読む** — あちらが既定を
  // 書き込むので、順序を逆にすると共有された値が既定で上書きされる。
  const minScore = params.get("min_score");
  if (minScore) $("min-score").value = minScore;
  // 共有された URL でモーラ範囲が効いているなら、絞り込みを開いて見せる。
  // **検索ビューの中を指定する** — `.advanced` は分割合成ビューにもあるので、
  // 文書全体から引くと先に出てくる別のビューの詳細が開く。
  if (minMora || maxMora) {
    document.querySelector("#view-search .advanced")?.setAttribute("open", "");
  }

  const query = params.get("q");
  if (query) {
    $("q").value = query;
    runSearch();
  } else {
    $("q").focus();
  }
}

main();

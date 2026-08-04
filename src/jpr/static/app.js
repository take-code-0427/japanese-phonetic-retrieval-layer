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

/** 音素の素性表。/api/phonemes で埋める。 */
let features = { consonants: {}, vowels: {}, special: {} };

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

/** 音素の読み方の説明。チップの title に入れる。 */
function phonemeTitle(symbol) {
  const consonant = features.consonants[symbol];
  if (consonant) {
    const parts = [PLACE_LABELS[consonant.place] || consonant.place, consonant.manner];
    parts.push(consonant.voiced ? "有声" : "無声");
    if (consonant.palatalized) parts.push("口蓋化");
    return `${symbol} — ${parts.join(" / ")}`;
  }
  if (features.vowels[symbol]) return `${symbol} — 母音 ${VOWEL_LABELS[symbol] || ""}`.trim();
  if (SPECIAL_LABELS[symbol]) return `${symbol} — ${SPECIAL_LABELS[symbol]}`;
  return symbol;
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
  return el;
}

function renderPhonemes(container, symbols) {
  container.replaceChildren(...symbols.map(phonemeChip));
}

/* ---------- 検索 ---------- */

let activePreset = "pun";
const activeCategories = new Set();

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

/** スコア棒の長さ。結果内の最小〜最大に張り直す。
 *
 * 音韻スコアは上位が 0.90〜0.98 に密集するので、0〜1 をそのまま幅に写すと
 * 全部が満杯に見えて順位の差が読めない。棒は結果同士の相対差を示す役目に絞り、
 * 絶対値は隣の数値で読ませる。 */
function scoreBarWidth(score, min, max) {
  if (max - min < 1e-6) return 100;
  return 8 + ((score - min) / (max - min)) * 92;
}

/** 結果 1 件。クリックで音素アライメントを開く。 */
function resultRow(result, queryPhonemes, rank, range) {
  const li = document.createElement("li");
  li.className = "result";

  const button = document.createElement("button");
  button.className = "result-open";
  button.type = "button";
  button.setAttribute("aria-expanded", "false");

  const scoreCell = document.createElement("div");
  const score = document.createElement("span");
  score.className = "score";
  score.textContent = result.score.toFixed(3);
  const bar = document.createElement("span");
  bar.className = "score-bar";
  bar.style.width = `${scoreBarWidth(result.score, range.min, range.max)}%`;
  bar.title = "棒はこの結果内での相対差。絶対値は数値で読む";
  scoreCell.append(score, bar);

  const wordCell = document.createElement("div");
  wordCell.className = "result-word";
  const word = document.createElement("span");
  word.className = "word";
  word.textContent = result.word;
  const reading = document.createElement("span");
  reading.className = "word-reading";
  reading.textContent = result.reading;
  const phonemes = document.createElement("div");
  phonemes.className = "phonemes result-phonemes";
  renderPhonemes(phonemes, result.phonemes);
  wordCell.append(word, reading, phonemes);

  const tags = document.createElement("div");
  tags.className = "result-tags";
  const mora = document.createElement("span");
  mora.className = "tag tag-mora";
  mora.textContent = `${result.mora_count} モーラ`;
  const category = document.createElement("span");
  category.className = "tag";
  category.textContent = CATEGORY_LABELS[result.category] || result.category;
  tags.append(mora, category, familiarityMeter(result.familiarity));

  // 4 列目は余白。タグを右端まで飛ばさず語の隣に留めるための受け皿。
  button.append(scoreCell, wordCell, tags, document.createElement("span"));
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
    } catch (error) {
      panel.textContent = `対応付けを取得できません: ${error.message}`;
    }
  });

  li.dataset.rank = String(rank);
  return li;
}

/** 署名要素: 音素の対応付けを縦 2 段で並べ、対ごとの素性距離を縦棒で示す。 */
function alignmentNodes(data, result) {
  const title = document.createElement("p");
  title.className = "align-title";
  title.textContent = "音素の対応 — 上がクエリ、下が候補。縦棒は素性距離";

  const track = document.createElement("div");
  track.className = "align-track";

  // 距離を高さに写す基準。子音と母音の置換 (1.0) が満杯になるよう固定し、
  // 行ごとに最大値で正規化しない。行間で高さが比較できなくなるため。
  const GAUGE_MAX = 1.0;
  const GAUGE_PX = 30;

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

  return [title, track, foot];
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
    $("rq-mora").textContent = `${data.mora_count}`;
    renderPhonemes($("rq-phonemes"), data.phonemes);
    $("query-readout").hidden = false;

    const scores = data.results.map((r) => r.score);
    const range = { min: Math.min(...scores), max: Math.max(...scores) };
    const list = $("results");
    list.replaceChildren(
      ...data.results.map((result, i) => resultRow(result, data.phonemes, i + 1, range)),
    );

    if (!data.results.length) {
      // 全走査では候補数を増やしても何も変わらないので、助言を変える。
      const advice = data.scanned == null
        ? "候補数を増やすか、カテゴリの絞り込みを外す"
        : "モーラ範囲かスコア下限をゆるめる";
      setStatus(status, `該当なし (${elapsed}ms) — ${advice}`);
    } else {
      const parts = [`${data.results.length} 件`, `${elapsed}ms`];
      if (data.scanned != null) {
        parts.push(`${data.scanned.toLocaleString("ja-JP")} 語を全走査`);
      }
      if (data.truncated) {
        parts.push(`全 ${data.total.toLocaleString("ja-JP")} 件中の先頭のみ`);
      }
      setStatus(status, `${parts.join(" / ")} — 行をクリックすると音素の対応が開く`);
    }

    // モーラ範囲も URL に載せて、絞り込んだ結果を共有できるようにする。
    const shared = new URLSearchParams({ q: query, preset: activePreset });
    if ($("min-mora").value) shared.set("min_mora", $("min-mora").value);
    if ($("max-mora").value) shared.set("max_mora", $("max-mora").value);
    history.replaceState(null, "", `?${shared}`);
  } catch (error) {
    setStatus(status, error.message, true);
    $("results").replaceChildren();
    $("query-readout").hidden = true;
  } finally {
    button.disabled = false;
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
        ["辞書", `SudachiDict ${data.dict_type}`],
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
      item.append(dot, `${symbol} ${VOWEL_LABELS[symbol] || ""}`.trim());
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
  $("pronounce-form").addEventListener("submit", runPronounce);

  for (const tab of document.querySelectorAll(".tab")) {
    tab.addEventListener("click", () => selectView(tab.dataset.view));
  }

  for (const seg of document.querySelectorAll(".seg")) {
    seg.addEventListener("click", () => {
      activePreset = seg.dataset.preset;
      for (const other of document.querySelectorAll(".seg")) {
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
    $("corpus-note").textContent =
      `${info.count.toLocaleString("ja-JP")} 語 / SudachiDict ${info.dict_type}`;
    $("candidates").value = String(info.default_candidates);
    renderCategoryChips(info.categories);
  } catch {
    $("corpus-note").textContent = "索引が未構築 — `jpr build-index` で構築する";
  }

  const params = new URLSearchParams(location.search);
  const preset = params.get("preset");
  if (preset) {
    const seg = document.querySelector(`.seg[data-preset="${CSS.escape(preset)}"]`);
    if (seg) seg.click();
  }
  const minMora = params.get("min_mora");
  const maxMora = params.get("max_mora");
  if (minMora) $("min-mora").value = minMora;
  if (maxMora) $("max-mora").value = maxMora;
  // 共有された URL でモーラ範囲が効いているなら、絞り込みを開いて見せる。
  if (minMora || maxMora) document.querySelector(".advanced")?.setAttribute("open", "");

  const query = params.get("q");
  if (query) {
    $("q").value = query;
    runSearch();
  } else {
    $("q").focus();
  }
}

main();

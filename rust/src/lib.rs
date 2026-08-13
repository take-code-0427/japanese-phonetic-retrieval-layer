//! 重み付き音素編集距離のバッチ計算。
//!
//! Python 側 (`jpr.distance.edit_distance_batch`) と同じ値を返さなければならない。
//! コスト表は Python 側で素性表から構成したものを受け取るので、素性の重みを
//! 変えてもこちらを直す必要はない。表を持たずに渡してもらうのが要点で、
//! 定義を二重に持つと静かに食い違う。
//!
//! NumPy 版は「候補方向をベクトル化し、DP の 1 行を全候補ぶん同時に進める」形を
//! とっていた。Python では 1 件ずつ NumPy を呼ぶ呼び出しオーバーヘッドが実計算を
//! 上回るためで、そこは避けようがない。Rust では逆にする — 1 候補の DP を逐次で
//! 回し、候補方向を rayon で並列化する。音素列は 3〜24 要素なので DP の 2 行
//! (最大 25 要素 x 4 バイト) が L1 に収まり、候補ごとに閉じた計算になる。
//! NumPy 版のように (クエリ長 x 候補数 x 音素長) の置換コストテンソルを
//! 実体化する必要がなくなる (53 万候補で 306MB、float32 でも 153MB)。

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

mod gemv;

/// パディングで「音素なし」を表す値。Python 側の `PAD_ID` と一致させる。
const PAD_ID: i64 = -1;

/// 1 候補ぶんの DP。`query` と `candidate` はどちらも距離テーブルの音素 ID。
///
/// `substitution` は (P, P) の置換コストを行優先で平坦化したもの、`indel` は
/// 音素ごとの挿入削除コスト。`phoneme_count` は P。
#[inline]
fn edit_distance_one(
    query: &[i32],
    candidate: &[i32],
    substitution: &[f32],
    indel: &[f32],
    phoneme_count: usize,
    // DP の 2 行を呼び出し側から借りる。候補ごとに確保すると割り当てが支配的になる。
    previous: &mut [f32],
    current: &mut [f32],
) -> f32 {
    let m = candidate.len();
    if query.is_empty() {
        // クエリが空なら候補を全削除するコスト。
        return candidate.iter().map(|&p| indel[p as usize]).sum();
    }
    if m == 0 {
        return query.iter().map(|&p| indel[p as usize]).sum();
    }

    // 0 行目: 候補側を先頭から挿入していくコストの累積。
    previous[0] = 0.0;
    for j in 0..m {
        previous[j + 1] = previous[j] + indel[candidate[j] as usize];
    }

    // 2 行を入れ替えながら進める。行の複製 (`copy_from_slice`) を挟むと
    // クエリ音素ごとに m+1 要素を書き戻すことになり、DP 本体と同じ量の
    // 書き込みが増える。参照を差し替えれば同じ結果が複製なしで得られる。
    let mut previous: &mut [f32] = previous;
    let mut current: &mut [f32] = current;

    for &qp in query {
        let q_indel = indel[qp as usize];
        // 置換コスト表のうち、このクエリ音素に対応する行。
        let sub_row = &substitution[(qp as usize) * phoneme_count..][..phoneme_count];
        current[0] = previous[0] + q_indel;
        for j in 0..m {
            let cp = candidate[j] as usize;
            // 置換と削除は前の行だけに依存する。
            let substitute = previous[j] + sub_row[cp];
            let delete = previous[j + 1] + q_indel;
            let mut best = if substitute < delete { substitute } else { delete };
            // 挿入は同じ行の左隣に依存するため、この順でしか解けない。
            let insert = current[j] + indel[cp];
            if insert < best {
                best = insert;
            }
            current[j + 1] = best;
        }
        std::mem::swap(&mut previous, &mut current);
    }

    // 入れ替えた直後なので、最後に書いた行は `previous` のほう。
    previous[m]
}

/// パディング行列を受ける経路。NumPy 版と同じ引数なので差し替えの検証に使える。
#[pyfunction]
fn edit_distance_batch<'py>(
    py: Python<'py>,
    query_ids: PyReadonlyArray1<'py, i32>,
    candidates: PyReadonlyArray2<'py, i64>,
    lengths: PyReadonlyArray1<'py, i64>,
    substitution: PyReadonlyArray1<'py, f32>,
    indel: PyReadonlyArray1<'py, f32>,
    phoneme_count: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query = query_ids.as_slice()?;
    let matrix = candidates.as_array();
    let lengths = lengths.as_slice()?;
    let substitution = substitution.as_slice()?;
    let indel = indel.as_slice()?;

    let count = lengths.len();
    let width = matrix.shape()[1];

    // 行ごとの音素 ID を平坦な Vec に詰め直す。パディングはここで落とす。
    let mut flat: Vec<i32> = Vec::with_capacity(count * width);
    let mut offsets: Vec<usize> = Vec::with_capacity(count + 1);
    offsets.push(0);
    for row in 0..count {
        for column in 0..(lengths[row] as usize) {
            let value = matrix[[row, column]];
            if value != PAD_ID {
                flat.push(value as i32);
            }
        }
        offsets.push(flat.len());
    }

    let distances = py.detach(|| {
        compute(&flat, &offsets, query, substitution, indel, phoneme_count, width)
    });
    Ok(PyArray1::from_vec(py, distances))
}

/// CSR を直接受ける経路。**こちらが検索の本線。**
///
/// 索引は音素列を連結した 1 本の配列と境界インデックスで持っている
/// (`store.py` の `_encode_entries`)。パディング行列を経由せずにそれを直接
/// 読めば、Python 側で (候補数, 音素長) の行列を組む手間が消える
/// (実測 53 万候補で 145ms)。
///
/// `phoneme_ids` は索引内の音素 ID、`distance_ids` はそれを距離テーブルの ID に
/// 写す表。索引の語彙順は構築時の出現順なので、この写像が必要になる。
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn edit_distance_csr<'py>(
    py: Python<'py>,
    query_ids: PyReadonlyArray1<'py, i32>,
    rows: PyReadonlyArray1<'py, i64>,
    phoneme_ids: PyReadonlyArray1<'py, u8>,
    // 索引の境界は int32 (`store.py` の `_encode_strings`)。int64 を要求すると
    // 呼び出しごとに 16MB の変換コピーが走る。
    phoneme_bounds: PyReadonlyArray1<'py, i32>,
    distance_ids: PyReadonlyArray1<'py, i32>,
    substitution: PyReadonlyArray1<'py, f32>,
    indel: PyReadonlyArray1<'py, f32>,
    phoneme_count: usize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let query = query_ids.as_slice()?;
    let rows = rows.as_slice()?;
    let phoneme_ids = phoneme_ids.as_slice()?;
    let bounds = phoneme_bounds.as_slice()?;
    let distance_ids = distance_ids.as_slice()?;
    let substitution = substitution.as_slice()?;
    let indel = indel.as_slice()?;

    // 索引の ID から距離テーブルの ID への写像をここで済ませ、平坦に詰める。
    // 行の並びは呼び出し側の `rows` 順で、mmap 上を飛び飛びに読むのはこの 1 回だけ。
    let count = rows.len();
    let mut offsets: Vec<usize> = Vec::with_capacity(count + 1);
    offsets.push(0);
    let mut total = 0usize;
    for &row in rows {
        let row = row as usize;
        total += (bounds[row + 1] - bounds[row]) as usize;
        offsets.push(total);
    }
    let mut flat: Vec<i32> = Vec::with_capacity(total);
    let mut width = 0usize;
    for &row in rows {
        let row = row as usize;
        let start = bounds[row] as usize;
        let end = bounds[row + 1] as usize;
        if end - start > width {
            width = end - start;
        }
        for &id in &phoneme_ids[start..end] {
            flat.push(distance_ids[id as usize]);
        }
    }

    let distances = py.detach(|| {
        compute(&flat, &offsets, query, substitution, indel, phoneme_count, width)
    });
    Ok(PyArray1::from_vec(py, distances))
}

/// クエリの音素列を「完全な形で」含むグループを索引全体から拾う。
///
/// 編集距離では表現できない性質を測るために別の経路にしてある。距離は挿入を
/// 一律に減点するので、クエリを丸ごと含む語 (riNgo in riNgoku、類似度 0.735) が
/// 1 音素だけ違う同じ長さの語 (riNbo、0.933) に必ず負ける。「入っているかどうか」は
/// 連続一致という離散的な性質なので、距離の重みを緩めて近似するのではなく
/// 判定として持つ。
///
/// **特殊モーラ (長音・促音・撥音) の挿入だけは無料。** 「リンゴー」「リンゴッ」は
/// 音として riNgo を完全に含んでいるが、記号列としては一致しない。この 3 つは
/// 単独で音素を成さず後続や前続の伸縮として実現するので、挿入されていても
/// 「元の音が完全な形で入っている」という感覚は保たれる。それ以外の音素が
/// 挟まれば別の音になるので即失敗させる。
///
/// 返すのは (グループ ID, 占有率)。占有率はクエリの音素数を候補の音素数で
/// 割った値で、余分が多い語ほど下がる。`riNgo` は `riNgoku` (7 音素) で 0.71、
/// `riNgojuRsu` (10 音素) で 0.50。長い地名がクエリを含むだけで上位に来るのを
/// 抑える。**一致に消費した長さではなく候補全体の長さで割る** — 分母を一致長に
/// すると特殊モーラを挟んだ語が 1.0 になり、余分の多寡が消える。
///
/// 走査は索引の音素 CSR 全体に対して行う。**候補生成の Top-K では拾えない** —
/// 実測で「りんご」を含む 204 グループのうち phonetic 空間の Top-8000 に
/// 入るのは 48 件しかない。モーラ帯に限っても 123 件のうち 48 件で、
/// 包含は phonetic 空間の近さと相関しないため。
#[pyfunction]
fn containment_scan<'py>(
    py: Python<'py>,
    query_ids: PyReadonlyArray1<'py, i32>,
    phoneme_ids: PyReadonlyArray1<'py, u8>,
    phoneme_bounds: PyReadonlyArray1<'py, i32>,
    distance_ids: PyReadonlyArray1<'py, i32>,
    // 特殊モーラ (長音・促音・撥音) の距離テーブル ID。挿入を無料にする対象。
    elastic_ids: PyReadonlyArray1<'py, i32>,
    group_start: usize,
    group_end: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
    let query = query_ids.as_slice()?;
    let phoneme_ids = phoneme_ids.as_slice()?;
    let bounds = phoneme_bounds.as_slice()?;
    let distance_ids = distance_ids.as_slice()?;
    let elastic = elastic_ids.as_slice()?;

    if query.is_empty() || group_start >= group_end {
        return Ok((
            PyArray1::from_vec(py, Vec::new()),
            PyArray1::from_vec(py, Vec::new()),
        ));
    }

    let (groups, ratios) = py.detach(|| {
        // 索引 ID のまま比較できるよう、クエリを索引側の ID に写しておく。
        // 内側のループで `distance_ids` を引かずに済む (索引の音素語彙は 37 個
        // しかないので、逆引き表は真偽配列 1 本で足りる)。
        //
        // 索引の語彙に無い音素をクエリが含むなら、その音素は索引のどの語にも
        // 現れないので包含は成立しない。写せない時点で空を返す。
        // 添字は距離テーブルの ID なので、索引語彙数ではなくその最大値で取る。
        let table_size = distance_ids.iter().copied().max().unwrap_or(0) as usize + 1;
        let mut index_of: Vec<Option<u8>> = vec![None; table_size];
        let mut is_elastic = vec![false; distance_ids.len()];
        for (index, &id) in distance_ids.iter().enumerate() {
            index_of[id as usize] = Some(index as u8);
            if elastic.contains(&id) {
                is_elastic[index] = true;
            }
        }
        let mut needle: Vec<u8> = Vec::with_capacity(query.len());
        for &qp in query {
            match index_of.get(qp as usize).copied().flatten() {
                Some(id) => needle.push(id),
                None => return (Vec::new(), Vec::new()),
            }
        }

        // **グループ 1 件を 1 タスクにしてはいけない。** 1 件の判定は音素 12 個
        // 程度の比較で数十ナノ秒しかないので、rayon のスケジューリングが
        // 支配的になる。編集距離側 (`compute`) と同じ理由で塊にする。
        const CHUNK: usize = 8192;

        let total = group_end - group_start;
        (0..total.div_ceil(CHUNK))
            .into_par_iter()
            .map(|chunk| {
                let first = group_start + chunk * CHUNK;
                let last = (first + CHUNK).min(group_end);
                let mut groups: Vec<i64> = Vec::new();
                let mut ratios: Vec<f32> = Vec::new();
                for group in first..last {
                    let start = bounds[group] as usize;
                    let end = bounds[group + 1] as usize;
                    let candidate = &phoneme_ids[start..end];
                    if candidate.len() < needle.len() {
                        continue;
                    }
                    // 開始位置を総当たりする。クエリの先頭音素と一致する位置だけ
                    // 試す。特殊モーラの挿入を許すので一致は needle.len() より
                    // 長くなりうるが、開始位置の上限を
                    // candidate.len() - needle.len() で切ってよい — それより
                    // 後ろから始めても残りが足りない。
                    for offset in 0..=(candidate.len() - needle.len()) {
                        if candidate[offset] != needle[0] {
                            continue;
                        }
                        let mut position = offset;
                        let mut matched = 0usize;
                        while matched < needle.len() && position < candidate.len() {
                            let symbol = candidate[position];
                            if symbol == needle[matched] {
                                matched += 1;
                                position += 1;
                            } else if matched > 0 && is_elastic[symbol as usize] {
                                // 途中に挟まった特殊モーラは飛ばす。先頭より前の
                                // 特殊モーラは開始位置の走査が担うので、ここでは
                                // matched > 0 のときだけ許す。
                                position += 1;
                            } else {
                                break;
                            }
                        }
                        if matched == needle.len() {
                            groups.push(group as i64);
                            ratios.push(needle.len() as f32 / candidate.len() as f32);
                            break;
                        }
                    }
                }
                (groups, ratios)
            })
            // **チャンクの順に連結する。** `collect` は入力の順序を保つので
            // グループ ID が昇順で返る。`reduce` は結合順が実行ごとに変わりうる
            // ので使えない — 呼び出し側 (`search._expand_groups`) が行の昇順を
            // 前提にしており、崩れると同音異表記の畳み込みが黙って別のグループの
            // 値を配る。
            .collect::<Vec<(Vec<i64>, Vec<f32>)>>()
            .into_iter()
            .fold(
                (Vec::new(), Vec::new()),
                |mut all: (Vec<i64>, Vec<f32>), part| {
                    all.0.extend(part.0);
                    all.1.extend(part.1);
                    all
                },
            )
    });

    Ok((
        PyArray1::from_vec(py, groups),
        PyArray1::from_vec(py, ratios),
    ))
}

/// 候補方向を rayon で並列化して距離を出す。
///
/// 候補ごとの DP は互いに独立なので分割の仕方を選ぶ必要がない。DP の作業配列だけ
/// はチャンクごとに 1 度確保して使い回す — 候補ごとに `vec!` すると 53 万回の
/// 割り当てが計算より重くなる。
fn compute(
    flat: &[i32],
    offsets: &[usize],
    query: &[i32],
    substitution: &[f32],
    indel: &[f32],
    phoneme_count: usize,
    width: usize,
) -> Vec<f32> {
    let count = offsets.len() - 1;
    let mut distances = vec![0.0f32; count];

    // チャンクの粒度。候補 1 件の DP は数百ナノ秒なので、細かく割ると
    // rayon のスケジューリングが支配的になる。
    const CHUNK: usize = 2048;

    distances
        .par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(chunk_index, out)| {
            let mut previous = vec![0.0f32; width + 1];
            let mut current = vec![0.0f32; width + 1];
            let base = chunk_index * CHUNK;
            for (local, slot) in out.iter_mut().enumerate() {
                let row = base + local;
                let candidate = &flat[offsets[row]..offsets[row + 1]];
                *slot = edit_distance_one(
                    query,
                    candidate,
                    substitution,
                    indel,
                    phoneme_count,
                    &mut previous,
                    &mut current,
                );
            }
        });

    distances
}

#[pymodule]
fn jpr_distance(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(edit_distance_batch, module)?)?;
    module.add_function(wrap_pyfunction!(edit_distance_csr, module)?)?;
    module.add_function(wrap_pyfunction!(containment_scan, module)?)?;
    module.add_function(wrap_pyfunction!(gemv::top_candidates, module)?)?;
    module.add_function(wrap_pyfunction!(gemv::dot_all, module)?)?;
    Ok(())
}

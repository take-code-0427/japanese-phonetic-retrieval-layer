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
        previous[..=m].copy_from_slice(&current[..=m]);
    }

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
    module.add_function(wrap_pyfunction!(gemv::top_candidates, module)?)?;
    module.add_function(wrap_pyfunction!(gemv::dot_all, module)?)?;
    Ok(())
}

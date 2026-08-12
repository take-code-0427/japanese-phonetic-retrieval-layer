//! 量子化ベクトルの内積と Top-K 抽出。
//!
//! 候補生成は「索引全体との内積」なので帯域律速になる。実測 (202 万語 x
//! 72 次元、8 コア、**プロセスを分けて** 20 回の最小値):
//!
//! | | 内積 | Top-K |
//! |---|---|---|
//! | NumPy f32 (BLAS + argpartition) | 15.4〜26.0ms | 9.0〜15.5ms |
//! | Rust int8 | 16.8〜28.8ms | 2.6〜10.0ms |
//!
//! **内積は BLAS と互角で、int8 にしても速くならない。** 読むバイト数は 1/4 に
//! なるが、int8 の積和は BLAS の SIMD された f32 積和より 1 要素あたりの
//! スループットが出ないので相殺される。量子化の主目的は**サイズ**
//! (索引 1.64GB -> 508MB) であって速度ではない。速度で得しているのは Top-K のみ。
//!
//! それでも内積をここに置いているのは、**索引が int8 だから**。NumPy には
//! int8 の GEMV 経路が無く、`astype(int32)` を経由すると 202 万行ぶんの
//! 中間配列 (582MB) を実体化してしまう。f32 に戻して BLAS に渡す案も同じ理由で
//! 採れない。
//!
//! **測定は必ずプロセスを分ける。** 同じプロセスで f32 と int8 を交互に
//! 測ると、582MB の f32 行列が毎回 L3 を流すので次に走る int8 (145MB) だけが
//! 有利になり、int8 が 2.6 倍速いという誤った結論が出た。
//!
//! **内側のループは素朴に書く。** 4 アキュムレータへの手動アンロールを
//! 試したら 3 倍遅くなった。LLVM は素朴な形なら自動ベクトル化できるが、
//! 手で展開すると崩れる。`target-cpu` の指定にも頼らない。

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// 行の分割単位。細かすぎると rayon のスケジューリングが支配的になる。
const GEMV_CHUNK: usize = 4096;

/// Top-K をスレッド局所に求めるときの分割単位。
const TOPK_CHUNK: usize = 262144;

/// int8 行列と int8 クエリの内積。結果は i32 の生の積和で返す。
///
/// スケールの復元 (`scale_row * scale_query` を掛ける) は呼び出し側で行う。
/// 順位付けにはスケールが要らないので、Top-K を取るまでは整数のまま扱える。
fn gemv_i8_into(matrix: &[i8], query: &[i8], dim: usize, out: &mut [i32]) {
    out.par_chunks_mut(GEMV_CHUNK)
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let base = chunk_index * GEMV_CHUNK;
            // 末尾のチャンクは短くなるが `par_chunks_mut` が長さを合わせるので、
            // ここで行数を再確認する必要はない。
            for (local, slot) in chunk.iter_mut().enumerate() {
                let row = base + local;
                let values = &matrix[row * dim..row * dim + dim];
                // 素朴な積和のまま置く (モジュールの docstring 参照)。
                let mut accumulator: i32 = 0;
                for k in 0..dim {
                    accumulator += (values[k] as i32) * (query[k] as i32);
                }
                *slot = accumulator;
            }
        });
}

/// 上位 `k` 件の行番号をスコア降順で返す。
///
/// 全体を 1 度に `select_nth_unstable` へ渡すと単一スレッドになるので、
/// チャンクごとに局所 Top-K を取ってからマージする。実測で NumPy の
/// `argpartition` (9.0〜15.5ms) に対し 2.6〜10.0ms。**ここは実際に速い** —
/// 内積と違って比較と入れ替えが主で、帯域ではなく並列度で決まる。
///
/// 行番号は u32 で持つ (スコアとの組を 8 バイトに収めて移動を軽くするため)。
/// 索引は full でも 202 万行なので 42 億の上限には遠い。
fn topk_i32(scores: &[i32], k: usize) -> Vec<i64> {
    let k = k.min(scores.len());
    if k == 0 {
        return Vec::new();
    }

    // **スコアだけで比べてはいけない。** 同点が k 件を跨ぐとき、どれが残るかが
    // 決まらなくなる。チャンク内の選抜は入力順を保証しない
    // (`select_nth_unstable`) ので、最後に並べ替えても「生き残った中での順序」
    // が揃うだけで、**どれが生き残ったか**は揺れたまま。実測で全件同点の
    // 5000 行から上位 10 件を取ると 0,1 を飛ばして 2852 が入った。
    //
    // 索引の代表選び (`search._representative_rank`) は候補の到着順に依存する
    // ので、ここが揺れると同じクエリが違う表記を返す。スコアが同じなら
    // 行番号の小さい方を上に置き、全順序にする。
    let order = |a: &(i32, u32), b: &(i32, u32)| b.0.cmp(&a.0).then(a.1.cmp(&b.1));

    let mut merged: Vec<(i32, u32)> = scores
        .par_chunks(TOPK_CHUNK)
        .enumerate()
        .map(|(chunk_index, chunk)| {
            let base = (chunk_index * TOPK_CHUNK) as u32;
            let mut local: Vec<(i32, u32)> = chunk
                .iter()
                .enumerate()
                .map(|(offset, &score)| (score, base + offset as u32))
                .collect();
            let take = k.min(local.len());
            local.select_nth_unstable_by(take - 1, order);
            local.truncate(take);
            local
        })
        .reduce(Vec::new, |mut acc, mut part| {
            acc.append(&mut part);
            acc
        });

    let take = k.min(merged.len());
    merged.select_nth_unstable_by(take - 1, order);
    merged.truncate(take);
    merged.sort_unstable_by(order);
    merged.into_iter().map(|(_, row)| row as i64).collect()
}

/// 索引全体と内積を取り、上位 `k` 件の行と復元済みスコアを返す。
///
/// `scale` は量子化のスケール (行列側 x クエリ側をあらかじめ掛けたもの)。
#[pyfunction]
pub fn top_candidates<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, i8>,
    query: PyReadonlyArray1<'py, i8>,
    k: usize,
    scale: f32,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f32>>)> {
    let matrix_view = matrix.as_array();
    let dim = matrix_view.shape()[1];
    let rows = matrix_view.shape()[0];
    let matrix_slice = matrix
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("行列が C 連続ではありません"))?;
    let query_slice = query.as_slice()?;

    if query_slice.len() != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "クエリの次元が行列と一致しません: {} != {}",
            query_slice.len(),
            dim
        )));
    }

    // Top-K が行番号を u32 で持つので、それを超える索引は扱えない
    // (`topk_i32` を参照)。黙って回るより落ちるほうがいい。
    if rows > u32::MAX as usize {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "索引の行数が Top-K の上限を超えています ({} > {})",
            rows,
            u32::MAX
        )));
    }

    let (top, scores) = py.detach(|| {
        let mut raw = vec![0i32; rows];
        gemv_i8_into(matrix_slice, query_slice, dim, &mut raw);
        let top = topk_i32(&raw, k);
        let scores: Vec<f32> = top.iter().map(|&row| raw[row as usize] as f32 * scale).collect();
        (top, scores)
    });

    Ok((PyArray1::from_vec(py, top), PyArray1::from_vec(py, scores)))
}

/// 行列の全行と内積を取り、復元済みスコアをそのまま返す (Top-K を取らない)。
///
/// モーラ範囲の全走査 (`search._scan_candidates`) と rerank 用空間の
/// スコア (`search._space_scores`) が使う。呼び出し側が母集団の連続区間を
/// スライスして渡すので、ここは受け取った行列を素直に全部見る。
#[pyfunction]
pub fn dot_all<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray2<'py, i8>,
    query: PyReadonlyArray1<'py, i8>,
    scale: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let matrix_view = matrix.as_array();
    let dim = matrix_view.shape()[1];
    let rows = matrix_view.shape()[0];
    let matrix_slice = matrix
        .as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("行列が C 連続ではありません"))?;
    let query_slice = query.as_slice()?;

    if query_slice.len() != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "クエリの次元が行列と一致しません: {} != {}",
            query_slice.len(),
            dim
        )));
    }

    let scores = py.detach(|| {
        let mut raw = vec![0i32; rows];
        gemv_i8_into(matrix_slice, query_slice, dim, &mut raw);
        raw.into_iter().map(|value| value as f32 * scale).collect::<Vec<f32>>()
    });

    Ok(PyArray1::from_vec(py, scores))
}

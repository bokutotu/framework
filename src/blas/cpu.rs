/// warning: このファイルでwrapされるblasはshapeに関する一切のcheckを行いません。
use std::fmt::Debug;

use crate::tensor::{CpuViewMutTensor, CpuViewTensor};
use crate::wrapper::cpu_blas::*;

use num_traits::Num;

/// ベクトルの各成分の絶対値を合計した値を計算します。
/// 結果は戻り値として返ってきます。
/// 複素数のベクトルを与えた場合でも、絶対値の合計ですので、実数が返ってくることに注意してください。
/// 例えば、doublecomplexのベクトルを与えた場合は、doubleで受け取る、などです。
pub fn asum<E: CpuAsum + Copy>(a: CpuViewTensor<E>, inc: i32) -> <E as CpuAsum>::Out {
    let num_elm = a.num_elm as i32;
    let ptr = unsafe { std::slice::from_raw_parts(a.as_ptr(), a.num_elm) };
    E::cpu_asum(num_elm, ptr, inc)
}

/// ベクトル同士の加算を行います。
/// 行列も大きさが非常に長いベクトルだと思えば使えます。
/// 与えたベクトルYの内容は破壊され、計算結果が書きこまれます。
/// Y := alpha * X + Y
pub fn axpy<E: CpuAxpy + Copy + Num + Debug>(
    alpha: E,
    incx: i32,
    incy: i32,
    a: CpuViewTensor<E>,
    b: CpuViewMutTensor<E>,
) {
    let a_slice = a.to_slice();
    let b_slice = b.to_slice_mut();
    E::cpu_axpy(
        a.num_elm.try_into().unwrap(),
        alpha,
        a_slice,
        incx,
        b_slice,
        incy,
    )
}

#[test]
fn asum_test_f32() {
    use crate::shape::Shape;
    use crate::tensor::CpuTensor;
    let mut v: Vec<f32> = Vec::new();
    for i in 0..1_000 {
        v.push(i as f32);
    }
    let a = CpuTensor::from_vec(v.clone(), Shape::new(vec![1_000_000_000]));
    let res = asum(a.to_view(), 1);
    let ans = v.iter().fold(0., |x, y| x + y);
    assert_eq!(ans, res);
}

#[test]
fn axpy_test_f32() {
    use crate::shape::Shape;
    use crate::tensor::CpuTensor;
    let mut a: Vec<f32> = Vec::new();
    let mut b: Vec<f32> = Vec::new();
    let mut c: Vec<f32> = Vec::new();
    for i in 0..1_000 {
        a.push(i as f32);
        b.push(i as f32);
        c.push((i * 2) as f32);
    }
    let a = CpuTensor::from_vec(a, Shape::new(vec![10, 100]));
    let mut b = CpuTensor::from_vec(b, Shape::new(vec![10, 100]));
    axpy(1., 1, 1, a.to_view(), b.to_view_mut());
    let res = b.to_vec();
    assert_eq!(res, c);
}

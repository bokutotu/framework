use crate::wrapper::cpu_blas::*;
use crate::tensor::CpuViewTensor;

/// ベクトルの各成分の絶対値を合計した値を計算します。
/// 結果は戻り値として返ってきます。
/// 複素数のベクトルを与えた場合でも、絶対値の合計ですので、実数が返ってくることに注意してください。
/// 例えば、doublecomplexのベクトルを与えた場合は、doubleで受け取る、などです。
pub fn asum<E: CpuAsum + Copy>(a: CpuViewTensor<E>, inc: i32) -> <E as CpuAsum>::Out {
    let num_elm = a.num_elm as i32;
    let ptr = unsafe {std::slice::from_raw_parts(a.as_ptr(), a.num_elm) };
    E::cpu_asum(num_elm, ptr, inc)
}

/// ベクトル同士の加算を行います。
/// 行列も大きさが非常に長いベクトルだと思えば使えます。
/// 与えたベクトルYの内容は破壊され、計算結果が書きこまれます。
/// Y := alpha * X + Y
// pub fn axpy<E: CpuAxpy + Copy>(alpha: E, a: CpuViewTensor<E>, b: CpuViewMutTensor<E>) {
//     if a.shape() == b.shape() && a.stride() == b.stride() {
//     }
// }

#[test]
fn asum_test_f32() {
    use crate::tensor::CpuTensor;
    use crate::shape::Shape;
    let mut v: Vec<f32> = Vec::new();
    for i in 0..1_000 {
        v.push(i as f32);
    }
    let a = CpuTensor::from_vec(v.clone(), Shape::new(vec![1_000_000_000]));
    let res = asum(a.to_view(), 1);
    let ans = v.iter().fold(0., |x, y| x+y);
    assert_eq!(ans, res);
}

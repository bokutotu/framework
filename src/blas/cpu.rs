use std::convert::TryInto;
/// warning: このファイルでwrapされるblasはshapeに関する一切のcheckを行いません。
use std::fmt::Debug;

use crate::tensor::{CpuViewMutTensor, CpuViewTensor};
use crate::wrapper::cpu_blas::*;

use num_traits::Num;

/// ベクトルの各成分の絶対値を合計した値を計算します。
/// 結果は戻り値として返ってきます。
/// 複素数のベクトルを与えた場合でも、絶対値の合計ですので、実数が返ってくることに注意してください。
/// 例えば、doublecomplexのベクトルを与えた場合は、doubleで受け取る、などです。
pub fn asum<E: CpuAsum>(a: CpuViewTensor<E>, inc: i32) -> <E as CpuAsum>::Out {
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

/// ベクトルをXからYにコピーします。行列も大きさが非常に長いベクトルだと思えば使えます。
/// 与えたベクトルYの内容は破壊され、内容がXになります。
/// 単なるコピー演算なので、自力で書いてもそこまで違わないのではないか、
/// と思われるかもしれませんが、若干何故か速いことがあるようです。
/// アラインメントの関係などのあたりでうまく最適化してるのだろうなと想像しています。
pub fn copy<E: CpuCopy>(
    incx: i32, incy: i32, x: CpuViewTensor<E>, y: CpuViewMutTensor<E>) 
{
    let x_slice = x.to_slice();
    let y_slice = y.to_slice_mut();
    E::cpu_copy(x.num_elm.try_into().unwrap(), x_slice, incx, y_slice, incy)
}

/// ベクトル同士の内積の値を計算します。計算結果は戻り値として返ってきます。
/// ?dot_は実数の物にしか提供されていないことに注意してください。
/// 複素数ベクトルには専用のルーチンが用意されています。
/// （若干マニアックですが）行列の内積(tr(XY^t))も大きさが非常に長いベクトルだと思えば使えます。 
pub fn dot<E: CpuDot<Out=E>>(x: CpuViewTensor<E>, y: CpuViewTensor<E>, incx: i32, incy: i32) -> E {
    E::cpu_dot(x.num_elm.try_into().unwrap(), x.to_slice(), incx, y.to_slice(), incy)
}

pub fn sdot(x: CpuViewTensor<f32>, y: CpuViewTensor<f32>, incx: i32, incy: i32) -> f64 {
    f32::cpu_sdot(x.num_elm.try_into().unwrap(), x.to_slice(), incx, y.to_slice(), incy)
}

/// ベクトルのユークリッドノルム、つまり普通のノルムを計算します。
/// 結果は戻り値として返ってきます。
/// 複素数のベクトルを与えた場合でも、実数が返ってくることに注意してください。
/// 例えば、doublecomplexのベクトルを与えた場合は、doubleで受け取る、などです。
/// 行列をベクトルとして渡した場合、フロベニウスノルムが計算できます。 
pub fn nrm2<E: CpuNrm2>(x: CpuViewTensor<E>, incx: i32) -> <E as CpuNrm2>::Out {
    E::cpu_nrm2(x.num_elm.try_into().unwrap(), x.to_slice(), 1)
}

/// 長さが同じベクトルX,Yを与えます。
/// これは(X(i),Y(i))で点が与えられているものと解釈されます。
/// ここに、実数c,sを与えます。c=cos A, s=sin Aとすると、
/// 角度Aでこれらの点を回転させた結果がX,Yに上書きされます。
/// つまり、回転行列を掛けた結果が返ってくるということになります。
///
/// c,sの値は必ず実数になりますので注意してください。
///
/// X(i) := c * X(i) + s * Y(i)
/// Y(i) :=-s * X(i) + c * Y(i)
pub fn rot<E: CpuRot>
    (x: CpuViewMutTensor<E>, y: CpuViewMutTensor<E>, incx: i32, incy: i32, c: E, s: E) 
{
    E::cpu_rot(
        x.num_elm.try_into().unwrap(), x.to_slice_mut(), incx, y.to_slice_mut(), incy, c, s);
}

/// 与えたベクトルをスカラ倍します。
/// 複素数のベクトルの場合は、実数倍をする専用のルーチンが用意されています
pub fn scal<E: CpuScal>(alpha: E, x: CpuViewMutTensor<E>, incx: i32) {
    E::cpu_scal(x.num_elm.try_into().unwrap(), alpha, x.to_slice_mut(), incx)
}

/// ベクトルの中で最小の絶対値を持つ要素の添字を計算します。
/// 結果は戻り値として返ってきます。
/// 添字が返ってくるので、当然整数を受け取ることになります。
/// ただし、この添字は1から始まるので注意してください。0が返ってきたときは、nが不正な場合です。
pub fn iamax<E: CpuIamax<Out=i32>>(x: CpuViewTensor<E>, incx: i32) -> i32 {
    E::cpu_iamax(x.num_elm.try_into().unwrap(), x.to_slice(), incx) 
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

#[test]
fn copy_test_f32() {
    use crate::shape::Shape;
    use crate::tensor::CpuTensor;
    let a = vec![0., 1., 2.];
    let b = vec![0., 0., 0.];
    let a = CpuTensor::from_vec(a, Shape::new(vec![3]));
    let mut b = CpuTensor::from_vec(b, Shape::new(vec![3]));
    copy(1, 1, a.to_view(), b.to_view_mut());
    let b = b.to_vec();
    assert_eq!(vec![0., 1., 2.], b);
}

#[test]
fn dot_test_f32() {
    use crate::shape::Shape;
    use crate::tensor::CpuTensor;
    let a = vec![1., 1., 2.];
    let b = vec![2., 3., 4.];
    let a = CpuTensor::from_vec(a, Shape::new(vec![3]));
    let b = CpuTensor::from_vec(b, Shape::new(vec![3]));
    let res = dot(a.to_view(), b.to_view(), 1, 1);
    assert_eq!(res, 13.)
}

#[test]
fn sdot_test_f32() {
    use crate::shape::Shape;
    use crate::tensor::CpuTensor;
    let a = vec![1., 1., 2.];
    let b = vec![2., 3., 4.];
    let a = CpuTensor::from_vec(a, Shape::new(vec![3]));
    let b = CpuTensor::from_vec(b, Shape::new(vec![3]));
    let res = sdot(a.to_view(), b.to_view(), 1, 1);
    assert_eq!(res, 13.)
}

#[test]
fn nrm2_test_f32() {
    use crate::shape::Shape;
    use crate::tensor::CpuTensor;
    let a = vec![3., -4.];
    let a = CpuTensor::from_vec(a, Shape::new(vec![3]));
    let res = nrm2(a.to_view(), 1);
    assert_eq!(res, 5.);
}

#[test]
fn rot_test_f32() {
    use crate::shape::Shape;
    use crate::tensor::CpuTensor;
    let a = vec![1., 0.];
    let b = vec![0., 1.,];
    let mut a = CpuTensor::from_vec(a, Shape::new(vec![2]));
    let mut b = CpuTensor::from_vec(b, Shape::new(vec![2]));
    rot(a.to_view_mut(), b.to_view_mut(), 1, 1, 1./2f32.powf(0.5), 1./2f32.powf(0.5));
    let a = a.to_vec();
    let b = b.to_vec();
    assert_eq!(a, vec![1./2f32.powf(0.5), 1./2f32.powf(0.5)]);
    assert_eq!(b, vec![-1./2f32.powf(0.5), 1./2f32.powf(0.5)]);
}

#[test]
fn scal_test_f32() {
    use crate::shape::Shape;
    use crate::tensor::CpuTensor;
    let a = vec![1., 0.];
    let mut a = CpuTensor::from_vec(a, Shape::new(vec![2]));
    scal(2., a.to_view_mut(), 1);
    let a = a.to_vec();
    assert_eq!(a, vec![2., 0.]);
}

#[test]
fn amax_test_f32() {
    use crate::shape::Shape;
    use crate::tensor::CpuTensor;
    let a = vec![0., 1., 2., 3., 4., 5. ];
    let a = CpuTensor::from_vec(a, Shape::new(vec![6]));
    let idx = iamax(a.to_view(), 1);
    assert_eq!(idx, 5);
}

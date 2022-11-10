use cblas::*;
use crate::define_impl;

define_impl! {
    CpuGemm,
    cpu_gemm,
    ((sgemm, f32), (dgemm, f64)),
    (
        layout: Layout,
        transa: Transpose,
        transb: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        b: &[Self],
        ldb: i32,
        beta: Self,
        c: &mut [Self],
        ldc: i32
    )
}

define_impl!(
    CpuAsum,
    cpu_asum,
    ((sasum, f32, f32), (dasum, f64, f64)),
    (n: i32, x: &[Self], incx: i32)
);

define_impl!(
    CpuAxpy,
    cpu_axpy,
    ((saxpy, f32), (daxpy, f64)),
    (
        n: i32,
        alpha: Self,
        x: &[Self],
        incx: i32,
        y: &mut [Self],
        incy: i32
    )
);

define_impl!(
    CpuCopy,
    cpu_copy,
    ((scopy, f32), (dcopy, f64)),
    (n: i32, x: &[Self], incx: i32, y: &mut [Self], incy: i32)
);

define_impl!(
    CpyDot,
    cpu_dot,
    ((sdot, f32, f32), (ddot, f64, f64)),
    (n: i32, x: &[Self], incx: i32, y: &[Self], incy: i32)
);

define_impl!(
    CpySdot,
    cpu_sdot,
    ((dsdot, f32, f64)),
    (n: i32, x: &[Self], incx: i32, y: &[Self], incy: i32)
);

define_impl!(
    CpuNrm2,
    cpu_nrm2,
    ((snrm2, f32, f32), (dnrm2, f64, f64)),
    (n: i32, x: &[Self], incx: i32)
);

define_impl!(
    CpuRot,
    cpu_rot,
    ((srot, f32), (drot, f64)),
    (
        n: i32,
        x: &mut [Self],
        incx: i32,
        y: &mut [Self],
        incy: i32,
        c: Self,
        s: Self
    )
);

define_impl!(
    CpuRotg,
    cpu_rotg,
    ((srotg, f32), (drotg, f64)),
    (n: &mut Self, b: &mut Self, c: &mut Self, s: &mut [Self])
);

#[test]
fn gemm_test() {
    let (m, n, k) = (2, 4, 3);
    let a = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    let b = vec![
        1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0,
    ];
    let mut c = vec![2.0, 7.0, 6.0, 2.0, 0.0, 7.0, 4.0, 2.0];

    f64::cpu_gemm(
        Layout::ColumnMajor,
        Transpose::None,
        Transpose::None,
        m,
        n,
        k,
        1.0,
        &a,
        m,
        &b,
        k,
        1.0,
        &mut c,
        m,
    );
    assert!(c == vec![40.0, 90.0, 50.0, 100.0, 50.0, 120.0, 60.0, 130.0,]);
}

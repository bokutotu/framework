use std::fmt::Debug;

use num_traits::Num;

use crate::define_impl;
use cblas::*;

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
    CpuDot,
    cpu_dot,
    ((sdot, f32, f32), (ddot, f64, f64)),
    (n: i32, x: &[Self], incx: i32, y: &[Self], incy: i32)
);

define_impl!(
    CpuSdot,
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

// define_impl!(
//     CpuRotg,
//     cpu_rotg,
//     ((srotg, f32), (drotg, f64)),
//     (n: &mut Self, b: &mut Self, c: &mut Self, s: &mut [Self])
// );
//
// define_impl!(
//     CpuRotm,
//     cpu_rotm,
//     ((srotm, f32), (drotm, f64)),
//     (
//         n: i32,
//         x: &mut [Self],
//         incx: i32,
//         y: &mut [Self],
//         incy: i32,
//         p: &[Self]
//     )
// );

// define_impl!(
//     CpuSrotmg,
//     cpu_srotmg,
//     ((srotmg, f32), (drotmg, f64)),
//     (
//         d1: &mut [Self],
//         d2: &mut [Self],
//         b1: &mut [Self],
//         b2: Self,
//         p: &mut [Self]
//     )
// );

define_impl!(
    CpuScal,
    cpu_scal,
    ((sscal, f32), (dscal, f64)),
    (n: i32, alpha: Self, x: &mut [Self], incx: i32)
);

// define_impl!(
//     CpuSwap,
//     cpu_swap,
//     ((sswap, f32), (dswap, f64)),
//     (n: i32, x: &mut [Self], incx: i32, y: &mut [Self], incy: i32)
// );

define_impl!(
    CpuIamax,
    cpu_iamax,
    ((isamax, f32, i32), (idamax, f64, i32)),
    (n: i32, x: &[Self], incx: i32)
);

define_impl!(
    CpuGbmv,
    cpu_gbmv,
    ((sgbmv, f32), (dgbmv, f64)),
    (
        layout: Layout,
        transa: Transpose,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        x: &[Self],
        incx: i32,
        beta: Self,
        y: &mut [Self],
        incy: i32
    )
);

define_impl!(
    CpuGemv,
    cpu_gemv,
    ((sgbmv, f32), (dgbmv, f64)),
    (
        layout: Layout,
        transa: Transpose,
        m: i32,
        n: i32,
        kl: i32,
        ku: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        x: &[Self],
        incx: i32,
        beta: Self,
        y: &mut [Self],
        incy: i32
    )
);

define_impl!(
    CpuGer,
    cpu_ger,
    ((sger, f32), (dger, f64)),
    (
        layout: Layout,
        m: i32,
        n: i32,
        alpha: Self,
        x: &[Self],
        incx: i32,
        y: &[Self],
        incy: i32,
        a: &mut [Self],
        lda: i32
    )
);

define_impl!(
    CpuGerc,
    cpu_gerc,
    ((sger, f32), (dger, f64)),
    (
        layout: Layout,
        m: i32,
        n: i32,
        alpha: Self,
        x: &[Self],
        incx: i32,
        y: &[Self],
        incy: i32,
        a: &mut [Self],
        lda: i32
    )
);

define_impl!(
    CpuSbmv,
    cpu_sbmv,
    ((ssbmv, f32), (dsbmv, f64)),
    (
        layout: Layout,
        uplo: Part,
        n: i32,
        k: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        x: &[Self],
        incx: i32,
        beta: Self,
        y: &mut [Self],
        incy: i32
    )
);

define_impl!(
    CpuSpmv,
    cpu_spmv,
    ((sspmv, f32), (dspmv, f64)),
    (
        layout: Layout,
        uplo: Part,
        n: i32,
        alpha: Self,
        ap: &[Self],
        x: &[Self],
        incx: i32,
        beta: Self,
        y: &mut [Self],
        incy: i32
    )
);

define_impl!(
    CpuSpr,
    cpu_spr,
    ((sspr, f32), (dspr, f64)),
    (
        layout: Layout,
        uplo: Part,
        n: i32,
        alpha: Self,
        x: &[Self],
        incx: i32,
        ap: &mut [Self]
    )
);

define_impl!(
    CpuSpr2,
    cpu_spr2,
    ((sspr2, f32), (dspr2, f64)),
    (
        layout: Layout,
        uplo: Part,
        n: i32,
        alpha: Self,
        x: &[Self],
        incx: i32,
        y: &[Self],
        incy: i32,
        a: &mut [Self]
    )
);

define_impl!(
    CpuSymv,
    cpu_symv,
    ((ssymv, f32), (dsymv, f64)),
    (
        layout: Layout,
        uplo: Part,
        n: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        x: &[Self],
        incx: i32,
        beta: Self,
        y: &mut [Self],
        incy: i32
    )
);

define_impl!(
    CpuSyr,
    cpu_syr,
    ((ssyr, f32), (dsyr, f64)),
    (
        layout: Layout,
        uplo: Part,
        n: i32,
        alpha: Self,
        x: &[Self],
        incx: i32,
        a: &mut [Self],
        lda: i32
    )
);

define_impl!(
    CpuSyr2,
    cpu_syr2,
    ((ssyr2, f32), (dsyr2, f64)),
    (
        layout: Layout,
        uplo: Part,
        n: i32,
        alpha: Self,
        x: &[Self],
        incx: i32,
        y: &[Self],
        incy: i32,
        a: &mut [Self],
        lda: i32
    )
);

define_impl!(
    CpuTbmv,
    cpu_tbmv,
    ((stbmv, f32), (dtbmv, f64)),
    (
        layout: Layout,
        uplo: Part,
        transa: Transpose,
        diag: Diagonal,
        n: i32,
        k: i32,
        a: &[Self],
        lda: i32,
        x: &mut [Self],
        incx: i32
    )
);

define_impl!(
    CpuTbsv,
    cpu_tbsv,
    ((stbsv, f32), (dtbsv, f64)),
    (
        layout: Layout,
        uplo: Part,
        transa: Transpose,
        diag: Diagonal,
        n: i32,
        k: i32,
        a: &[Self],
        lda: i32,
        x: &mut [Self],
        incx: i32
    )
);

define_impl!(
    CpuTpmv,
    cpu_tpmv,
    ((stpmv, f32), (dtpmv, f64)),
    (
        layout: Layout,
        uplo: Part,
        transa: Transpose,
        diag: Diagonal,
        n: i32,
        ap: &[Self],
        x: &mut [Self],
        incx: i32
    )
);

define_impl!(
    CpuTpsv,
    cpu_tpsv,
    ((stpsv, f32), (dtpsv, f64)),
    (
        layout: Layout,
        uplo: Part,
        transa: Transpose,
        diag: Diagonal,
        n: i32,
        ap: &[Self],
        x: &mut [Self],
        incx: i32
    )
);

define_impl!(
    CpuTrmv,
    cpu_trmv,
    ((strmv, f32), (dtrmv, f64)),
    (
        layout: Layout,
        uplo: Part,
        transa: Transpose,
        diag: Diagonal,
        n: i32,
        a: &[Self],
        lda: i32,
        x: &mut [Self],
        incx: i32
    )
);

define_impl!(
    CpuTrsv,
    cpu_trsv,
    ((strsv, f32), (dtrsv, f64)),
    (
        layout: Layout,
        uplo: Part,
        transa: Transpose,
        diag: Diagonal,
        n: i32,
        a: &[Self],
        lda: i32,
        x: &mut [Self],
        incx: i32
    )
);

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
    CpuSymm,
    cpu_symm,
    ((ssymm, f32), (dsymm, f64)),
    (
        layout: Layout,
        side: Side,
        uplo: Part,
        m: i32,
        n: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        b: &[Self],
        ldb: i32,
        beta: Self,
        c: &mut [Self],
        ldc: i32
    )
);

define_impl!(
    CpuSyrk,
    cpu_syrk,
    ((ssyrk, f32), (dsyrk, f64)),
    (
        layout: Layout,
        uplo: Part,
        trans: Transpose,
        n: i32,
        k: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        beta: Self,
        c: &mut [Self],
        ldc: i32
    )
);

define_impl!(
    CpuSyr2k,
    cpu_syrk,
    ((ssyr2k, f32), (dsyr2k, f64)),
    (
        layout: Layout,
        uplo: Part,
        trans: Transpose,
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
);

define_impl!(
    CpuTrmm,
    cpu_trmm,
    ((strmm, f32), (dtrmm, f64)),
    (
        layout: Layout,
        side: Side,
        uplo: Part,
        transa: Transpose,
        diag: Diagonal,
        m: i32,
        n: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32
    )
);

define_impl!(
    CpuTrsm,
    cpu_trsm,
    ((strsm, f32), (dtrsm, f64)),
    (
        layout: Layout,
        side: Side,
        uplo: Part,
        transa: Transpose,
        diag: Diagonal,
        m: i32,
        n: i32,
        alpha: Self,
        a: &[Self],
        lda: i32,
        b: &mut [Self],
        ldb: i32
    )
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

use cblas::*;

macro_rules! define_impl_blas {
    ($trait:ident, $fn_name:ident, ($(($call_fn:ident, $impl_ty:ty $(, $return_ty:ty)?)),*), ($($arg:ident : $ty: ty),*)) => {
        define_impl_blas!(@define $trait, $fn_name, ($(($call_fn, $impl_ty $(, $return_ty)?)),*), ($($arg : $ty),*));
    };

    (@define $trait:ident, $fn_name:ident, ($(($call_fn:ident, $impl_ty:ty)),*), ($($arg:ident : $ty:ty),*)) => {
        trait $trait: Sized {
            fn $fn_name($($arg:$ty),*);
        }
        define_impl_blas!(@r $trait, $fn_name, {$(($call_fn, $impl_ty ))*}, ($($arg: $ty),*));
    };

    (@define $trait:ident, $fn_name:ident, ($(($call_fn:ident, $impl_ty:ty, $return_ty:ty )),*), ($($arg:ident : $ty:ty),*)) => {
        trait $trait: Sized {
            type Out : Sized;
            fn $fn_name($($arg:$ty),*)  -> Self::Out ;
        }
        define_impl_blas!(@r $trait, $fn_name, {$(($call_fn, $impl_ty, $return_ty ))*}, ($($arg: $ty),*));
    };

    (@r $trait:ident, $fn_name:ident, {($call_fn:ident, $impl_ty:ty ) $(($call_fn_:ident, $impl_ty_:ty ))*}, ($($arg:ident : $ty:ty),*)) => {
        impl $trait for $impl_ty {
            fn $fn_name( $($arg: $ty),* ) {
                unsafe {
                    $call_fn( $( $arg ),* )
                }
            }
        }
        define_impl_blas!(@r $trait, $fn_name, {$(($call_fn_, $impl_ty_))*}, ($($arg: $ty),*));
    };

    (@r $trait:ident, $fn_name:ident, {($call_fn:ident, $impl_ty:ty, $return_ty:ty ) $(($call_fn_:ident, $impl_ty_:ty, $return_ty_:ty ))*}, ($($arg:ident : $ty:ty),*)) => {
        impl $trait for $impl_ty {
            type Out = $return_ty;
            fn $fn_name( $($arg: $ty),* ) -> Self::Out{
                unsafe {
                    $call_fn( $( $arg ),* )
                }
            }
        }
        define_impl_blas!(@r $trait, $fn_name, {$(($call_fn_, $impl_ty_, $return_ty_))*}, ($($arg: $ty),*));
    };

    (@r $trait:ident, $fn_name:ident, {}, ($($arg:ident : $ty: ty),*)) => {}
}

#[test]
fn macro_test() {
    unsafe fn float_func(a: f32, b: f32, c: &mut f32) {
        *c = a + b;
    }
    unsafe fn double_func(a: f64, b: f64, c: &mut f64) {
        *c = a + b;
    }
    define_impl_blas!(
        Func,
        func,
        ((float_func, f32), (double_func, f64)),
        (a: Self, b: Self, c: &mut Self)
    );
    let mut c: f32 = 0.;
    let a: f32 = 10.;
    let b: f32 = 2.;
    f32::func(a, b, &mut c);
    assert_eq!(c, 12.);
    let mut c: f64 = 0.;
    let a: f64 = 10.;
    let b: f64 = 2.;
    f64::func(a, b, &mut c);
    assert_eq!(c, 12.);
}

#[test]
fn macro_test_return() {
    unsafe fn float_func_r(a: f32, b: f32) -> f32 {
        a + b
    }
    unsafe fn double_func_r(a: f64, b: f64) -> f64 {
        a + b
    }
    define_impl_blas!(
        FuncR,
        func_r,
        ((float_func_r, f32, f32), (double_func_r, f64, f64)),
        (a: Self, b: Self)
    );
    assert_eq!(f32::func_r(0., 1.), 1.);
    assert_eq!(f64::func_r(0., 1.), 1.);
}

#[test]
fn macro_test_return_diff_type() {
    unsafe fn float_func_r_n(_: f32, _: f32) -> f64 {
        1.
    }
    unsafe fn double_func_r_n(_: f64, _: f64) -> f32 {
        1.
    }
    define_impl_blas!(
        FuncRN,
        func_r_n,
        ((float_func_r_n, f32, f64), (double_func_r_n, f64, f32)),
        (a: Self, b: Self)
    );
    assert_eq!(f32::func_r_n(0., 1.), 1.);
    assert_eq!(f64::func_r_n(0., 1.), 1.);
}

define_impl_blas! {
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

define_impl_blas!(
    CpuAsum,
    cpu_asum,
    ((sasum, f32, f32), (dasum, f64, f64)),
    (n: i32, x: &[Self], incx: i32)
);

define_impl_blas!(
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

define_impl_blas!(
    CpuCopy,
    cpu_copy,
    ((scopy, f32), (dcopy, f64)),
    (n: i32, x: &[Self], incx: i32, y: &mut [Self], incy: i32)
);

define_impl_blas!(
    CpyDot,
    cpu_dot,
    ((sdot, f32, f32), (ddot, f64, f64)),
    (n: i32, x: &[Self], incx: i32, y: &[Self], incy: i32)
);

define_impl_blas!(
    CpySdot,
    cpu_sdot,
    ((dsdot, f32, f64)),
    (n: i32, x: &[Self], incx: i32, y: &[Self], incy: i32)
);

define_impl_blas!(
    CpuNrm2,
    cpu_nrm2,
    ((snrm2, f32, f32), (dnrm2, f64, f64)),
    (n: i32, x: &[Self], incx: i32)
);

define_impl_blas!(
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

define_impl_blas!(
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

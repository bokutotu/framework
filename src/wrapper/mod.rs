pub mod cpu_blas;

#[macro_export]
macro_rules! define_impl {
    ($trait:ident, $fn_name:ident, ($(($call_fn:ident, $impl_ty:ty $(, $return_ty:ty)?)),*), ($($arg:ident : $ty: ty),*)) => {
        define_impl!(@define $trait, $fn_name, ($(($call_fn, $impl_ty $(, $return_ty)?)),*), ($($arg : $ty),*));
    };

    (@define $trait:ident, $fn_name:ident, ($(($call_fn:ident, $impl_ty:ty)),*), ($($arg:ident : $ty:ty),*)) => {
        pub trait $trait: Sized {
            fn $fn_name($($arg:$ty),*);
        }
        define_impl!(@r $trait, $fn_name, {$(($call_fn, $impl_ty ))*}, ($($arg: $ty),*));
    };

    (@define $trait:ident, $fn_name:ident, ($(($call_fn:ident, $impl_ty:ty, $return_ty:ty )),*), ($($arg:ident : $ty:ty),*)) => {
        pub trait $trait: Sized {
            type Out : Sized;
            fn $fn_name($($arg:$ty),*)  -> Self::Out ;
        }
        define_impl!(@r $trait, $fn_name, {$(($call_fn, $impl_ty, $return_ty ))*}, ($($arg: $ty),*));
    };

    (@r $trait:ident, $fn_name:ident, {($call_fn:ident, $impl_ty:ty ) $(($call_fn_:ident, $impl_ty_:ty ))*}, ($($arg:ident : $ty:ty),*)) => {
        impl $trait for $impl_ty {
            #[inline(always)]
            fn $fn_name( $($arg: $ty),* ) {
                unsafe {
                    $call_fn( $( $arg ),* )
                }
            }
        }
        define_impl!(@r $trait, $fn_name, {$(($call_fn_, $impl_ty_))*}, ($($arg: $ty),*));
    };

    (@r $trait:ident, $fn_name:ident, {($call_fn:ident, $impl_ty:ty, $return_ty:ty ) $(($call_fn_:ident, $impl_ty_:ty, $return_ty_:ty ))*}, ($($arg:ident : $ty:ty),*)) => {
        impl $trait for $impl_ty {
            type Out = $return_ty;
            #[inline(always)]
            fn $fn_name( $($arg: $ty),* ) -> Self::Out{
                unsafe {
                    $call_fn( $( $arg ),* )
                }
            }
        }
        define_impl!(@r $trait, $fn_name, {$(($call_fn_, $impl_ty_, $return_ty_))*}, ($($arg: $ty),*));
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
    define_impl!(
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
    define_impl!(
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
    define_impl!(
        FuncRN,
        func_r_n,
        ((float_func_r_n, f32, f64), (double_func_r_n, f64, f32)),
        (a: Self, b: Self)
    );
    assert_eq!(f32::func_r_n(0., 1.), 1.);
    assert_eq!(f64::func_r_n(0., 1.), 1.);
}

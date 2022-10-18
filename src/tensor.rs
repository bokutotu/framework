use std::clone::Clone;
use std::iter::Iterator;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape(Vec<isize>);

impl Shape {
    fn new(vec: Vec<isize>) -> Self {
        Shape(vec)
    }

    fn default_stride(&self) -> Stride {
        let mut res = vec![1];
        let mut shape = self.0.clone();
        shape.reverse();
        for i in 0..shape.len() - 1 {
            res.push(res[i] * shape[i]);
        }
        res.reverse();
        Stride::new(res)
    }

    fn is_default_stride(&self, stride: &Stride) -> bool {
        self.default_stride() == *stride
    }

    fn num_elms(&self) -> usize {
        self.0.iter().product::<isize>() as usize
    }

    fn valid_index(&self, index: &TensorIndex) {
        if self.0.len() != index.0.len() {
            panic!(
                "This tensor is {} dimensions but your index is {} dimensions",
                self.0.len(),
                index.0.len()
            );
        }
        index.0.iter().zip(self.0.iter()).for_each(|(idx, shape)| {
            if idx >= shape {
                panic!("this tensor's shape is {} but your index is {}", idx, shape)
            }
        });
    }

    fn iter(&self) -> ShapeIter {
        ShapeIter::new(self.clone())
    }
}

pub struct ShapeIter {
    shape: Shape,
    index: TensorIndex,
}

impl ShapeIter {
    fn new(shape: Shape) -> Self {
        let shape_len = shape.0.len();
        let mut index_vec = Vec::with_capacity(shape_len);
        for _ in 0..shape_len {
            index_vec.push(0);
        }

        let index = TensorIndex::new(index_vec);

        Self { shape, index }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stride(Vec<isize>);

impl Stride {
    fn new(vec: Vec<isize>) -> Self {
        Stride(vec)
    }

    // must use before valid_index
    fn cal_offset(&self, index: &TensorIndex) -> isize {
        self.0
            .iter()
            .zip(index.0.iter())
            .fold(0, |offset, (dim_index, dim_stride)| {
                offset + dim_index * dim_stride
            })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorIndex(Vec<isize>);

impl TensorIndex {
    fn new(index: Vec<isize>) -> Self {
        Self(index)
    }
}

pub(crate) struct OwnedInner<T> {
    pointer: *mut T,
    shape: Shape,
    stride: Stride,
    num_elm: usize,
}

impl<T: Copy + Clone> OwnedInner<T> {
    fn new(pointer: *mut T, shape: Shape, stride: Stride) -> Self {
        let num_elm = shape.num_elms();
        Self {
            pointer,
            shape,
            stride,
            num_elm,
        }
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn stride(&self) -> Stride {
        self.stride.clone()
    }
}

pub(crate) struct CpuInner<T>(OwnedInner<T>);
pub(crate) struct GpuInner<T>(OwnedInner<T>);

impl<T: Copy + Clone> CpuInner<T> {
    fn shape(&self) -> Shape {
        self.0.shape()
    }

    fn stride(&self) -> Stride {
        self.0.stride()
    }

    fn cpu_malloc(shape: Shape, stride: Stride) -> Self {
        let alloc_vec: Vec<T> = Vec::with_capacity(shape.num_elms());
        let pointer: *mut T = alloc_vec.as_ptr() as *mut T;
        std::mem::forget(alloc_vec);
        CpuInner(OwnedInner::new(pointer, shape, stride))
    }

    fn cal_offset(&self, index: &TensorIndex) -> isize {
        self.shape().valid_index(index);
        self.stride().cal_offset(index)
    }

    pub unsafe fn access_by_offset(&self, offset: isize) -> *mut T {
        self.0.pointer.offset(offset) as *mut T
    }

    pub unsafe fn access_by_idx(&self, index: &TensorIndex) -> *mut T {
        let offset = self.cal_offset(index);
        self.access_by_offset(offset)
    }

    fn to_vec(&self) -> Vec<T> {
        let inner = CpuInner::cpu_malloc(self.shape(), self.stride());
        for i in 0..self.0.num_elm {
            let offset = i.try_into().unwrap();
            unsafe {
                let inner_ptr = inner.access_by_offset(offset);
                let self_ptr = self.access_by_offset(offset);
                *inner_ptr = *self_ptr;
            }
        }
        inner.to_vec()
    }

    fn into_vec(self) -> Vec<T> {
        unsafe { Vec::from_raw_parts(self.0.pointer, self.0.num_elm, self.0.num_elm) }
    }

    fn from_vec(vec: Vec<T>, shape: Shape) -> Self {
        let stride = shape.default_stride();
        let inner = OwnedInner::new(vec.as_ptr() as *mut T, shape, stride);
        Self(inner)
    }
}

impl<T: Copy + Clone> Clone for CpuInner<T> {
    fn clone(&self) -> Self {
        let pointer: *mut T = self.to_vec().as_mut_ptr();
        let inner = OwnedInner::new(pointer, self.shape(), self.stride());
        Self(inner)
    }
}

impl<T> GpuInner<T> {
    fn shape(&self) -> Shape {
        self.0.shape.clone()
    }

    fn stride(&self) -> Stride {
        self.0.stride.clone()
    }

    fn gpu_malloc(shape: Shape, stride: Stride) -> Self {
        let _shape = shape;
        let _stride = stride;
        todo!()
    }

    fn cal_offset(self, index: TensorIndex) -> isize {
        self.shape().valid_index(&index);
        self.stride().cal_offset(&index)
    }
}

pub(crate) enum Pointer<T> {
    Gpu(GpuInner<T>),
    Cpu(CpuInner<T>),
}

impl<T: Copy + Clone> Pointer<T> {
    fn shape(&self) -> Shape {
        match self {
            Pointer::Gpu(inner) => inner.0.shape(),
            Pointer::Cpu(inner) => inner.0.shape(),
        }
    }

    fn stride(&self) -> Stride {
        match self {
            Pointer::Gpu(inner) => inner.0.stride(),
            Pointer::Cpu(inner) => inner.0.stride(),
        }
    }

    fn cpu_malloc(shape: Shape, stride: Stride) -> Self {
        let pointer = CpuInner::cpu_malloc(shape, stride);
        Pointer::Cpu(pointer)
    }

    fn gpu_malloc(shape: Shape, stride: Stride) -> Self {
        let pointer = GpuInner::gpu_malloc(shape, stride);
        Pointer::Gpu(pointer)
    }

    fn cal_offset(self, index: TensorIndex) -> isize {
        self.shape().valid_index(&index);
        self.stride().cal_offset(&index)
    }

    fn is_default_stride(&self) -> bool {
        self.shape().default_stride() == self.stride()
    }
}

pub struct Tensor<T> {
    inner: Pointer<T>,
}

impl<T: Copy + Clone> Tensor<T> {
    pub fn shape(&self) -> Shape {
        self.inner.shape()
    }

    pub fn stride(&self) -> Stride {
        self.inner.stride()
    }

    pub fn cpu_malloc_from_shape(shape: Vec<isize>) -> Self {
        let shape = Shape::new(shape);
        let stride = shape.default_stride();
        let pointer = Pointer::cpu_malloc(shape, stride);
        Tensor { inner: pointer }
    }

    pub fn gpu_malloc_from_shape(shape: Vec<isize>) -> Self {
        let shape = Shape::new(shape);
        let stride = shape.default_stride();
        let pointer = Pointer::gpu_malloc(shape, stride);
        Tensor { inner: pointer }
    }
}

macro_rules! impl_defalut_stride_test {
    ($fn_name:ident, $vec:expr, $ans:expr) => {
        #[test]
        fn $fn_name() {
            let shape = Shape::new($vec);
            let default_stride = shape.default_stride();
            assert_eq!(default_stride.0, $ans)
        }
    };
}

impl_defalut_stride_test!(test_default_stride_1d, vec![10], vec![1]);
impl_defalut_stride_test!(test_default_stride_2d, vec![2, 3], vec![3, 1]);
impl_defalut_stride_test!(test_default_stride_3d, vec![2, 3, 4], vec![12, 4, 1]);

macro_rules! impl_is_default_stride_test {
    (@inner $shape:ident, $stride:ident, $shape_vec:expr, $stride_vec:expr) => {
        let $shape = Shape::new($shape_vec);
        let $stride = Stride::new($stride_vec);
    };

    (@eq $fn_name:ident, $shape_vec:expr, $stride_vec:expr) => {
        #[test]
        fn $fn_name() {
            impl_is_default_stride_test!(@inner shape, stride, $shape_vec, $stride_vec);
            assert!(shape.is_default_stride(&stride));
        }
    };

    (@nq $fn_name:ident, $shape_vec:expr, $stride_vec:expr) => {
        #[test]
        fn $fn_name() {
            impl_is_default_stride_test!(@inner shape, stride, $shape_vec, $stride_vec);
            assert!(!shape.is_default_stride(&stride));
        }
    };
}

impl_is_default_stride_test!(@eq test_default_stride_1d_eq, vec![10], vec![1]);
impl_is_default_stride_test!(@nq test_default_stride_1d_nq, vec![10], vec![2]);
impl_is_default_stride_test!(@eq test_default_stride_2d_eq, vec![2, 3], vec![3,1]);
impl_is_default_stride_test!(@nq test_default_stride_2d_nq, vec![2, 3], vec![1,1]);
impl_is_default_stride_test!(@eq test_default_stride_3d_eq, vec![4, 2, 3], vec![6, 3,1]);
impl_is_default_stride_test!(@nq test_default_stride_3d_nq, vec![4, 2, 3], vec![5, 1,1]);

macro_rules! impl_num_elm_test {
    ($fn_name:ident, $shape:expr, $ans:expr) => {
        #[test]
        fn $fn_name() {
            let shape = Shape::new($shape);
            assert_eq!(shape.num_elms(), $ans);
        }
    };
}

impl_num_elm_test!(num_elm_1d, vec![2], 2);
impl_num_elm_test!(num_elm_2d, vec![3, 2], 6);
impl_num_elm_test!(num_elm_3d, vec![1, 2, 2], 4);
impl_num_elm_test!(num_elm_4d, vec![2, 3, 4, 5], 120);

macro_rules! impl_valid_index_test {
    (@should_panic $fn_name:ident, $shape:expr, $index:expr) => {
        impl_valid_index_test!(@impl_fn $fn_name, $shape, $index, should_panic, test);
    };

    (@should_success $fn_name:ident, $shape:expr, $index: expr) => {
        impl_valid_index_test!(@impl_fn $fn_name, $shape, $index, test);
    };

    (@impl_fn $fn_name:ident, $shape:expr, $index:expr, $($meta:meta),*) => {
        $(
            #[$meta]
        )*
        fn $fn_name() {
            let shape = Shape::new($shape);
            let index = TensorIndex::new($index);
            shape.valid_index(&index);
        }
    };
}

impl_valid_index_test!(@should_success valid_index_1d, vec![10], vec![3]);
impl_valid_index_test!(@should_panic valid_index_1d_panic, vec![10], vec![10]);

macro_rules! impl_cal_offset {
    ($fn_name:ident, $stride:expr, $index:expr, $ans:expr) => {
        #[test]
        fn $fn_name() {
            let stride = Stride::new($stride);
            let index = TensorIndex::new($index);
            let offset = stride.cal_offset(&index);
            assert_eq!(offset, $ans);
        }
    };
}

impl_cal_offset!(cal_offset_1d, vec![1], vec![3], 3);
impl_cal_offset!(cal_offset_2d, vec![4, 1], vec![2, 3], 11);
impl_cal_offset!(cal_offset_3d, vec![12, 4, 1], vec![2, 2, 3], 35);
impl_cal_offset!(cal_offset_4d, vec![24, 12, 4, 1], vec![2, 2, 2, 3], 83);

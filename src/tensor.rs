use std::clone::Clone;
use std::fmt::Debug;
use std::iter::Iterator;
use std::ops::Drop;
use std::ops::Index;

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
    reference_vec: Vec<isize>,
    index: usize,
}

impl ShapeIter {
    fn new(shape: Shape) -> Self {
        let index = 0;
        let default_stride_vec = shape.default_stride().0;
        let reference_vec = default_stride_vec[0..default_stride_vec.len() - 1].to_vec();
        Self {
            shape,
            reference_vec,
            index,
        }
    }
}

impl Iterator for ShapeIter {
    type Item = TensorIndex;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.shape.num_elms() {
            return None;
        }
        let mut _index = self.index as isize;
        let mut index_vec = Vec::new();
        for stride in self.reference_vec.iter() {
            index_vec.push(_index / stride);
            _index %= stride;
        }
        index_vec.push(_index);
        self.index += 1;
        Some(TensorIndex::new(index_vec))
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

impl<T: Copy + Clone> Clone for CpuInner<T> {
    fn clone(&self) -> Self {
        let cloned = CpuInner::cpu_malloc(self.shape(), self.shape().default_stride());
        for index in self.shape().iter() {
            unsafe {
                let cloned_ptr: *mut T = cloned.access_by_idx(&index);
                let self_ptr: *mut T = self.access_by_idx(&index);
                cloned_ptr.write(*self_ptr);
            }
        }
        cloned
    }
}

impl<T: Copy + Clone> Clone for GpuInner<T> {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl<T> Drop for CpuInner<T> {
    fn drop(&mut self) {
        let vec = unsafe { Vec::from_raw_parts(self.0.pointer, self.0.num_elm, self.0.num_elm) };
        drop(vec);
    }
}

impl<T: Copy + Clone> CpuInner<T> {
    fn shape(&self) -> Shape {
        self.0.shape()
    }

    fn stride(&self) -> Stride {
        self.0.stride()
    }

    fn is_default_stride(&self) -> bool {
        self.shape().is_default_stride(&self.stride())
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

    unsafe fn access_by_offset(&self, offset: isize) -> *mut T {
        self.0.pointer.offset(offset) as *mut T
    }

    unsafe fn access_by_idx(&self, index: &TensorIndex) -> *mut T {
        let offset = self.cal_offset(index);
        self.access_by_offset(offset)
    }

    fn to_vec(&self) -> Vec<T> {
        let cloned = self.clone();
        let vec =
            unsafe { Vec::from_raw_parts(cloned.0.pointer, cloned.0.num_elm, cloned.0.num_elm) };
        std::mem::forget(cloned);
        vec
    }

    fn from_vec(vec: Vec<T>, shape: Shape) -> Self {
        let stride = shape.default_stride();
        let inner = OwnedInner::new(vec.as_ptr() as *mut T, shape, stride);
        std::mem::forget(vec);
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

    fn is_default_stride(&self) -> bool {
        self.shape().is_default_stride(&self.stride())
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

    // GPUのポインタをそのまま返すわけには行かないので対策を後で考える
    pub unsafe fn access_by_idx(&self, index: &TensorIndex) -> *mut T {
        let _index = index;
        todo!();
    }

    fn to_vec(&self) -> Vec<T> {
        todo!();
    }

    fn into_vec(self) -> Vec<T> {
        todo!();
    }

    fn from_vec(vec: Vec<T>, shape: Shape) -> Self {
        let _vec = vec;
        let _shape = shape;
        todo!();
    }
}

impl<T> Drop for GpuInner<T> {
    fn drop(&mut self) {
        todo!();
    }
}

pub(crate) enum Pointer<T> {
    Gpu(GpuInner<T>),
    Cpu(CpuInner<T>),
}

impl<T: Copy + Clone> Pointer<T> {
    fn shape(&self) -> Shape {
        match self {
            Pointer::Gpu(inner) => inner.shape(),
            Pointer::Cpu(inner) => inner.shape(),
        }
    }

    fn is_default_stride(&self) -> bool {
        match self {
            Pointer::Gpu(inner) => inner.is_default_stride(),
            Pointer::Cpu(inner) => inner.is_default_stride(),
        }
    }

    fn stride(&self) -> Stride {
        match self {
            Pointer::Gpu(inner) => inner.stride(),
            Pointer::Cpu(inner) => inner.stride(),
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

    // fn cal_offset(self, index: TensorIndex) -> isize {
    //     self.shape().valid_index(&index);
    //     self.stride().cal_offset(&index)
    // }

    fn to_vec(&self) -> Vec<T> {
        match self {
            Pointer::Gpu(inner) => inner.to_vec(),
            Pointer::Cpu(inner) => inner.to_vec(),
        }
    }

    fn from_vec(vec: Vec<T>, shape: Shape, is_gpu: bool) -> Self {
        if is_gpu {
            return Pointer::Gpu(GpuInner::from_vec(vec, shape));
        }
        Pointer::Cpu(CpuInner::from_vec(vec, shape))
    }

    unsafe fn access_by_index(&self, index: &TensorIndex) -> *mut T {
        match self {
            Pointer::Cpu(inner) => inner.access_by_idx(index),
            Pointer::Gpu(inner) => inner.access_by_idx(index),
        }
    }
}

impl<T> Drop for Pointer<T> {
    fn drop(&mut self) {
        match self {
            Pointer::Gpu(inner) => drop(inner),
            Pointer::Cpu(inner) => drop(inner),
        };
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

    pub fn to_vec(&self) -> Vec<T> {
        self.inner.to_vec()
    }

    pub fn into_vec(self) -> Vec<T> {
        self.to_vec()
    }

    pub fn from_vec(vec: Vec<T>, shape: Shape, is_gpu: bool) -> Self {
        if vec.len() != shape.num_elms() {
            panic!("length of vectoro and shape is not collect");
        }
        Self {
            inner: Pointer::from_vec(vec, shape, is_gpu),
        }
    }

    unsafe fn access_by_index(&self, index: &TensorIndex) -> *mut T {
        self.inner.access_by_index(index)
    }

    fn is_default_stride(&self) -> bool {
        self.inner.is_default_stride()
    }
}

// impl<T> Drop for Tensor<T> {
//     fn drop(&mut self) {
//         drop(self.inner);
//     }
// }

impl<T: Copy + Clone> Index<TensorIndex> for Tensor<T> {
    type Output = T;

    fn index(&self, index: TensorIndex) -> &Self::Output {
        unsafe { &*self.access_by_index(&index) }
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

macro_rules! impl_shape_iter_test {
    ($fn_name:ident, $shape:expr, $($index:expr),*) => {
        #[test]
        fn $fn_name() {
            let shape = Shape::new($shape);
            let mut shape_iter = shape.iter();
            $(
                assert_eq!(shape_iter.next(), Some(TensorIndex::new($index)));
             )*
            assert_eq!(shape_iter.next(), None);
        }
    };
}

impl_shape_iter_test!(shape_iter_1d, vec![2], vec![0], vec![1]);
impl_shape_iter_test!(
    shape_iter_2d,
    vec![2, 3],
    vec![0, 0],
    vec![0, 1],
    vec![0, 2],
    vec![1, 0],
    vec![1, 1],
    vec![1, 2]
);
impl_shape_iter_test!(
    shape_iter_3d,
    vec![4, 2, 3],
    vec![0, 0, 0],
    vec![0, 0, 1],
    vec![0, 0, 2],
    vec![0, 1, 0],
    vec![0, 1, 1],
    vec![0, 1, 2],
    vec![1, 0, 0],
    vec![1, 0, 1],
    vec![1, 0, 2],
    vec![1, 1, 0],
    vec![1, 1, 1],
    vec![1, 1, 2],
    vec![2, 0, 0],
    vec![2, 0, 1],
    vec![2, 0, 2],
    vec![2, 1, 0],
    vec![2, 1, 1],
    vec![2, 1, 2],
    vec![3, 0, 0],
    vec![3, 0, 1],
    vec![3, 0, 2],
    vec![3, 1, 0],
    vec![3, 1, 1],
    vec![3, 1, 2]
);

macro_rules! impl_from_to_vec_test {
    ($fn_name:ident, $from_vec:expr, $shape:expr) => {
        #[test]
        fn $fn_name() {
            let tensor = Tensor::from_vec($from_vec, Shape::new($shape), false);
            let vec = tensor.to_vec();
            let into_vec = tensor.into_vec();
            assert_eq!(vec, $from_vec);
            assert_eq!(vec, into_vec);
        }
    };
}

impl_from_to_vec_test!(from_to_vec_1d, vec![0, 1, 2, 3, 4], vec![5]);
impl_from_to_vec_test!(from_to_vec_2d, vec![0, 1, 2, 3, 4, 5], vec![2, 3]);
impl_from_to_vec_test!(
    from_to_vec_3d,
    vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    vec![2, 3, 3]
);

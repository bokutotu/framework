#[derive(Clone, PartialEq, Debug)]
pub struct Shape ( Vec::<usize> );

impl Shape {
    fn new(vec: Vec<usize>) -> Self {
        Shape ( vec )
    }

    fn default_stride(&self) -> Stride {
        let mut res = vec![1];
        let mut shape = self.0.clone();
        shape.reverse();
        for i in 0..shape.len()-1 {
            res.push(res[i] * shape[i]);
        }
        res.reverse();
        Stride::new(res)
    }

    fn is_default_stride(&self, stride: &Stride) -> bool {
        self.default_stride() == *stride
    }

    fn num_elms(&self) -> usize {
        self.0.iter().fold(1, |x, y| { x * y })
    }

    fn valid_index(&self, index: &TensorIndex) {
        if self.0.len() != index.0.len() {
            panic!("This tensor is {} dimensions but your index is {} dimensions", 
                   self.0.len(), index.0.len());
        }
        index.0.iter().zip(self.0.iter())
            .for_each(|(idx, shape)| { 
                if idx >= shape {
                    panic!("this tensor's shape is {} but your index is {}",
                           idx, shape)
                }
            });
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Stride ( Vec::<usize> );

impl Stride {
    fn new(vec: Vec<usize>) -> Self {
        Stride ( vec )
    }

    // must use before valid_index
    fn cal_offset(&self, index: &TensorIndex) -> usize {
        self.0.iter()
            .zip(index.0.iter())
            .fold(1, |offset, (dim_index, dim_stride)| offset + dim_index * dim_stride)
    }
}

pub(crate) struct OwnedInner<T> {
    pointer: *mut T,
    shape: Shape,
    stride: Stride,
}

impl<T> OwnedInner<T> {
    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn stride(&self) -> Stride {
        self.stride.clone()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorIndex ( Vec<usize> );

pub(crate) struct CPUInner<T> ( OwnedInner<T> );
pub(crate) struct GPUInner<T> ( OwnedInner<T> );

impl<T> CPUInner<T> {
    fn cpu_malloc(shape: Shape, stride: Stride) -> Self{
        let alloc_vec: Vec::<T> = Vec::with_capacity(shape.num_elms());
        let pointer: *mut T = alloc_vec.as_ptr() as *mut T;
        std::mem::forget(alloc_vec);
        CPUInner(OwnedInner {pointer, shape, stride } )
    }
}

impl<T> GPUInner<T> {
    fn gpu_malloc(shape: Shape, stride: Stride) -> Self {
        todo!()
    }
}

pub(crate) enum Pointer<T> {
    GPU(GPUInner<T>),
    CPU(CPUInner<T>),
}

impl<T> Pointer<T> {
    fn shape(&self) -> Shape {
        match self {
            Pointer::GPU(inner) => inner.0.shape(),
            Pointer::CPU(inner) => inner.0.shape(),
        }
    }

    fn stride(&self) -> Stride {
        match self {
            Pointer::GPU(inner) => inner.0.stride(),
            Pointer::CPU(inner) => inner.0.stride()
        }
    }

    fn cpu_malloc(shape: Shape, stride: Stride) -> Self {
        let pointer = CPUInner::cpu_malloc(shape, stride);
        Pointer::CPU(pointer)
    }
    
    fn gpu_malloc(shape: Shape, stride: Stride) -> Self {
        let pointer = GPUInner::gpu_malloc(shape, stride);
        Pointer::GPU(pointer)
    }

    fn cal_offset(self, index: TensorIndex) -> usize {
        self.shape().clone().valid_index(&index);
        self.stride().clone().cal_offset(&index)
    }

    fn is_default_stride(&self) -> bool {
        self.shape().default_stride().0 == self.stride().0
    }
}

pub struct Tensor<T> {
    inner: Pointer<T>
}

impl<T> Tensor<T> {
    pub fn shape(&self) -> Shape{
        self.inner.shape()
    }

    pub fn stride(&self) -> Stride {
        self.inner.stride()
    }

    pub fn cpu_malloc_from_shape(shape: Vec<usize>) -> Self {
        let shape = Shape::new(shape);
        let stride = shape.default_stride();
        let pointer = Pointer::cpu_malloc(shape, stride);
        Tensor { inner: pointer }
    }

    pub fn gpu_malloc_from_shape(shape: Vec<usize>) -> Self {
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
impl_defalut_stride_test!(test_default_stride_3d, vec![2,3,4], vec![12, 4, 1]);

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
            let index = TensorIndex ( $index );
            shape.valid_index(&index);
        }
    };
}

impl_valid_index_test!(@should_success valid_index_1d, vec![10], vec![3]);
impl_valid_index_test!(@should_panic valid_index_1d_panic, vec![10], vec![10]);

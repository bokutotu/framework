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
        Stride::new(res)
    }

    fn num_elms(&self) -> usize {
        self.0.iter().fold(1, |x, y| { x * y })
    }

    fn valid_index(&self, index: &Index) {
        if self.0.len() != index.0.len() {
            panic!("This tensor is {} dimensions but your index is {} dimensions", 
                   self.0.len(), index.0.len());
        }
        let mut flag = true;
        index.iter().zip(self.0.iter())
            .for_each(|(idx, shpae)| { 
                if flag == (idx < shpae) {
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
    fn cal_offset(&self, index: &Index) -> usize {

    }
}

pub(crate) struct Inner<T> {
    pointer: *mut T,
    shape: Shape,
    stride: Stride,
}

impl<T> Inner<T> {
    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn stride(&self) -> Stride {
        self.stride.clone()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub Index ( Vec<usize> );

pub(crate) struct CPUInner<T> ( Inner<T> );
pub(crate) struct GPUInner<T> ( Inner<T> );

impl<T> CPUInner<T> {
    fn cpu_malloc(shape: Shape, stride: Stride) -> Self{
        shape.check_stride(&stride);
        let alloc_vec: Vec::<T> = Vec::with_capacity(shape.num_elms());
        let pointer: *mut T = alloc_vec.as_ptr() as *mut T;
        std::mem::forget(alloc_vec);
        CPUInner(Inner {pointer, shape, stride } )
    }

    fn cpu_cal_offset(&self, index: &Index) -> Self {

    }
}

impl<T> GPUInner<T> {
    fn gpu_malloc(shape: Shape, stride: Stride) -> Self {
        shape.check_stride(&stride);
        todo!()
    }

    fn gpu_cal_offset(index: &Index) -> usize {
        todo!();
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

    fn cal_index_from_shape(index: Index) -> usize {
        
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

use std::ptr::NonNull;

use crate::pointer_traits::TensorPointer;
use crate::shape::{Shape, Stride};
use crate::tensor::TensorBase;

impl<P: TensorPointer<Elem = E>, E: Copy> TensorBase<P, E> {
    pub fn from_vec(v: Vec<E>, shape: Shape) -> Self {
        let num_elm = v.len();
        let ptr = P::from_vec(v);
        let stride = shape.default_stride();
        TensorBase {
            ptr,
            shape,
            stride,
            num_elm,
        }
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<E> {
        self.ptr.to_vec()
    }

    #[inline]
    pub fn offset(&self, offset: isize) -> NonNull<E> {
        self.ptr.offset(offset)
    }

    #[inline]
    pub fn as_ptr(&self) -> *const E {
        self.ptr.as_ptr()
    }

    #[inline]
    pub fn shape(&self) -> Shape {
        self.shape.clone()
    }

    #[inline]
    pub fn shape_vec(&self) -> Vec<isize> {
        self.shape.clone().to_vec()
    }

    /// Change SHAPE to the one entered. stride is changed to the default stride
    #[inline]
    pub fn reshape(&mut self, shape: Shape) {
        if shape.num_elms() != self.shape().num_elms() {
            panic!("The input shape and the number of elements in the sensor do not match.");
        }
        self.stride = shape.default_stride();
        self.shape = shape;
    }

    #[inline]
    pub fn is_column_major(&self) -> bool {
        let dim = self.shape();
        let num_dim = dim.num_dim();
        dim[num_dim - 2] > dim[num_dim - 1]
    }

    #[inline]
    pub fn stride(&self) -> Stride {
        self.stride.clone()
    }

    #[inline]
    pub fn stride_vec(&self) -> Vec<isize> {
        self.stride().to_vec()
    }
}

#[test]
#[should_panic]
fn reshape_test() {
    use crate::tensor::CpuTensor;
    let a = vec![10, 20, 30];
    let mut a = CpuTensor::from_vec(a, Shape::new(vec![3]));
    a.reshape(Shape::new(vec![100000]));
}

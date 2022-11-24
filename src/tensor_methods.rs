use std::ptr::NonNull;

use crate::pointer_traits::TensorPointer;
use crate::shape::{Shape, Stride};
use crate::tensor::TensorBase;

impl<P: TensorPointer<Elem = E>, E: Copy> TensorBase<P, E> {
    pub fn from_vec(v: Vec<E>, shape: Shape) -> Self {
        let num_elm = v.len();
        let ptr = P::from_vec(v);
        let stride = shape.default_stride();
        if num_elm != shape.num_elms() {
            panic!("shape is not collect");
        }
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

    #[inline]
    pub fn num_elms(&self) -> usize {
        self.shape.num_elms()
    }

    #[inline]
    pub fn swap_axis(&mut self, a: usize, b: usize) {
        if usize::max(a, b) >= self.shape.len() {
            panic!("swap axis argument is smaller than axis length");
        }
        self.shape.swap(a, b);
        self.stride.swap(a, b);
    }

    #[inline]
    pub fn add_axis(&mut self, axis: usize) {
        if axis < self.shape().num_dim() {
            panic!("axis must be smaller than length of shape");
        }
        self.shape.add_axis_unchecked(axis);
        self.stride.add_axis_unchecked(axis);
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

#[test]
#[should_panic]
fn from_vec_panic() {
    use crate::tensor::CpuTensor;
    let a = vec![1., 2., 3., 4., 5.];
    let _ = CpuTensor::from_vec(a, Shape::new(vec![1, 2, 3, 4]));
}

#[test]
#[should_panic]
fn swap_axis_test_panic() {
    use crate::tensor::CpuTensor;
    let mut v = vec![];
    for i in 0..125 {
        v.push(i);
    }
    let mut a = CpuTensor::from_vec(v, Shape::new(vec![5, 5, 5]));
    a.swap_axis(1, 3);
}

#[test]
fn swap_axis_test() {
    use crate::tensor::CpuTensor;
    let mut v = vec![];
    for idx in 0..8 {
        v.push(idx);
    }
    let mut a = CpuTensor::from_vec(v, Shape::new(vec![2, 2, 2]));
    a.swap_axis(0, 2);
    let a_v = a.to_view();
    let a_v_v = a_v.into_owned().to_vec();
    assert_eq!(a_v_v, vec![0, 4, 2, 6, 1, 5, 3, 7]);
}

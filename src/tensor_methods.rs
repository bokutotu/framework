use std::ptr::NonNull;

use crate::pointer_traits::TensorPointer;
use crate::shape::Shape;
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
}

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

    pub fn to_vec(&self) -> Vec<E> {
        self.ptr.to_vec()
    }

    pub fn offset(&self, offset: isize) -> NonNull<E> {
        self.ptr.offset(offset)
    }

    pub fn as_ptr(&self) -> *const E {
        self.ptr.as_ptr()
    }
}

use crate::tensor::{CpuViewTensor, CpuTensor, TensorBase, CpuViewMutTensor};
use crate::pointer_traits::View;

macro_rules! impl_to_owned {
    (($($generics:tt)*), $self:ty,  $out:ty) => {
        impl<$($generics)*>  $self {
            pub fn into_owned(self) -> $out {
                let ptr = self.ptr.to_owned();
                let stride = self.stride.clone();
                let shape = self.shape.clone();
                let num_elm = self.num_elm;
                TensorBase { ptr, shape, stride, num_elm } as $out
            } 
        }
    };
}
impl_to_owned!((E: Copy), CpuViewTensor<E>, CpuTensor<E>);
impl_to_owned!((E: Copy), CpuViewMutTensor<E>, CpuTensor<E>);


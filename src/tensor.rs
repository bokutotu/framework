// use std::clone::Clone;
// use std::ops::Index;
// use std::ptr::NonNull;

use crate::pointer_cpu::{OwnedCpu, ViewCpu, ViewMutCpu};
use crate::pointer_traits::TensorPointer;
use crate::shape::{Shape, Stride, TensorIndex};

pub struct TensorBase<P, E>
where
    P: TensorPointer<Elem = E>,
{
    pub(crate) ptr: P,
    pub(crate) shape: Shape,
    pub(crate) stride: Stride,
    pub(crate) num_elm: usize,
}

pub type CpuTensor<E> = TensorBase<OwnedCpu<E>, E>;
pub type CpuViewTensor<E> = TensorBase<ViewCpu<E>, E>;
pub type CpuViewMutTensor<E> = TensorBase<ViewMutCpu<E>, E>;

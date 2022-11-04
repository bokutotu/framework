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

type CpuTensor<E> = TensorBase<OwnedCpu<E>, E>;
type CpuViewTensor<E> = TensorBase<ViewCpu<E>, E>;
type CpuViewMutTenssor<E> = TensorBase<ViewMutCpu<E>, E>;

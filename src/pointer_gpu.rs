use std::ptr::NonNull;

use crate::pointer_traits::{Cpu, Mut, Owned, TensorPointer, View, ViewMut};

#[repr(C)]
pub struct OwnedGpu<E> {
    ptr: NonNull<E>,
    len: usize,
    cap: usize,
}

#[repr(C)]
pub struct GpuViewPointer<E> {
    ptr: NonNull<E>,
    len: usize,
    offset: usize,
    cap: usize,
}

pub type ViewGpu<E> = GpuViewPointer<E>;
pub type ViewMutGpu<E> = GpuViewPointer<E>;

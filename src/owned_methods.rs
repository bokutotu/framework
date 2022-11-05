use crate::pointer_traits::Owned;
use crate::tensor::{CpuTensor, CpuViewMutTensor, CpuViewTensor};

impl<E: Copy> CpuTensor<E> {
    pub fn clone_mem_layout(&self) -> Self {
        let ptr = self.ptr.clone_mem_layout();
        let shape = self.shape.clone();
        let stride = self.stride.clone();
        Self {
            ptr,
            shape,
            stride,
            num_elm: self.num_elm,
        }
    }

    pub fn to_view(&self) -> CpuViewTensor<E> {
        let ptr = self.ptr.to_view(0);
        let shape = self.shape.clone();
        let stride = self.stride.clone();
        let num_elm = self.num_elm;
        CpuViewTensor {
            ptr,
            shape,
            stride,
            num_elm,
        }
    }
    pub fn to_view_mut(&mut self) -> CpuViewMutTensor<E> {
        let ptr = self.ptr.to_view_mut(0);
        let shape = self.shape.clone();
        let stride = self.stride.clone();
        let num_elm = self.num_elm;
        CpuViewMutTensor {
            ptr,
            shape,
            stride,
            num_elm,
        }
    }
}

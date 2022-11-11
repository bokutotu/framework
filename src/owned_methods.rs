use std::convert::TryInto;

use crate::index::TensorIndex;
use crate::pointer_traits::{Owned, ToSlice};
use crate::shape::{slice_update_offset, slice_update_shape_stride};
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

    pub fn slice(&self, index: TensorIndex) -> CpuViewTensor<E> {
        let offset = slice_update_offset(&self.shape, &self.stride, &index);
        let (shape, stride) = slice_update_shape_stride(&self.shape, &self.stride, &index);
        let ptr = self.ptr.to_view(offset.try_into().unwrap());
        let num_elm = self.num_elm;
        CpuViewTensor {
            ptr,
            shape,
            stride,
            num_elm,
        }
    }

    pub fn slice_mut(&mut self, index: TensorIndex) -> CpuViewMutTensor<E> {
        let offset = slice_update_offset(&self.shape, &self.stride, &index);
        let (shape, stride) = slice_update_shape_stride(&self.shape, &self.stride, &index);
        let ptr = self.ptr.to_view_mut(offset.try_into().unwrap());
        let num_elm = self.num_elm;
        CpuViewMutTensor {
            ptr,
            shape,
            stride,
            num_elm,
        }
    }

    pub fn to_slice<'a>(&'a self) -> &'a [E] {
        self.ptr.to_slice()
    }

    pub fn to_slice_mut<'a>(&'a mut self) -> &'a mut [E] {
        self.ptr.to_slice_mut()
    }
}

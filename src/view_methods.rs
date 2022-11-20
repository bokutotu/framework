use std::convert::TryInto;
use std::fmt::Debug;

use num_traits::Num;

use crate::pointer_cpu::{OwnedCpu, ViewCpu};
use crate::pointer_traits::{Mut, TensorPointer, ToSlice, View};
use crate::shape::cal_offset;
use crate::tensor::{CpuTensor, CpuViewMutTensor, TensorBase};

#[inline]
fn cpu_shrink_to<P, E>(a: TensorBase<P, E>) -> OwnedCpu<E>
where
    // P: View<AccessOutput = ViewCpu<E>, OwnedOutput = OwnedCpu<E>> + TensorPointer<Elem = E>,
    P: View<ViewCpu<E>, OwnedCpu<E>> + TensorPointer<Elem = E>,
    E: Copy + Num + Debug,
{
    let shape = a.shape.clone();
    let default_stride = shape.default_stride();
    let num_elm = shape.num_elms();

    let mut v: Vec<E> = Vec::with_capacity(num_elm);
    for _ in 0..num_elm {
        v.push(E::zero());
    }

    let mut ptr = OwnedCpu::from_vec(v);

    let shape_iter = shape.to_shape_iter();
    for index in shape_iter {
        let ptr_offset: usize = cal_offset(&shape, &default_stride, &index)
            .try_into()
            .unwrap();
        let a_offset = cal_offset(&shape, &a.stride, &index).try_into().unwrap();
        let other_ptr = a.ptr.access_by_offset_region(a_offset, 1);
        ptr.assign_region(&other_ptr, ptr_offset, 1);
    }
    ptr
}

impl<P: TensorPointer<Elem = E>, E> TensorBase<P, E>
where
    P: TensorPointer<Elem = E> + ToSlice + View<ViewCpu<E>, OwnedCpu<E>>,
    E: Copy + Num + Debug,
{
    #[inline]
    pub fn into_owned(self) -> CpuTensor<E> {
        if self.stride == self.shape.default_stride() {
            TensorBase {
                ptr: self.ptr.to_owned(),
                shape: self.shape.clone(),
                stride: self.stride.clone(),
                num_elm: self.num_elm,
            }
        } else {
            let shape = self.shape.clone();
            let stride = shape.default_stride();
            let num_elm = shape.num_elms();
            let ptr = cpu_shrink_to::<P, E>(self);
            TensorBase {
                ptr,
                shape,
                stride,
                num_elm,
            }
        }
    }

    #[inline]
    pub fn to_slice(&'_ self) -> &'_ [E] {
        let mut sorted_stride = self.stride.to_vec();
        sorted_stride.sort();
        sorted_stride.reverse();
        self.ptr.to_slice()
    }
}

impl<E: Copy> CpuViewMutTensor<E> {
    #[inline]
    pub fn to_slice_mut(&'_ self) -> &'_ mut [E] {
        let mut sorted_stride = self.stride.to_vec();
        sorted_stride.sort();
        sorted_stride.reverse();
        self.ptr.to_slice_mut()
    }
}

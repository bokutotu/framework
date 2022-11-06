use std::convert::TryInto;

use num_traits::Num;

use crate::pointer_cpu::{OwnedCpu, ViewCpu};
use crate::pointer_traits::{TensorPointer, View};
use crate::shape::cal_offset;
use crate::tensor::{CpuTensor, CpuViewMutTensor, CpuViewTensor, TensorBase};

fn cpu_shrink_to<P, E>(a: TensorBase<P, E>) -> OwnedCpu<E>
where
    P: View<AccessOutput = ViewCpu<E>, OwnedOutput = OwnedCpu<E>> + TensorPointer<Elem = E>,
    E: Copy + Num,
{
    let shape = a.shape.clone();
    let default_stride = shape.default_stride();
    let num_elm = shape.num_elms();

    let mut v: Vec<E> = Vec::with_capacity(num_elm);
    for _ in 0..num_elm {
        v.push(E::zero());
    }

    let shape_iter = shape.to_shape_iter();
    for index in shape_iter {
        let v_offset: usize = cal_offset(&shape, &default_stride, &index)
            .try_into()
            .unwrap();
        let a_offset = cal_offset(&shape, &a.stride, &index).try_into().unwrap();
        v[v_offset] = unsafe { *a.ptr.access_by_offset_region(a_offset, 1).as_ptr() };
    }
    OwnedCpu::from_vec(v)
}

macro_rules! impl_to_owned {
    (($($generics:tt)*), $self:ty,  $out:ty) => {
        impl<$($generics)*>  $self {
            pub fn into_owned(self) -> $out {
                if self.stride == self.shape.default_stride() {
                    return TensorBase {
                        ptr: self.ptr.to_owned(),
                        shape: self.shape.clone(),
                        stride: self.stride.clone(),
                        num_elm: self.num_elm
                    };
                } else {
                    let shape = self.shape.clone();
                    let stride = shape.default_stride();
                    let num_elm = shape.num_elms();
                    let ptr = cpu_shrink_to(self);
                    return TensorBase { ptr, shape, stride, num_elm };
                }
            }
        }
    };
}
impl_to_owned!((E: Copy + Num), CpuViewTensor<E>, CpuTensor<E>);
impl_to_owned!((E: Copy + Num), CpuViewMutTensor<E>, CpuTensor<E>);

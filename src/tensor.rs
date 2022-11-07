use crate::pointer_cpu::{OwnedCpu, ViewCpu, ViewMutCpu};
use crate::pointer_traits::TensorPointer;
use crate::shape::{Shape, Stride};

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

#[test]
fn to_view_to_onwend_test() {
    use crate::shape::Shape;
    let from_vec = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
    let from_array = CpuTensor::from_vec(from_vec.clone(), Shape::new(vec![3, 3]));
    let view = from_array.to_view();
    let to_owned = view.into_owned();
    assert_eq!(from_vec, to_owned.to_vec());
}

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

#[test]
fn shrink_to_view_test() {
    use crate::index;
    let mut v = Vec::new();
    for i in 0..25 {
        v.push(i);
    }
    let a = CpuTensor::from_vec(v, Shape::new(vec![5, 5]));
    let av = a.slice(index![2..4, ..;2]);
    let avv = av.into_owned().to_vec();
    let ans = vec![10, 12, 14, 15, 17, 19];
    assert_eq!(ans, avv);
}

#[test]
fn to_slice_view() {
    let v = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
    let a = CpuTensor::from_vec(v.clone(), Shape::new(vec![3, 3]));
    let a = a.to_view();
    let slice = a.to_slice();
    assert_eq!(&v, slice)
}

#[test]
fn index_test() {
    use crate::index;
    use crate::shape::Shape;
    use crate::tensor::CpuTensor;

    let mut a = vec![];
    for i in 0..125 {
        a.push(i);
    }
    let a = CpuTensor::from_vec(a, Shape::new(vec![5, 5, 5]));
    let a = a.slice(index![.., 2, ..]);
    let a_vec = a.into_owned().to_vec();
    let ans = [
        10, 11, 12, 13, 14, 35, 36, 37, 38, 39, 60, 61, 62, 63, 64, 85, 86, 87, 88, 89, 110, 111,
        112, 113, 114,
    ];
    assert_eq!(a_vec, ans);
}

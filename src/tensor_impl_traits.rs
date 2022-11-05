// use std::ops::Index;
//
// use crate::{tensor::TensorBase, pointer_traits::{TensorPointer, View}, shape::TensorIndex};
//
// impl<P, E> Index<TensorIndex> for TensorBase<P, E>
// where
//     P: TensorPointer<Elem = E>,
// {
//     type Output<PV: TensorPointer<Elem=E> + View> = TensorBase<PV, E>;
//     fn index(&self, index: TensorIndex) -> &Self::Output<PV> {
//         // let offset = index.cal_offset();
//         todo!();
//     }
// }

use std::fmt::Debug;
use std::iter::Iterator;

use crate::index::TensorIndex;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Shape(Vec<isize>);

impl Shape {
    pub fn new(vec: Vec<isize>) -> Self {
        Shape(vec)
    }

    pub(crate) fn default_stride(&self) -> Stride {
        let mut res = vec![1];
        let mut shape = self.0.clone();
        shape.reverse();
        for i in 0..shape.len() - 1 {
            res.push(res[i] * shape[i]);
        }
        res.reverse();
        Stride::new(res)
    }

    pub(crate) fn is_default_stride(&self, stride: &Stride) -> bool {
        self.default_stride() == *stride
    }

    pub(crate) fn num_elms(&self) -> usize {
        self.0.iter().product::<isize>() as usize
    }

    // pub(crate) fn valid_index(&self, index: &TensorIndex) {
    //     if self.0.len() != index.0.len() {
    //         panic!(
    //             "This tensor is {} dimensions but your index is {} dimensions",
    //             self.0.len(),
    //             index.0.len()
    //         );
    //     }
    //     index.0.iter().zip(self.0.iter()).for_each(|(idx, shape)| {
    //         if idx >= shape {
    //             panic!("this tensor's shape is {} but your index is {}", idx, shape)
    //         }
    //     });
    // }

    pub(crate) fn iter(&self) -> ShapeIter {
        ShapeIter::new(self.clone())
    }
}

pub struct ShapeIter {
    shape: Shape,
    reference_vec: Vec<isize>,
    index: usize,
}

impl ShapeIter {
    pub(crate) fn new(shape: Shape) -> Self {
        let index = 0;
        let default_stride_vec = shape.default_stride().0;
        let reference_vec = default_stride_vec[0..default_stride_vec.len() - 1].to_vec();
        Self {
            shape,
            reference_vec,
            index,
        }
    }
}

// impl Iterator for ShapeIter {
//     type Item = TensorIndex;
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index == self.shape.num_elms() {
//             return None;
//         }
//         let mut _index = self.index as isize;
//         let mut index_vec = Vec::new();
//         for stride in self.reference_vec.iter() {
//             index_vec.push(_index / stride);
//             _index %= stride;
//         }
//         index_vec.push(_index);
//         self.index += 1;
//         Some(TensorIndex::new(index_vec))
//     }
// }

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stride(Vec<isize>);

impl Stride {
    pub(crate) fn new(vec: Vec<isize>) -> Self {
        Stride(vec)
    }

    // must use before valid_index
    // pub(crate) fn cal_offset(&self, index: &TensorIndex) -> isize {
    //     self.0
    //         .iter()
    //         .zip(index.0.iter())
    //         .fold(0, |offset, (dim_index, dim_stride)| {
    //             offset + dim_index * dim_stride
    //         })
    // }
}

// #[derive(Clone, Debug, PartialEq, Eq)]
// pub struct TensorIndex(Vec<isize>);
//
// impl TensorIndex {
//     pub(crate) fn new(index: Vec<isize>) -> Self {
//         Self(index)
//     }
// }

// macro_rules! impl_defalut_stride_test {
//     ($fn_name:ident, $vec:expr, $ans:expr) => {
//         #[test]
//         fn $fn_name() {
//             let shape = Shape::new($vec);
//             let default_stride = shape.default_stride();
//             assert_eq!(default_stride.0, $ans)
//         }
//     };
// }
//
// impl_defalut_stride_test!(test_default_stride_1d, vec![10], vec![1]);
// impl_defalut_stride_test!(test_default_stride_2d, vec![2, 3], vec![3, 1]);
// impl_defalut_stride_test!(test_default_stride_3d, vec![2, 3, 4], vec![12, 4, 1]);
//
// macro_rules! impl_is_default_stride_test {
//     (@inner $shape:ident, $stride:ident, $shape_vec:expr, $stride_vec:expr) => {
//         let $shape = Shape::new($shape_vec);
//         let $stride = Stride::new($stride_vec);
//     };
//
//     (@eq $fn_name:ident, $shape_vec:expr, $stride_vec:expr) => {
//         #[test]
//         fn $fn_name() {
//             impl_is_default_stride_test!(@inner shape, stride, $shape_vec, $stride_vec);
//             assert!(shape.is_default_stride(&stride));
//         }
//     };
//
//     (@nq $fn_name:ident, $shape_vec:expr, $stride_vec:expr) => {
//         #[test]
//         fn $fn_name() {
//             impl_is_default_stride_test!(@inner shape, stride, $shape_vec, $stride_vec);
//             assert!(!shape.is_default_stride(&stride));
//         }
//     };
// }
//
// impl_is_default_stride_test!(@eq test_default_stride_1d_eq, vec![10], vec![1]);
// impl_is_default_stride_test!(@nq test_default_stride_1d_nq, vec![10], vec![2]);
// impl_is_default_stride_test!(@eq test_default_stride_2d_eq, vec![2, 3], vec![3,1]);
// impl_is_default_stride_test!(@nq test_default_stride_2d_nq, vec![2, 3], vec![1,1]);
// impl_is_default_stride_test!(@eq test_default_stride_3d_eq, vec![4, 2, 3], vec![6, 3,1]);
// impl_is_default_stride_test!(@nq test_default_stride_3d_nq, vec![4, 2, 3], vec![5, 1,1]);
//
// macro_rules! impl_num_elm_test {
//     ($fn_name:ident, $shape:expr, $ans:expr) => {
//         #[test]
//         fn $fn_name() {
//             let shape = Shape::new($shape);
//             assert_eq!(shape.num_elms(), $ans);
//         }
//     };
// }
//
// impl_num_elm_test!(num_elm_1d, vec![2], 2);
// impl_num_elm_test!(num_elm_2d, vec![3, 2], 6);
// impl_num_elm_test!(num_elm_3d, vec![1, 2, 2], 4);
// impl_num_elm_test!(num_elm_4d, vec![2, 3, 4, 5], 120);

// macro_rules! impl_valid_index_test {
//     (@should_panic $fn_name:ident, $shape:expr, $index:expr) => {
//         impl_valid_index_test!(@impl_fn $fn_name, $shape, $index, should_panic, test);
//     };
//
//     (@should_success $fn_name:ident, $shape:expr, $index: expr) => {
//         impl_valid_index_test!(@impl_fn $fn_name, $shape, $index, test);
//     };
//
//     (@impl_fn $fn_name:ident, $shape:expr, $index:expr, $($meta:meta),*) => {
//         $(
//             #[$meta]
//         )*
//         fn $fn_name() {
//             let shape = Shape::new($shape);
//             let index = TensorIndex::new($index);
//             shape.valid_index(&index);
//         }
//     };
// }
//
// impl_valid_index_test!(@should_success valid_index_1d, vec![10], vec![3]);
// impl_valid_index_test!(@should_panic valid_index_1d_panic, vec![10], vec![10]);
//
// macro_rules! impl_cal_offset {
//     ($fn_name:ident, $stride:expr, $index:expr, $ans:expr) => {
//         #[test]
//         fn $fn_name() {
//             let stride = Stride::new($stride);
//             let index = TensorIndex::new($index);
//             let offset = stride.cal_offset(&index);
//             assert_eq!(offset, $ans);
//         }
//     };
// }
//
// impl_cal_offset!(cal_offset_1d, vec![1], vec![3], 3);
// impl_cal_offset!(cal_offset_2d, vec![4, 1], vec![2, 3], 11);
// impl_cal_offset!(cal_offset_3d, vec![12, 4, 1], vec![2, 2, 3], 35);
// impl_cal_offset!(cal_offset_4d, vec![24, 12, 4, 1], vec![2, 2, 2, 3], 83);
//
// macro_rules! impl_shape_iter_test {
//     ($fn_name:ident, $shape:expr, $($index:expr),*) => {
//         #[test]
//         fn $fn_name() {
//             let shape = Shape::new($shape);
//             let mut shape_iter = shape.iter();
//             $(
//                 assert_eq!(shape_iter.next(), Some(TensorIndex::new($index)));
//              )*
//             assert_eq!(shape_iter.next(), None);
//         }
//     };
// }
//
// impl_shape_iter_test!(shape_iter_1d, vec![2], vec![0], vec![1]);
// impl_shape_iter_test!(
//     shape_iter_2d,
//     vec![2, 3],
//     vec![0, 0],
//     vec![0, 1],
//     vec![0, 2],
//     vec![1, 0],
//     vec![1, 1],
//     vec![1, 2]
// );
// impl_shape_iter_test!(
//     shape_iter_3d,
//     vec![4, 2, 3],
//     vec![0, 0, 0],
//     vec![0, 0, 1],
//     vec![0, 0, 2],
//     vec![0, 1, 0],
//     vec![0, 1, 1],
//     vec![0, 1, 2],
//     vec![1, 0, 0],
//     vec![1, 0, 1],
//     vec![1, 0, 2],
//     vec![1, 1, 0],
//     vec![1, 1, 1],
//     vec![1, 1, 2],
//     vec![2, 0, 0],
//     vec![2, 0, 1],
//     vec![2, 0, 2],
//     vec![2, 1, 0],
//     vec![2, 1, 1],
//     vec![2, 1, 2],
//     vec![3, 0, 0],
//     vec![3, 0, 1],
//     vec![3, 0, 2],
//     vec![3, 1, 0],
//     vec![3, 1, 1],
//     vec![3, 1, 2]
// );

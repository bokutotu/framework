use std::fmt::Debug;
use std::iter::Iterator;
use std::ops::Deref;

use crate::index::TensorIndex;
// use for tests
#[allow(unused_imports)]
use crate::index;

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

    pub fn is_default_stride(&self, stride: &Stride) -> bool {
        self.default_stride() == *stride
    }

    pub fn num_elms(&self) -> usize {
        self.0.iter().product::<isize>() as usize
    }

    pub fn to_shape_iter(&self) -> ShapeIter {
        ShapeIter::new(self.clone())
    }

    pub fn num_dim(&self) -> usize {
        self.0.len()
    }
}

impl Deref for Shape {
    type Target = Vec<isize>;
    fn deref(&self) -> &Self::Target {
        &self.0
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

impl Iterator for ShapeIter {
    type Item = TensorIndex;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.shape.num_elms() {
            return None;
        }
        let mut _index = self.index as isize;
        let mut index_vec = Vec::new();
        for stride in self.reference_vec.iter() {
            index_vec.push(_index / stride);
            _index %= stride;
        }
        index_vec.push(_index);
        self.index += 1;
        Some(TensorIndex::from_single_elm_vec(index_vec))
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stride(Vec<isize>);

impl Stride {
    pub(crate) fn new(vec: Vec<isize>) -> Self {
        Stride(vec)
    }

    pub(crate) fn to_vec(&self) -> Vec<isize> {
        self.0.clone()
    }
}

impl Deref for Stride {
    type Target = Vec<isize>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// index is not collect then panic
pub fn valid_index(shape: &Shape, stride: &Stride, index: &TensorIndex) {
    if shape
        .iter()
        .zip(stride.iter())
        .zip(index.iter())
        .any(|((sh, st), idx)| {
            let e = match idx.end {
                Some(e) => e,
                None => idx.start,
            };
            sh.abs() <= idx.start.abs() && sh.abs() <= e.abs() && st <= &idx.step.abs()
        })
    {
        panic!("index is not collect");
    }
}

pub fn cal_offset(shape: &Shape, stride: &Stride, index: &TensorIndex) -> isize {
    valid_index(shape, stride, index);
    if index.is_point_single_elm() {
        stride
            .iter()
            .zip(index.iter())
            .fold(0, |offset, (st, idx)| offset + st * idx.start)
    } else {
        panic!("this index points region so cannot cal_offset");
    }
}

pub fn slice_update_shape_stride(
    shape: &Shape,
    stride: &Stride,
    index: &TensorIndex,
) -> (Shape, Stride) {
    valid_index(shape, stride, index);
    if !(shape.len() == stride.len() && shape.len() == index.len()) {
        panic!("length of shape, stride, index must be same");
    }
    let mut skip_idx = vec![];
    for (i, idx) in index.iter().enumerate() {
        if idx.is_range() {
            skip_idx.push(i);
        }
    }

    let mut shape_vec = index
        .iter()
        .enumerate()
        .map(|(idx, x)| {
            let mut end = x.end.unwrap_or(x.start);
            if end < 0 {
                end += shape[idx];
            }
            let num_elm = (end - x.start).abs() + 1;
            num_elm / x.step + num_elm % x.step
        })
        .collect::<Vec<isize>>();

    let mut stride_vec = index
        .iter()
        .zip(stride.iter())
        .map(|(idx, st)| st * idx.step)
        .collect::<Vec<isize>>();

    for x in skip_idx.iter() {
        let _ = shape_vec.remove(*x);
        let _ = stride_vec.remove(*x);
    }
    let shape = Shape::new(shape_vec);
    let stride = Stride::new(stride_vec);
    (shape, stride)
}

pub fn slice_update_offset(shape: &Shape, stride: &Stride, index: &TensorIndex) -> isize {
    valid_index(shape, stride, index);
    if !(shape.len() == stride.len() && shape.len() == index.len()) {
        panic!("length of shape, stride, index must be same");
    }

    stride
        .iter()
        .zip(index.iter())
        .fold(0, |prev, (st, idx)| prev + st * idx.start)
}

macro_rules! impl_defalut_stride_test {
    ($fn_name:ident, $vec:expr, $ans:expr) => {
        #[test]
        fn $fn_name() {
            let shape = Shape::new($vec);
            let default_stride = shape.default_stride();
            assert_eq!(default_stride.0, $ans)
        }
    };
}

impl_defalut_stride_test!(test_default_stride_1d, vec![10], vec![1]);
impl_defalut_stride_test!(test_default_stride_2d, vec![2, 3], vec![3, 1]);
impl_defalut_stride_test!(test_default_stride_3d, vec![2, 3, 4], vec![12, 4, 1]);

macro_rules! impl_is_default_stride_test {
    (@inner $shape:ident, $stride:ident, $shape_vec:expr, $stride_vec:expr) => {
        let $shape = Shape::new($shape_vec);
        let $stride = Stride::new($stride_vec);
    };

    (@eq $fn_name:ident, $shape_vec:expr, $stride_vec:expr) => {
        #[test]
        fn $fn_name() {
            impl_is_default_stride_test!(@inner shape, stride, $shape_vec, $stride_vec);
            assert!(shape.is_default_stride(&stride));
        }
    };

    (@nq $fn_name:ident, $shape_vec:expr, $stride_vec:expr) => {
        #[test]
        fn $fn_name() {
            impl_is_default_stride_test!(@inner shape, stride, $shape_vec, $stride_vec);
            assert!(!shape.is_default_stride(&stride));
        }
    };
}

impl_is_default_stride_test!(@eq test_default_stride_1d_eq, vec![10], vec![1]);
impl_is_default_stride_test!(@nq test_default_stride_1d_nq, vec![10], vec![2]);
impl_is_default_stride_test!(@eq test_default_stride_2d_eq, vec![2, 3], vec![3,1]);
impl_is_default_stride_test!(@nq test_default_stride_2d_nq, vec![2, 3], vec![1,1]);
impl_is_default_stride_test!(@eq test_default_stride_3d_eq, vec![4, 2, 3], vec![6, 3,1]);
impl_is_default_stride_test!(@nq test_default_stride_3d_nq, vec![4, 2, 3], vec![5, 1,1]);

macro_rules! impl_num_elm_test {
    ($fn_name:ident, $shape:expr, $ans:expr) => {
        #[test]
        fn $fn_name() {
            let shape = Shape::new($shape);
            assert_eq!(shape.num_elms(), $ans);
        }
    };
}

impl_num_elm_test!(num_elm_1d, vec![2], 2);
impl_num_elm_test!(num_elm_2d, vec![3, 2], 6);
impl_num_elm_test!(num_elm_3d, vec![1, 2, 2], 4);
impl_num_elm_test!(num_elm_4d, vec![2, 3, 4, 5], 120);

macro_rules! impl_valid_index_test {
    (@should_panic $fn_name:ident, $shape:expr, $index:expr) => {
        impl_valid_index_test!(@impl_fn $fn_name, $shape, $index, should_panic, test);
    };

    (@should_success $fn_name:ident, $shape:expr, $index: expr) => {
        impl_valid_index_test!(@impl_fn $fn_name, $shape, $index, test);
    };

    (@impl_fn $fn_name:ident, $shape:expr, $index:expr, $($meta:meta),*) => {
        $(
            #[$meta]
        )*
        fn $fn_name() {
            let shape = Shape::new($shape);
            let stride = shape.default_stride();
            let index = TensorIndex::from_single_elm_vec($index);
            valid_index(&shape, &stride, &index);
        }
    };
}

impl_valid_index_test!(@should_success valid_index_1d, vec![10], vec![3]);
impl_valid_index_test!(@should_panic valid_index_1d_panic, vec![10], vec![10]);

macro_rules! impl_cal_offset {
    ($fn_name:ident, $shape:expr, $index:expr, $ans:expr) => {
        #[test]
        fn $fn_name() {
            let shape = Shape::new($shape);
            let stride = shape.default_stride();
            let index = TensorIndex::from_single_elm_vec($index);
            let offset = cal_offset(&shape, &stride, &index);
            assert_eq!(offset, $ans);
        }
    };
}

impl_cal_offset!(cal_offset_1d, vec![4], vec![3], 3);
impl_cal_offset!(cal_offset_2d, vec![4, 4], vec![2, 3], 11);
impl_cal_offset!(cal_offset_3d, vec![5, 3, 4], vec![2, 2, 3], 35);

macro_rules! impl_shape_iter_test {
    ($fn_name:ident, $shape:expr, $($index:expr),*) => {
        #[test]
        fn $fn_name() {
            let shape = Shape::new($shape);
            let mut shape_iter = shape.to_shape_iter();
            $(
                assert_eq!(
                    shape_iter.next().unwrap(),
                    TensorIndex::from_single_elm_vec($index)
                );
             )*
            assert_eq!(shape_iter.next(), None);
        }
    };
}

impl_shape_iter_test!(shape_iter_1d, vec![2], vec![0], vec![1]);
impl_shape_iter_test!(
    shape_iter_2d,
    vec![2, 3],
    vec![0, 0],
    vec![0, 1],
    vec![0, 2],
    vec![1, 0],
    vec![1, 1],
    vec![1, 2]
);
impl_shape_iter_test!(
    shape_iter_3d,
    vec![4, 2, 3],
    vec![0, 0, 0],
    vec![0, 0, 1],
    vec![0, 0, 2],
    vec![0, 1, 0],
    vec![0, 1, 1],
    vec![0, 1, 2],
    vec![1, 0, 0],
    vec![1, 0, 1],
    vec![1, 0, 2],
    vec![1, 1, 0],
    vec![1, 1, 1],
    vec![1, 1, 2],
    vec![2, 0, 0],
    vec![2, 0, 1],
    vec![2, 0, 2],
    vec![2, 1, 0],
    vec![2, 1, 1],
    vec![2, 1, 2],
    vec![3, 0, 0],
    vec![3, 0, 1],
    vec![3, 0, 2],
    vec![3, 1, 0],
    vec![3, 1, 1],
    vec![3, 1, 2]
);

macro_rules! impl_slice_update_test {
    ($fn_name:ident, $shape:expr, $stride:expr, $index:expr, $ans_shape:expr, $ans_stride:expr) => {
        #[test]
        fn $fn_name() {
            let shape = Shape::new($shape);
            let stride = Stride::new($stride);
            let (shape, stride) = slice_update_shape_stride(&shape, &stride, &$index);
            assert_eq!(Shape::new($ans_shape), shape);
            assert_eq!(Stride::new($ans_stride), stride);
        }
    };
}

impl_slice_update_test!(
    stride_shape_update_default_stride_1d,
    vec![20],
    vec![1],
    index!(2..15),
    vec![13],
    vec![1]
);
impl_slice_update_test!(
    stride_shape_update_default_stride_1d_stride,
    vec![20],
    vec![1],
    index!(0..10;2),
    vec![5],
    vec![2]
);
impl_slice_update_test!(
    stride_shape_update_custom_stride_1d,
    vec![20],
    vec![2],
    index![2..=4;2],
    vec![2],
    vec![4]
);
impl_slice_update_test!(
    stride_shape_update_defalut_stride_2d,
    vec![15, 10],
    vec![10, 1],
    index![2, 2..5],
    vec![3],
    vec![1]
);

impl_slice_update_test!(
    stride_shape_update_default_stride_1d_range_full_stride,
    vec![15],
    vec![1],
    index![..;2],
    vec![8],
    vec![2]
);

macro_rules! impl_slice_update_test {
    ($fn_name:ident, $shape:expr, $stride:expr, $index:expr, $ans:expr) => {
        #[test]
        fn $fn_name() {
            let shape = Shape::new($shape);
            let stride = Stride::new($stride);
            let offset = slice_update_offset(&shape, &stride, &$index);
            assert_eq!(offset, $ans)
        }
    };
}

impl_slice_update_test!(default_stride_1d_offset, vec![10], vec![1], index![3..4], 3);
impl_slice_update_test!(
    default_stride_2d_offset,
    vec!(5, 2),
    vec![2, 1],
    index![2..3, 1..3],
    5
);

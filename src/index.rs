use std::ops::{Deref, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inner {
    pub(crate) start: isize,
    pub(crate) end: Option<isize>,
    pub(crate) step: isize,
}

impl Inner {
    pub fn new(start: isize, end: Option<isize>, step: isize) -> Self {
        Self { start, end, step }
    }

    pub fn set_step(self, step: isize) -> Self {
        let mut s = self;
        s.step = step;
        s
    }

    fn is_point_single_elm(&self) -> bool {
        self.end.is_none()
    }
}

impl From<Range<isize>> for Inner {
    fn from(range: Range<isize>) -> Self {
        Inner::new(range.start, Some(range.end - 1), 1)
    }
}

impl From<RangeFull> for Inner {
    fn from(_: RangeFull) -> Self {
        Inner::new(0, Some(-1), 1)
    }
}

impl From<RangeTo<isize>> for Inner {
    fn from(range: RangeTo<isize>) -> Self {
        Inner::new(0, Some(range.end - 1), 1)
    }
}

impl From<RangeFrom<isize>> for Inner {
    fn from(range: RangeFrom<isize>) -> Self {
        Inner::new(range.start, None, 1)
    }
}

impl From<RangeInclusive<isize>> for Inner {
    fn from(range: RangeInclusive<isize>) -> Inner {
        Inner::new(*range.start(), Some(*range.end()), 1)
    }
}

impl From<isize> for Inner {
    fn from(index: isize) -> Self {
        Inner::new(index, None, 1)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TensorIndex(Vec<Inner>);

impl TensorIndex {
    pub(crate) fn from_single_elm_vec(v: Vec<isize>) -> Self {
        let v = v
            .iter()
            .map(|index| Inner::from(*index))
            .collect::<Vec<Inner>>();

        Self::from(v)
    }

    pub(crate) fn is_point_single_elm(&self) -> bool {
        self.iter()
            .map(|item| item.is_point_single_elm())
            .any(|x| x)
    }

    fn push(&mut self, inner: Inner) {
        self.0.push(inner);
    }
}

impl From<Vec<Inner>> for TensorIndex {
    fn from(v: Vec<Inner>) -> Self {
        Self(v)
    }
}

impl Deref for TensorIndex {
    type Target = Vec<Inner>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[macro_export]
macro_rules! index {
    (@parse [$($stack:tt)*] $range:expr;$step:expr) => {
            TensorIndex::from(
                vec![
                    $($stack)*
                    index!(@convert $range, $step)
                ]
            )
    };

    (@parse [$($stack:tt)*] $range:expr) => {
            TensorIndex::from(
                vec![
                    $($stack)*
                    index!(@convert $range)
                ]
            )
    };

    (@parse [$stack:tt]* $range:expr, $step:expr,) => {
        index!(@parse [$($stack)*] $range;$step)
    };

    (@parse [$($stack:tt)*] $range:expr,) => {
        index!(@parse [$($stack)*] $range)
    };

    (@parse [$($stack:tt)*] $range:expr;$step:expr, $($t:tt)*) => {
                index!(@parse [$($stack)* $index!(@convert $range $step)] $($t)*)
    };

    (@parse [$($stack:tt)*] $range:expr, $($t:tt)*) => {
                index!(@parse [$($stack)* index!(@convert $range), ] $($t)*)
    };

    (@parse [] ) => {
        TensorIndex::from(vec![])
    };

    (@parse $($t:tt)*) => {
        compile_error!("oppai")
    };

    (@convert $range:expr) => {
        $crate::index::Inner::from($range)
    };

    (@convert $range:expr, $step:expr) => {
        $crate::index::Inner::from($range).set_step($step)
    };

    ($($t:tt)*) => {
        index!(@parse [] $($t)*)
    };
}

#[test]
fn index_test_null() {
    let a = index![];
    assert_eq!(a, TensorIndex::from(vec![]));
}

#[test]
fn index_test_int() {
    let index = index![1];
    assert_eq!(index, TensorIndex::from(vec![Inner::new(1, None, 1)]));
}
#[test]
fn index_test_1d() {
    let inner = index![1..2];
    assert_eq!(inner, TensorIndex::from(vec![Inner::new(1, Some(1), 1)]));
}

#[test]
fn index_test_1d_eq() {
    let index = index![1..=4];
    assert_eq!(index, TensorIndex::from(vec![Inner::new(1, Some(4), 1)]));
}

#[test]
fn index_test_full() {
    let index = index![1..10;2];
    assert_eq!(index, TensorIndex::from(vec![Inner::new(1, Some(9), 2)]))
}

#[test]
fn index_test_2d() {
    let index = index![1..2, ..3];
    let v = vec![Inner::new(1, Some(1), 1), Inner::new(0, Some(2), 1)];
    let ans = TensorIndex::from(v);
    assert_eq!(ans, index);
}

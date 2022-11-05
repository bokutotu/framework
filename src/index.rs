use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inner {
    start: isize,
    end: Option<isize>,
    step: isize,
}

impl Inner {
    fn new(start: isize, end: Option<isize>, step: isize) -> Self {
        Self { start, end, step }
    }

    fn set_step(self, step: isize) -> Self {
        let mut s = self;
        s.step = step;
        s
    }
}

impl From<Range<isize>> for Inner {
    fn from(range: Range<isize>) -> Self {
        Inner::new(range.start, Some(range.end), 1)
    }
}

impl From<RangeFull> for Inner {
    fn from(_: RangeFull) -> Self {
        Inner::new(0, Some(-1), 1)
    }
}

impl From<RangeTo<isize>> for Inner {
    fn from(range: RangeTo<isize>) -> Self {
        Inner::new(0, Some(range.end), 1)
    }
}

impl From<RangeFrom<isize>> for Inner {
    fn from(range: RangeFrom<isize>) -> Self {
        Inner::new(range.start, None, 1)
    }
}

impl From<RangeInclusive<isize>> for Inner {
    fn from(range: RangeInclusive<isize>) -> Inner {
        Inner::new(*range.start(), Some(*range.end() + 1), 1)
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
    fn push(&mut self, inner: Inner) {
        self.0.push(inner);
    }
}

impl From<Vec<Inner>> for TensorIndex {
    fn from(v: Vec<Inner>) -> Self {
        Self(v)
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
        Inner::from($range)
    };

    (@convert $range:expr, $step:expr) => {
        Inner::from($range).set_step($step)
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
}
#[test]
fn index_test_1d() {
    let inner = index![1..2];
    assert_eq!(inner, TensorIndex::from(vec![Inner::new(1, Some(2), 1)]));
}

#[test]
fn index_test_full() {
    let index = index![1..10;2];
    assert_eq!(index, TensorIndex::from(vec![Inner::new(1, Some(10), 2)]))
}

#[test]
fn index_test_2d() {
    let index = index![1..2, ..3];
    let v = vec![Inner::new(1, Some(2), 1), Inner::new(0, Some(3), 1)];
    let ans = TensorIndex::from(v);
    assert_eq!(ans, index);
}

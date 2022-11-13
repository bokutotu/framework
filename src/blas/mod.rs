use std::convert::From;

use cblas::{Layout, Transpose};

pub(crate) mod cpu;

#[repr(C)]
pub enum CpuLayout {
    RowMajor,
    ColumnMajor,
}

pub enum CpuTranspose {
    None,
    Ordinary,
    Conjugate,
}

impl From<CpuLayout> for Layout {
    #[inline]
    fn from(item: CpuLayout) -> Layout {
        match item {
            CpuLayout::RowMajor => Layout::RowMajor,
            CpuLayout::ColumnMajor => Layout::ColumnMajor,
        }
    }
}

impl From<CpuTranspose> for Transpose {
    #[inline]
    fn from(item: CpuTranspose) -> Transpose {
        match item {
            CpuTranspose::None => Transpose::None,
            CpuTranspose::Ordinary => Transpose::Ordinary,
            CpuTranspose::Conjugate => Transpose::Conjugate,
        }
    }
}

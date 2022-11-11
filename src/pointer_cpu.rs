use std::ptr::NonNull;

use crate::pointer_traits::{Mut, Owned, TensorPointer, View, ViewMut, ToSlice};

macro_rules! impl_view {
    ( $name:ident, $view:ident, $owned: ident, $lt:tt ) => {
        impl<$lt: Copy> View for $name<$lt> {
            type AccessOutput = $view<$lt>;
            type OwnedOutput = $owned<$lt>;
            fn access_by_offset_region(&self, offset: usize, region: usize) -> Self::AccessOutput {
                if self.is_inbound((offset + region) as isize) {
                    let offset = self.offset + offset;
                    let len = offset + region;
                    let cap = self.cap;
                    let ptr = self.ptr.clone();
                    $view::from_nonnull(ptr, offset, len, cap)
                } else {
                    panic!("aaa");
                }
            }

            fn to_owned(&self) -> Self::OwnedOutput {
                let v = self.to_vec();
                Self::OwnedOutput::from_vec(v)
            }
        }
    };
}

macro_rules! impl_mut {
    ( $name:ident, $lt:tt ) => {
        impl<$lt: Copy> Mut for $name<$lt> {
            fn assign_region<P>(&mut self, other: &P, offset: usize, region: usize)
            where
                P: TensorPointer<Elem = <Self as TensorPointer>::Elem>,
            {
                if !self.is_inbound((offset + region - 1) as isize) {
                    panic!("this is out of bound");
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        other.as_ptr(),
                        self.as_ptr().add(offset) as *mut E,
                        region,
                    )
                }
            }
        }
    };
}

macro_rules! impl_cpu {
    ( $name:ident, $lt:tt) => {
        impl<$lt: Copy> ToSlice for $name<$lt> {
            fn to_slice<'a>(&'a self) -> &'a [<Self as TensorPointer>::Elem] {
                unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) } 
            }
        }
    };
}

#[repr(C)]
pub struct OwnedCpu<E> {
    ptr: NonNull<E>,
    len: usize,
    cap: usize,
}

impl<E> OwnedCpu<E> {
    fn from_vec(vec: Vec<E>) -> Self {
        let mut vec = vec;
        let (ptr, len, cap) = (
            NonNull::new(vec.as_mut_ptr()).expect("Failed to get Pointer for Vec"),
            vec.len(),
            vec.capacity(),
        );
        std::mem::forget(vec);
        Self { ptr, len, cap }
    }
}

impl<E: Copy> TensorPointer for OwnedCpu<E> {
    type Elem = E;
    fn is_inbound(&self, offset: isize) -> bool {
        self.len > offset as usize
    }

    #[allow(clippy::redundant_clone)]
    fn to_vec(&self) -> Vec<Self::Elem> {
        let v = unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.cap) };
        let res = v.clone();
        std::mem::forget(v);
        res
    }

    fn from_vec(vec: Vec<Self::Elem>) -> Self {
        Self::from_vec(vec)
    }

    fn offset(&self, offset: isize) -> NonNull<Self::Elem> {
        self.is_inbound(offset);
        unsafe { NonNull::new_unchecked(self.ptr.as_ptr().offset(offset)) }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        self.ptr.as_ptr()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn offset_num(&self) -> usize {
        0
    }
}

impl<E: Copy> Owned for OwnedCpu<E> {
    type View = ViewCpu<E>;
    type ViewMut = ViewMutCpu<E>;
    fn to_view(&self, offset: usize) -> ViewCpu<E> {
        if self.is_inbound(offset.try_into().unwrap()) {
            ViewCpu::from_nonnull(self.ptr, offset, self.len, self.cap)
        } else {
            panic!("cannot create view of tensor");
        }
    }

    fn to_view_mut(&mut self, offset: usize) -> ViewMutCpu<E> {
        if self.is_inbound(offset.try_into().unwrap()) {
            ViewMutCpu::from_nonnull(self.ptr, offset, self.len, self.cap)
        } else {
            panic!("cannot create view of tensor");
        }
    }
}

impl_mut!(OwnedCpu, E);
impl_cpu!(OwnedCpu, E);

impl<E: Copy> Clone for OwnedCpu<E> {
    fn clone(&self) -> Self {
        let v = self.to_vec();
        Self::from_vec(v)
    }
}

impl<E> Drop for OwnedCpu<E> {
    fn drop(&mut self) {
        let _ = unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.cap) };
    }
}

impl<E: Copy> OwnedCpu<E> {
    pub fn to_slice_mut<'a>(&'a mut self) -> &'a mut  [<Self as TensorPointer>::Elem] {
        unsafe { std::slice::from_raw_parts_mut(self.as_ptr().cast_mut(), self.len) }
    }
}

#[repr(C)]
pub struct ViewCpu<E> {
    ptr: NonNull<E>,
    offset: usize,
    len: usize,
    cap: usize,
}

impl<E> ViewCpu<E> {
    fn from_nonnull(ptr: NonNull<E>, offset: usize, len: usize, cap: usize) -> ViewCpu<E> {
        if offset >= len {
            panic!("must offset < len");
        }
        Self {
            ptr,
            offset,
            len,
            cap,
        }
    }
}

impl<E: Copy> TensorPointer for ViewCpu<E> {
    type Elem = E;

    fn is_inbound(&self, offset: isize) -> bool {
        self.len() > offset.try_into().unwrap()
    }

    #[allow(clippy::redundant_clone)]
    fn to_vec(&self) -> Vec<Self::Elem> {
        let v = unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.cap) };
        let res = v.clone();
        std::mem::forget(v);
        res
    }

    fn from_vec(vec: Vec<Self::Elem>) -> Self {
        let owned_cpu = OwnedCpu::from_vec(vec);
        owned_cpu.to_view(0)
    }

    fn offset(&self, offset: isize) -> NonNull<Self::Elem> {
        if !self.is_inbound(offset) {
            panic!("offset is out of bound");
        }
        unsafe { NonNull::new(self.ptr.as_ptr().offset(offset) as *mut Self::Elem).unwrap() }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        unsafe { self.ptr.as_ptr().add(self.offset) }
    }

    fn len(&self) -> usize {
        self.len - self.offset
    }

    fn offset_num(&self) -> usize {
        self.offset
    }
}

impl_view!(ViewCpu, ViewCpu, OwnedCpu, E);

impl_cpu!(ViewCpu, E);

#[repr(C)]
pub struct ViewMutCpu<E> {
    ptr: NonNull<E>,
    offset: usize,
    len: usize,
    cap: usize,
}

impl<E> ViewMutCpu<E> {
    fn from_nonnull(ptr: NonNull<E>, offset: usize, len: usize, cap: usize) -> ViewMutCpu<E> {
        if offset >= len {
            panic!("must offset < len");
        }
        Self {
            ptr,
            offset,
            len,
            cap,
        }
    }
}

impl<E: Copy> TensorPointer for ViewMutCpu<E> {
    type Elem = E;
    fn is_inbound(&self, offset: isize) -> bool {
        self.len() > offset.try_into().unwrap()
    }

    #[allow(clippy::redundant_clone)]
    fn to_vec(&self) -> Vec<Self::Elem> {
        let v = unsafe { Vec::from_raw_parts(self.ptr.as_ptr(), self.len, self.cap) };
        let res = v.clone();
        std::mem::forget(v);
        res
    }

    fn from_vec(vec: Vec<Self::Elem>) -> Self {
        let mut owned_cpu = OwnedCpu::from_vec(vec);
        owned_cpu.to_view_mut(0)
    }

    fn offset(&self, offset: isize) -> NonNull<Self::Elem> {
        if !self.is_inbound(offset) {
            panic!("offset is out of bound");
        }
        unsafe { NonNull::new(self.ptr.as_ptr().offset(offset) as *mut Self::Elem).unwrap() }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        unsafe { self.ptr.as_ptr().add(self.offset) }
    }

    fn len(&self) -> usize {
        self.len - self.offset
    }

    fn offset_num(&self) -> usize {
        self.offset
    }
}

impl_view!(ViewMutCpu, ViewCpu, OwnedCpu, E);

impl_mut!(ViewMutCpu, E);
impl_cpu!(ViewMutCpu, E);

impl<E: Copy> ViewMut for ViewMutCpu<E> {}

impl<E: Copy> ViewMutCpu<E> {
    pub fn to_slice_mut<'a>(&'a self) -> &'a mut [<Self as TensorPointer>::Elem] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

#[test]
fn owned_cpu_drop_test() {
    let mut _v = OwnedCpu::from_vec(vec![0, 1, 3]);
}

#[test]
fn from_vec_to_vec() {
    let a = vec![0, 1, 2, 3, 4, 5];
    let owned_cpu = OwnedCpu::from_vec(a.clone());
    let v = owned_cpu.to_vec();
    assert_eq!(a, v);
}

#[test]
fn assign_region_test() {
    let mut pointer = OwnedCpu::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let other = OwnedCpu::from_vec(vec![10, 20, 30]);
    pointer.assign_region(&other, 1, 3);
    let vec = pointer.to_vec();
    assert_eq!(vec![0, 10, 20, 30, 4, 5, 6, 7, 8, 9, 10], vec);
}

#[should_panic]
#[test]
fn assign_region_test_shoult_panic() {
    let mut pointer = OwnedCpu::from_vec(vec![0, 1, 3]);
    let other = OwnedCpu::from_vec(vec![1, 3, 3]);
    pointer.assign_region(&other, 1, 3);
}

#[should_panic]
#[test]
fn assign_region_test_shoult_panic_1() {
    let mut pointer = OwnedCpu::from_vec(vec![0, 1, 3]);
    let other = OwnedCpu::from_vec(vec![1, 2]);
    pointer.assign_region(&other, 3, 2);
}

#[test]
fn assign_region_test_mut() {
    let mut pointer = OwnedCpu::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let mut view_mut = pointer.to_view_mut(0);
    let other = OwnedCpu::from_vec(vec![10, 20, 30]);
    view_mut.assign_region(&other, 1, 3);
    let vec = pointer.to_vec();
    assert_eq!(vec![0, 10, 20, 30, 4, 5, 6, 7, 8, 9, 10], vec);
}

#[test]
fn assign_region_test_mut_offset() {
    let mut pointer = OwnedCpu::from_vec(vec![0, 1, 2, 3, 4, 5, 6]);
    let mut pointer = pointer.to_view_mut(1);
    let other = OwnedCpu::from_vec(vec![10, 20]);
    pointer.assign_region(&other, 0, 2);
    let v = pointer.to_vec();
    assert_eq!(vec![0, 10, 20, 3, 4, 5, 6], v);
}

#[test]
fn assign_region_test_view_mut_view_both_offset() {
    let mut pointer = OwnedCpu::from_vec(vec![0, 1, 2, 3, 4, 5, 6]);
    let mut pointer = pointer.to_view_mut(1);
    let other = OwnedCpu::from_vec(vec![10, 20, 30, 40]);
    let other = other.to_view(1);
    pointer.assign_region(&other, 1, 3);
    let v = pointer.to_vec();
    assert_eq!(v, vec![0, 1, 20, 30, 40, 5, 6]);
}

#[should_panic]
#[test]
fn assign_region_test_shoult_panic_mut() {
    let mut pointer = OwnedCpu::from_vec(vec![0, 1, 3]);
    let mut pointer = pointer.to_view_mut(0);
    let other = OwnedCpu::from_vec(vec![1, 3, 3]);
    pointer.assign_region(&other, 1, 3);
}

#[should_panic]
#[test]
fn assign_region_test_shoult_panic_1_mut() {
    let mut pointer = OwnedCpu::from_vec(vec![0, 1, 3]);
    let mut pointer = pointer.to_view_mut(1);
    let other = OwnedCpu::from_vec(vec![1, 2]);
    pointer.assign_region(&other, 3, 2);
}

#[test]
fn owned_cpu_to_slice() {
    let pointer = OwnedCpu::from_vec(vec![0, 1, 2]);
    let s = pointer.to_slice();
    assert_eq!(s, &[0, 1, 2]);
}

#[test]
fn view_cpu_to_slice() {
    let pointer = OwnedCpu::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let pointer = pointer.to_view(0);
    let s = pointer.to_slice();
    assert_eq!(s, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn view_cpu_to_slice_with_offset() {
    let pointer = OwnedCpu::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let pointer = pointer.to_view(1);
    let s = pointer.to_slice();
    assert_eq!(s, &[1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn view_mut_cpu_to_slice() {
    let mut pointer = OwnedCpu::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let pointer = pointer.to_view_mut(0);
    let s = pointer.to_slice();
    assert_eq!(s, &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
}

#[test]
fn view_mut_cpu_to_slice_with_offset() {
    let mut pointer = OwnedCpu::from_vec(vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
    let pointer = pointer.to_view_mut(1);
    let s = pointer.to_slice();
    assert_eq!(s, &[1, 2, 3, 4, 5, 6, 7, 8]);
}

use std::{convert::TryInto, mem::ManuallyDrop, ptr::NonNull};

use crate::pointer_traits::{Mut, Owned, TensorPointer, View, ViewMut};

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
            fn assign_region<P>(&mut self, other: P, offset: usize, region: usize)
            where
                P: TensorPointer<Elem = <Self as TensorPointer>::Elem>,
            {
                if !self.is_inbound((offset + region) as isize) {
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

pub struct OwnedCpu<E> {
    ptr: NonNull<E>,
    len: usize,
    cap: usize,
}

impl<E> OwnedCpu<E> {
    fn from_vec(vec: Vec<E>) -> Self {
        let mut vec = ManuallyDrop::new(vec);
        let (ptr, len, cap) = (
            NonNull::new(vec.as_mut_ptr()).expect("Failed to get Pointer for Vec"),
            vec.len(),
            vec.capacity(),
        );
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
        let vec = ManuallyDrop::new(v);
        ManuallyDrop::into_inner(vec).clone()
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
        let v_manually_drop = ManuallyDrop::into_inner(ManuallyDrop::new(v));
        v_manually_drop.clone()
    }

    fn from_vec(vec: Vec<Self::Elem>) -> Self {
        let owned_cpu = OwnedCpu::from_vec(vec);
        owned_cpu.to_view(0)
    }

    fn offset(&self, offset: isize) -> NonNull<Self::Elem> {
        if !self.is_inbound(offset) {
            panic!("offset is out of bound");
        }
        unsafe { NonNull::new(self.as_ptr().offset(offset) as *mut Self::Elem).unwrap() }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        self.ptr.as_ptr()
    }

    fn len(&self) -> usize {
        self.len - self.offset
    }
}

impl_view!(ViewCpu, ViewCpu, OwnedCpu, E);

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
        let v_manually_drop = ManuallyDrop::into_inner(ManuallyDrop::new(v));
        v_manually_drop.clone()
    }

    fn from_vec(vec: Vec<Self::Elem>) -> Self {
        let mut owned_cpu = OwnedCpu::from_vec(vec);
        owned_cpu.to_view_mut(0)
    }

    fn offset(&self, offset: isize) -> NonNull<Self::Elem> {
        if !self.is_inbound(offset) {
            panic!("offset is out of bound");
        }
        unsafe { NonNull::new(self.as_ptr().offset(offset) as *mut Self::Elem).unwrap() }
    }

    fn as_ptr(&self) -> *const Self::Elem {
        self.ptr.as_ptr()
    }

    fn len(&self) -> usize {
        self.len - self.offset
    }
}

impl_view!(ViewMutCpu, ViewCpu, OwnedCpu, E);

impl_mut!(ViewMutCpu, E);

impl<E: Copy> ViewMut for ViewMutCpu<E> {}

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
    let pointer_ = OwnedCpu::from_vec(vec![10, 20, 30]);
    pointer.assign_region(pointer_, 1, 3);
    let vec = pointer.to_vec();
    assert_eq!(vec![0, 10, 20, 30, 4, 5, 6, 7, 8, 9, 10], vec);
}
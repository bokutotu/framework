use num_traits::Num;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::ops::Drop;

pub(crate) trait Pointer<T> {
    fn to_vec(&self) -> Vec<T>;
    fn into_vec(self) -> Vec<T>;
    fn access_by_offset(&self) -> T;
}

pub(crate) trait MutPointer<T>: Pointer<T> {
    fn assign(&self);
    fn assign_to_offset(&self, offset: isize, other: T);
}

enum LocationPtr<T> {
    Cpu(NonNull<T>),
    Gpu(NonNull<T>)
}

impl<T: Copy> LocationPtr<T> {
    fn new_cpu(ptr: NonNull<T>) -> Self {
        Self::Cpu(ptr)
    }

    fn new_gpu(ptr: NonNull<T>) -> Self {
        Self::Gpu(ptr)
    }

    fn access_by_offset_mut_ptr(&self, offset: isize) -> *mut T {
        match self {
            LocationPtr::Cpu(ptr) => unsafe {
                return ptr.as_ptr().offset(offset)
            },
            LocationPtr::Gpu(ptr) => todo!(),
        }
    }

    fn access_by_offset(&self, offset: isize) -> T {
        unsafe {
            *self.access_by_offset_mut_ptr(offset)
        }
    }
}

pub(crate) struct OwnedPointer<T> {
    ptr: LocationPtr<T>,
    size: usize,
    mut_ref: Arc<Mutex<usize>>,
    ref_num: Arc<Mutex<usize>>,
}

impl<T: Num+Copy> OwnedPointer<T>  {
    fn cpu_alloc(size: usize) -> Self {
        let mut vec = Vec::with_capacity(size);
        for _ in 0..size {
            vec.push(T::zero());
        }
        let ptr = LocationPtr::new_cpu(
            NonNull::new(vec.as_mut_ptr()).expect("allocation failed"));
        std::mem::forget(vec);
        Self { ptr, size, mut_ref: Arc::new(Mutex::new(0)), ref_num: Arc::new(Mutex::new(0)) }
    }

    fn gpu_alloc(size: usize) -> Self {
        todo!();
    }

    pub(crate) fn alloc(size: usize, is_gpu: bool) -> Self {
        if is_gpu {
            return Self::gpu_alloc(size)
        }
        Self::cpu_alloc(size)
    }

    fn access_by_offset(&self, offset: isize) -> T {
        self.ptr.access_by_offset(offset)
    }

    fn access_by_offset_mut_ptr(&self, offset: isize) -> *mut T {
        self.ptr.access_by_offset_mut_ptr(offset)
    }

    fn is_mut_ref_able(&self) -> bool {
        true
    }
}

impl<T> Pointer<T> for OwnedPointer<T> {
    fn to_vec(&self) -> Vec<T> {
        todo!();
    }

    fn into_vec(self) -> Vec<T> {
        todo!();
    }

    fn access_by_offset(&self) -> T {
        todo!();
    }
}

impl<T> MutPointer<T> for OwnedPointer<T> {
    fn assign(&self) {
        todo!();
    }

    fn assign_to_offset(&self, offset: isize, other: T) {
        todo!()
    }
}

pub(crate) struct ViewPointer<T> {
    view: Arc<LocationPtr<T>>,
    offset: isize
}

impl<T> Pointer<T> for ViewPointer<T> {
    fn to_vec(&self) -> Vec<T> {
        todo!();
    }

    fn into_vec(self) -> Vec<T> {
        todo!();
    }

    fn access_by_offset(&self) -> T {
        todo!()
    }
}

pub(crate) struct ViewMutPointer<T> {
    view: Arc<LocationPtr<T>>,
    offset: isize,
}

impl<T> Pointer<T> for ViewMutPointer<T> {
    fn to_vec(&self) -> Vec<T> {
        todo!();
    }

    fn into_vec(self) -> Vec<T> {
        todo!();
    }

    fn access_by_offset(&self) -> T {
        todo!();
    }
}
impl<T> MutPointer<T> for ViewMutPointer<T> {
    fn assign(&self) {
        todo!();
    }

    fn assign_to_offset(&self, offset: isize, other: T) {
        todo!();
    }
}

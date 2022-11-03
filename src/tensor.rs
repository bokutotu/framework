// use std::clone::Clone;
// use std::ops::Index;
// use std::ptr::NonNull;

use crate::pointer_traits::TensorPointer;
use crate::shape::{Shape, Stride, TensorIndex};

pub struct TensorBase<P, E>
where
    P: TensorPointer<Elem = E>,
{
    pointer: P,
    shape: Shape,
    stride: Stride,
    num_elm: usize,
}

// impl<T: Copy + Clone> Tensor<T> {
//     fn new(pointer: *mut T, shape: Shape, stride: Stride) -> Self {
//         let num_elm = shape.num_elms();
//         Self {
//             pointer,
//             shape,
//             stride,
//             num_elm,
//         }
//     }
//
//     fn shape(&self) -> Shape {
//         self.shape.clone()
//     }
//
//     fn stride(&self) -> Stride {
//         self.stride.clone()
//     }
// }
//
// // pub(crate) struct CpuInner<T>(OwnedInner<T>);
// // pub(crate) struct GpuInner<T>(OwnedInner<T>);
//
// impl<T: Copy + Clone> Clone for CpuInner<T> {
//     fn clone(&self) -> Self {
//         let cloned = CpuInner::cpu_malloc(self.shape(), self.shape().default_stride());
//         for index in self.shape().iter() {
//             unsafe {
//                 let cloned_ptr: *mut T = cloned.access_by_idx(&index);
//                 let self_ptr: *mut T = self.access_by_idx(&index);
//                 cloned_ptr.write(*self_ptr);
//             }
//         }
//         cloned
//     }
// }
//
// impl<T: Copy + Clone> Clone for GpuInner<T> {
//     fn clone(&self) -> Self {
//         todo!()
//     }
// }
//
// impl<T> Drop for CpuInner<T> {
//     fn drop(&mut self) {
//         let vec = unsafe { Vec::from_raw_parts(self.0.pointer, self.0.num_elm, self.0.num_elm) };
//         drop(vec);
//     }
// }
//
// impl<T: Copy + Clone> CpuInner<T> {
//     fn shape(&self) -> Shape {
//         self.0.shape()
//     }
//
//     fn stride(&self) -> Stride {
//         self.0.stride()
//     }
//
//     fn is_default_stride(&self) -> bool {
//         self.shape().is_default_stride(&self.stride())
//     }
//
//     fn cpu_malloc(shape: Shape, stride: Stride) -> Self {
//         let alloc_vec: Vec<T> = Vec::with_capacity(shape.num_elms());
//         let pointer: *mut T = alloc_vec.as_ptr() as *mut T;
//         std::mem::forget(alloc_vec);
//         CpuInner(Tensor::new(pointer, shape, stride))
//     }
//
//     fn cal_offset(&self, index: &TensorIndex) -> isize {
//         self.shape().valid_index(index);
//         self.stride().cal_offset(index)
//     }
//
//     unsafe fn access_by_offset(&self, offset: isize) -> *mut T {
//         self.0.pointer.offset(offset) as *mut T
//     }
//
//     unsafe fn access_by_idx(&self, index: &TensorIndex) -> *mut T {
//         let offset = self.cal_offset(index);
//         self.access_by_offset(offset)
//     }
//
//     fn to_vec(&self) -> Vec<T> {
//         let cloned = self.clone();
//         let vec =
//             unsafe { Vec::from_raw_parts(cloned.0.pointer, cloned.0.num_elm, cloned.0.num_elm) };
//         std::mem::forget(cloned);
//         vec
//     }
//
//     fn from_vec(vec: Vec<T>, shape: Shape) -> Self {
//         let stride = shape.default_stride();
//         let inner = Tensor::new(vec.as_ptr() as *mut T, shape, stride);
//         std::mem::forget(vec);
//         Self(inner)
//     }
// }
//
// impl<T> GpuInner<T> {
//     fn shape(&self) -> Shape {
//         self.0.shape.clone()
//     }
//
//     fn stride(&self) -> Stride {
//         self.0.stride.clone()
//     }
//
//     fn is_default_stride(&self) -> bool {
//         self.shape().is_default_stride(&self.stride())
//     }
//
//     fn gpu_malloc(shape: Shape, stride: Stride) -> Self {
//         let _shape = shape;
//         let _stride = stride;
//         todo!()
//     }
//
//     fn cal_offset(self, index: TensorIndex) -> isize {
//         self.shape().valid_index(&index);
//         self.stride().cal_offset(&index)
//     }
//
//     // GPUのポインタをそのまま返すわけには行かないので対策を後で考える
//     pub unsafe fn access_by_idx(&self, index: &TensorIndex) -> *mut T {
//         let _index = index;
//         todo!();
//     }
//
//     fn to_vec(&self) -> Vec<T> {
//         todo!();
//     }
//
//     fn into_vec(self) -> Vec<T> {
//         todo!();
//     }
//
//     fn from_vec(vec: Vec<T>, shape: Shape) -> Self {
//         let _vec = vec;
//         let _shape = shape;
//         todo!();
//     }
// }
//
// impl<T> Drop for GpuInner<T> {
//     fn drop(&mut self) {
//         todo!();
//     }
// }

// pub(crate) enum Pointer<T> {
//     Gpu(GpuInner<T>),
//     Cpu(CpuInner<T>),
// }
//
// impl<T: Copy + Clone> Pointer<T> {
//     fn shape(&self) -> Shape {
//         match self {
//             Pointer::Gpu(inner) => inner.shape(),
//             Pointer::Cpu(inner) => inner.shape(),
//         }
//     }
//
//     fn is_default_stride(&self) -> bool {
//         match self {
//             Pointer::Gpu(inner) => inner.is_default_stride(),
//             Pointer::Cpu(inner) => inner.is_default_stride(),
//         }
//     }
//
//     fn stride(&self) -> Stride {
//         match self {
//             Pointer::Gpu(inner) => inner.stride(),
//             Pointer::Cpu(inner) => inner.stride(),
//         }
//     }
//
//     fn cpu_malloc(shape: Shape, stride: Stride) -> Self {
//         let pointer = CpuInner::cpu_malloc(shape, stride);
//         Pointer::Cpu(pointer)
//     }
//
//     fn gpu_malloc(shape: Shape, stride: Stride) -> Self {
//         let pointer = GpuInner::gpu_malloc(shape, stride);
//         Pointer::Gpu(pointer)
//     }
//
//     // fn cal_offset(self, index: TensorIndex) -> isize {
//     //     self.shape().valid_index(&index);
//     //     self.stride().cal_offset(&index)
//     // }
//
//     fn to_vec(&self) -> Vec<T> {
//         match self {
//             Pointer::Gpu(inner) => inner.to_vec(),
//             Pointer::Cpu(inner) => inner.to_vec(),
//         }
//     }
//
//     fn from_vec(vec: Vec<T>, shape: Shape, is_gpu: bool) -> Self {
//         if is_gpu {
//             return Pointer::Gpu(GpuInner::from_vec(vec, shape));
//         }
//         Pointer::Cpu(CpuInner::from_vec(vec, shape))
//     }
//
//     unsafe fn access_by_index(&self, index: &TensorIndex) -> *mut T {
//         match self {
//             Pointer::Cpu(inner) => inner.access_by_idx(index),
//             Pointer::Gpu(inner) => inner.access_by_idx(index),
//         }
//     }
// }
//
// impl<T> Drop for Pointer<T> {
//     fn drop(&mut self) {
//         match self {
//             Pointer::Gpu(inner) => drop(inner),
//             Pointer::Cpu(inner) => drop(inner),
//         };
//     }
// }

// pub struct Tensor<T> {
//     inner: Pointer<T>,
// }
//
// impl<T: Copy + Clone> Tensor<T> {
//     pub fn shape(&self) -> Shape {
//         self.inner.shape()
//     }
//
//     pub fn stride(&self) -> Stride {
//         self.inner.stride()
//     }
//
//     pub fn cpu_malloc_from_shape(shape: Vec<isize>) -> Self {
//         let shape = Shape::new(shape);
//         let stride = shape.default_stride();
//         let pointer = Pointer::cpu_malloc(shape, stride);
//         Tensor { inner: pointer }
//     }
//
//     pub fn gpu_malloc_from_shape(shape: Vec<isize>) -> Self {
//         let shape = Shape::new(shape);
//         let stride = shape.default_stride();
//         let pointer = Pointer::gpu_malloc(shape, stride);
//         Tensor { inner: pointer }
//     }
//
//     pub fn to_vec(&self) -> Vec<T> {
//         self.inner.to_vec()
//     }
//
//     pub fn into_vec(self) -> Vec<T> {
//         self.to_vec()
//     }
//
//     pub fn from_vec(vec: Vec<T>, shape: Shape, is_gpu: bool) -> Self {
//         if vec.len() != shape.num_elms() {
//             panic!("length of vectoro and shape is not collect");
//         }
//         Self {
//             inner: Pointer::from_vec(vec, shape, is_gpu),
//         }
//     }
//
//     unsafe fn access_by_index(&self, index: &TensorIndex) -> *mut T {
//         self.inner.access_by_index(index)
//     }
//
//     fn is_default_stride(&self) -> bool {
//         self.inner.is_default_stride()
//     }
// }

// impl<T> Drop for Tensor<T> {
//     fn drop(&mut self) {
//         drop(self.inner);
//     }
// }

// impl<T: Copy + Clone> Index<TensorIndex> for Tensor<T> {
//     type Output = T;
//
//     fn index(&self, index: TensorIndex) -> &Self::Output {
//         unsafe { &*self.access_by_index(&index) }
//     }
// }

// macro_rules! impl_from_to_vec_test {
//     ($fn_name:ident, $from_vec:expr, $shape:expr) => {
//         #[test]
//         fn $fn_name() {
//             let tensor = Tensor::from_vec($from_vec, Shape::new($shape), false);
//             let vec = tensor.to_vec();
//             let into_vec = tensor.into_vec();
//             assert_eq!(vec, $from_vec);
//             assert_eq!(vec, into_vec);
//         }
//     };
// }
//
// impl_from_to_vec_test!(from_to_vec_1d, vec![0, 1, 2, 3, 4], vec![5]);
// impl_from_to_vec_test!(from_to_vec_2d, vec![0, 1, 2, 3, 4, 5], vec![2, 3]);
// impl_from_to_vec_test!(
//     from_to_vec_3d,
//     vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
//     vec![2, 3, 3]
// );

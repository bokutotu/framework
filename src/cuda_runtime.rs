use std::ops::Deref;
use std::os::raw;
use std::ptr::null_mut;
use std::result::Result;

use cuda_runtime_sys::*;

use thiserror::Error;

#[derive(Error, Debug, Copy, Clone)]
pub enum CudaError {
    #[error("This indicates that one or more of the parameters passed to the API call is not within an acceptable range of values.")]
    CudaErrorInvalidValue,
    #[error("The API call failed because it was unable to allocate enough memory to perform the requested operation.")]
    CudaErrorMemoryAllocation,
}

#[derive(Debug, Copy, Clone)]
struct CudaResult(Result<(), CudaError>);

impl Deref for CudaResult {
    type Target = Result<(), CudaError>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<cuda_runtime_sys::cudaError_t> for CudaResult {
    fn from(error: cuda_runtime_sys::cudaError_t) -> Self {
        match error {
            cudaError_t::cudaErrorInvalidValue => CudaResult(Err(CudaError::CudaErrorInvalidValue)),
            cudaError_t::cudaErrorMemoryAllocation => {
                CudaResult(Err(CudaError::CudaErrorMemoryAllocation))
            }
            cudaError::cudaSuccess => CudaResult(Ok(())),
            _ => unreachable!(),
        }
    }
}

pub fn malloc<T>(size: usize) -> Result<(), CudaError> {
    let mut ptr: *mut T = null_mut();
    let result: CudaResult =
        unsafe { cudaMalloc(&mut ptr as *mut *mut T as *mut *mut raw::c_void, size).into() };
    *result
}

fn memcpy<T>(
    dst: *mut T,
    src: *const T,
    size: usize,
    kind: cudaMemcpyKind,
) -> Result<(), CudaError> {
    let cuda_result: CudaResult = unsafe {
        cuda_runtime_sys::cudaMemcpy(dst as *mut raw::c_void, src as *mut raw::c_void, size, kind)
            .into()
    };
    *cuda_result
}

fn memcpy_host_to_device<T>(dst: *mut T, src: *const T, size: usize) -> Result<(), CudaError> {
    memcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyHostToDevice)
}

fn memcpy_device_to_host<T>(dst: *mut T, src: *const T, size: usize) -> Result<(), CudaError> {
    memcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyDeviceToHost)
}

fn memcpy_device_to_device<T>(dst: *mut T, src: *const T, size: usize) -> Result<(), CudaError> {
    memcpy(dst, src, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice)
}

pub fn free<T>(devptr: *mut T) -> Result<(), CudaError> {
    let cuda_res: CudaResult =
        unsafe { cuda_runtime_sys::cudaFree(devptr as *mut raw::c_void).into() };
    *cuda_res
}

pub fn device_config(dev_id: usize) -> Result<(), CudaError> {
    let cuda_res: CudaResult =
        unsafe { cuda_runtime_sys::cudaSetDevice(dev_id as raw::c_int).into() };
    *cuda_res
}

pub fn device_synchronize() -> Result<(), CudaError> {
    let cuda_res: CudaResult = unsafe { cuda_runtime_sys::cudaDeviceSynchronize().into() };
    *cuda_res
}

pub fn launch(
    func: *const raw::c_void,
    grid_dim: dim3,
    block_dim: dim3,
    args: &mut [*mut raw::c_void],
    shared_mem: usize,
) -> Result<(), CudaError> {
    let cuda_error: CudaResult = unsafe {
        cuda_runtime_sys::cudaLaunchKernel(
            func,
            grid_dim,
            block_dim,
            args.as_mut_ptr(),
            shared_mem,
            null_mut(),
        )
        .into()
    };
    *cuda_error
}

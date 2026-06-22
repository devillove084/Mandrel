use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::backend::{Result, VortexBackendError};
use crate::vortex2::{Device, Kernel, Module};

#[derive(Default)]
pub(crate) struct VortexExecutor {
    kernels: HashMap<KernelCacheKey, Kernel>,
    modules: HashMap<PathBuf, Module>,
}

impl VortexExecutor {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn cached_module_count(&self) -> usize {
        self.modules.len()
    }

    pub(crate) fn cached_kernel_count(&self) -> usize {
        self.kernels.len()
    }

    pub(crate) fn clear_kernel_cache(&mut self) {
        self.kernels.clear();
        self.modules.clear();
    }

    pub(crate) fn ensure_kernel(
        &mut self,
        device: &Device,
        module_path: &Path,
        name: &'static str,
    ) -> Result<KernelLookup> {
        let module_path = module_path.to_path_buf();
        let key = KernelCacheKey {
            module_path: module_path.clone(),
            name: name.to_owned(),
        };
        let module_cache_hit = self.modules.contains_key(&module_path);
        if self.kernels.contains_key(&key) {
            return Ok(KernelLookup {
                key,
                module_cache_hit,
                kernel_cache_hit: true,
            });
        }

        if !module_cache_hit {
            let module = device.load_module_file(&module_path)?;
            self.modules.insert(module_path.clone(), module);
        }

        let kernel = {
            let module = self
                .modules
                .get(&module_path)
                .ok_or(VortexBackendError::Internal("cached module disappeared"))?;
            module.get_kernel(name)?
        };
        self.kernels.insert(key.clone(), kernel);
        Ok(KernelLookup {
            key,
            module_cache_hit,
            kernel_cache_hit: false,
        })
    }

    pub(crate) fn kernel(&self, key: &KernelCacheKey) -> Result<&Kernel> {
        self.kernels
            .get(key)
            .ok_or(VortexBackendError::Internal("cached kernel disappeared"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct KernelCacheKey {
    module_path: PathBuf,
    name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct KernelLookup {
    pub(crate) key: KernelCacheKey,
    pub(crate) module_cache_hit: bool,
    pub(crate) kernel_cache_hit: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VortexLaunchDims {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_memory_bytes: u32,
}

impl VortexLaunchDims {
    pub const fn new(grid: [u32; 3], block: [u32; 3], shared_memory_bytes: u32) -> Self {
        Self {
            grid,
            block,
            shared_memory_bytes,
        }
    }

    pub const fn workgroup_count(self) -> u32 {
        self.grid[0] * self.grid[1] * self.grid[2]
    }

    pub const fn threads_per_workgroup(self) -> u32 {
        self.block[0] * self.block[1] * self.block[2]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VortexLaunchTrace {
    pub kernel_symbol: &'static str,
    pub dims: VortexLaunchDims,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub module_cache_hit: bool,
    pub kernel_cache_hit: bool,
}

impl VortexLaunchTrace {
    pub const fn new(
        kernel_symbol: &'static str,
        dims: VortexLaunchDims,
        host_to_device_bytes: u64,
        device_to_host_bytes: u64,
        module_cache_hit: bool,
        kernel_cache_hit: bool,
    ) -> Self {
        Self {
            kernel_symbol,
            dims,
            host_to_device_bytes,
            device_to_host_bytes,
            module_cache_hit,
            kernel_cache_hit,
        }
    }

    pub const fn total_transfer_bytes(self) -> Option<u64> {
        self.host_to_device_bytes
            .checked_add(self.device_to_host_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::{KernelCacheKey, VortexLaunchDims, VortexLaunchTrace};
    use std::path::PathBuf;

    #[test]
    fn cache_key_includes_module_path_and_symbol() {
        let lhs = KernelCacheKey {
            module_path: PathBuf::from("a/kernel.vxbin"),
            name: "matmul_i8_i32".to_owned(),
        };
        let rhs = KernelCacheKey {
            module_path: PathBuf::from("b/kernel.vxbin"),
            name: "matmul_i8_i32".to_owned(),
        };

        assert_ne!(lhs, rhs);
    }

    #[test]
    fn launch_dims_summarize_work_shape() {
        let dims = VortexLaunchDims::new([8, 8, 1], [4, 4, 1], 0);

        assert_eq!(dims.workgroup_count(), 64);
        assert_eq!(dims.threads_per_workgroup(), 16);
    }

    #[test]
    fn launch_trace_summarizes_transfer_bytes() {
        let dims = VortexLaunchDims::new([8, 8, 1], [4, 4, 1], 0);
        let trace = VortexLaunchTrace::new("matmul_i8_i32", dims, 4096, 4096, false, false);

        assert_eq!(trace.total_transfer_bytes(), Some(8192));
    }
}

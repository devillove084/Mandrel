#![cfg_attr(not(feature = "std"), no_std)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFlavor {
    Reference,
}

pub mod reference {
    pub fn matmul_i8_i32(lhs: &[i8], rhs: &[i8], out: &mut [i32], m: usize, k: usize, n: usize) {
        assert_eq!(lhs.len(), m * k);
        assert_eq!(rhs.len(), k * n);
        assert_eq!(out.len(), m * n);

        for row in 0..m {
            for col in 0..n {
                let mut acc = 0_i32;
                for depth in 0..k {
                    acc += lhs[row * k + depth] as i32 * rhs[depth * n + col] as i32;
                }
                out[row * n + col] = acc;
            }
        }
    }
}

#![cfg_attr(not(feature = "std"), no_std)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFlavor {
    Reference,
}

pub mod reference {
    pub fn row_softmax_f32(input: &[f32], out: &mut [f32], rows: usize, cols: usize) {
        assert_eq!(input.len(), rows * cols);
        assert_eq!(out.len(), rows * cols);

        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let row_input = &input[start..end];
            let row_out = &mut out[start..end];

            let mut max_value = f32::NEG_INFINITY;
            for value in row_input {
                max_value = max_value.max(*value);
            }

            let mut sum = 0.0_f32;
            for (dst, value) in row_out.iter_mut().zip(row_input.iter()) {
                let exp = libm::expf(*value - max_value);
                *dst = exp;
                sum += exp;
            }

            for value in row_out {
                *value /= sum;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::reference;

    #[test]
    fn computes_row_softmax_reference() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let mut output = [0.0_f32; 4];

        reference::row_softmax_f32(&input, &mut output, 1, 4);

        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert!(output[3] > output[2]);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }
}

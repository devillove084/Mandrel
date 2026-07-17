#![cfg_attr(not(feature = "std"), no_std)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFlavor {
    Reference,
}

pub mod reference {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SwigluMlpShape {
        pub tokens: usize,
        pub hidden: usize,
        pub intermediate: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct CausalAttentionShape {
        pub tokens: usize,
        pub query_heads: usize,
        pub key_value_heads: usize,
        pub head_dim: usize,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RotaryEmbeddingShape {
        pub tokens: usize,
        pub heads: usize,
        pub head_dim: usize,
    }

    impl CausalAttentionShape {
        pub const fn new(
            tokens: usize,
            query_heads: usize,
            key_value_heads: usize,
            head_dim: usize,
        ) -> Self {
            Self {
                tokens,
                query_heads,
                key_value_heads,
                head_dim,
            }
        }

        pub const fn query_width(self) -> usize {
            self.query_heads * self.head_dim
        }

        pub const fn key_value_width(self) -> usize {
            self.key_value_heads * self.head_dim
        }
    }

    impl RotaryEmbeddingShape {
        pub const fn new(tokens: usize, heads: usize, head_dim: usize) -> Self {
            Self {
                tokens,
                heads,
                head_dim,
            }
        }

        pub const fn width(self) -> usize {
            self.heads * self.head_dim
        }
    }

    impl SwigluMlpShape {
        pub const fn new(tokens: usize, hidden: usize, intermediate: usize) -> Self {
            Self {
                tokens,
                hidden,
                intermediate,
            }
        }
    }

    pub fn add_f32(lhs: &[f32], rhs: &[f32], out: &mut [f32]) {
        assert_eq!(lhs.len(), rhs.len());
        assert_eq!(out.len(), lhs.len());

        for ((dst, lhs_value), rhs_value) in out.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
            *dst = *lhs_value + *rhs_value;
        }
    }

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

    pub fn embedding_lookup_f32(
        token_ids: &[usize],
        embedding_weight: &[f32],
        out: &mut [f32],
        tokens: usize,
        vocab_size: usize,
        hidden: usize,
    ) {
        assert_eq!(token_ids.len(), tokens);
        assert_eq!(embedding_weight.len(), vocab_size * hidden);
        assert_eq!(out.len(), tokens * hidden);

        for (token, token_id) in token_ids.iter().copied().enumerate() {
            assert!(token_id < vocab_size);
            let source_start = token_id * hidden;
            let source_end = source_start + hidden;
            let output_start = token * hidden;
            let output_end = output_start + hidden;
            out[output_start..output_end]
                .copy_from_slice(&embedding_weight[source_start..source_end]);
        }
    }

    pub fn rms_norm_f32(
        input: &[f32],
        weight: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        eps: f32,
    ) {
        assert_eq!(input.len(), rows * cols);
        assert_eq!(out.len(), rows * cols);
        assert_eq!(weight.len(), cols);

        for row in 0..rows {
            let start = row * cols;
            let end = start + cols;
            let row_input = &input[start..end];
            let row_out = &mut out[start..end];
            let mut sum_squares = 0.0_f32;

            for value in row_input {
                sum_squares += *value * *value;
            }

            let inv_rms = 1.0_f32 / libm::sqrtf(sum_squares / cols as f32 + eps);
            for ((dst, value), scale) in row_out.iter_mut().zip(row_input.iter()).zip(weight.iter())
            {
                *dst = *value * inv_rms * *scale;
            }
        }
    }

    /// Matrix multiply for row-major tensors: lhs `[rows, inner]`, rhs `[inner, cols]`,
    /// out `[rows, cols]`.
    pub fn matmul_f32(
        lhs: &[f32],
        rhs: &[f32],
        out: &mut [f32],
        rows: usize,
        inner: usize,
        cols: usize,
    ) {
        assert_eq!(lhs.len(), rows * inner);
        assert_eq!(rhs.len(), inner * cols);
        assert_eq!(out.len(), rows * cols);

        for row in 0..rows {
            for col in 0..cols {
                let mut acc = 0.0_f32;
                for index in 0..inner {
                    acc += lhs[row * inner + index] * rhs[index * cols + col];
                }
                out[row * cols + col] = acc;
            }
        }
    }

    /// Linear layer for HF-style row-major weights: input `[rows, in_features]`,
    /// weight `[out_features, in_features]`, optional bias `[out_features]`, out
    /// `[rows, out_features]`.
    pub fn linear_f32(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
        rows: usize,
        in_features: usize,
        out_features: usize,
    ) {
        assert_eq!(input.len(), rows * in_features);
        assert_eq!(weight.len(), out_features * in_features);
        assert_eq!(out.len(), rows * out_features);
        if let Some(bias) = bias {
            assert_eq!(bias.len(), out_features);
        }

        for row in 0..rows {
            for out_feature in 0..out_features {
                let mut acc = bias.map_or(0.0_f32, |values| values[out_feature]);
                for in_feature in 0..in_features {
                    acc += input[row * in_features + in_feature]
                        * weight[out_feature * in_features + in_feature];
                }
                out[row * out_features + out_feature] = acc;
            }
        }
    }

    pub fn silu_f32(input: &[f32], out: &mut [f32]) {
        assert_eq!(input.len(), out.len());

        for (dst, value) in out.iter_mut().zip(input.iter()) {
            *dst = silu_scalar(*value);
        }
    }

    pub fn swiglu_f32(gate: &[f32], up: &[f32], out: &mut [f32]) {
        assert_eq!(gate.len(), up.len());
        assert_eq!(out.len(), gate.len());

        for ((dst, gate_value), up_value) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
            *dst = silu_scalar(*gate_value) * *up_value;
        }
    }

    /// SwiGLU MLP used by Mixtral/Qwen-MoE-style experts. The gate/up weights are
    /// `[intermediate, hidden]`; the down weight is `[hidden, intermediate]`.
    /// `scratch` stores the activated intermediate tensor `[tokens, intermediate]`.
    pub fn swiglu_mlp_f32(
        input: &[f32],
        gate_weight: &[f32],
        up_weight: &[f32],
        down_weight: &[f32],
        scratch: &mut [f32],
        out: &mut [f32],
        shape: SwigluMlpShape,
    ) {
        assert_eq!(input.len(), shape.tokens * shape.hidden);
        assert_eq!(gate_weight.len(), shape.intermediate * shape.hidden);
        assert_eq!(up_weight.len(), shape.intermediate * shape.hidden);
        assert_eq!(down_weight.len(), shape.hidden * shape.intermediate);
        assert_eq!(scratch.len(), shape.tokens * shape.intermediate);
        assert_eq!(out.len(), shape.tokens * shape.hidden);

        for token in 0..shape.tokens {
            for inter in 0..shape.intermediate {
                let mut gate_acc = 0.0_f32;
                let mut up_acc = 0.0_f32;
                for dim in 0..shape.hidden {
                    let value = input[token * shape.hidden + dim];
                    gate_acc += value * gate_weight[inter * shape.hidden + dim];
                    up_acc += value * up_weight[inter * shape.hidden + dim];
                }
                scratch[token * shape.intermediate + inter] = silu_scalar(gate_acc) * up_acc;
            }
        }

        for token in 0..shape.tokens {
            for dim in 0..shape.hidden {
                let mut acc = 0.0_f32;
                for inter in 0..shape.intermediate {
                    acc += scratch[token * shape.intermediate + inter]
                        * down_weight[dim * shape.intermediate + inter];
                }
                out[token * shape.hidden + dim] = acc;
            }
        }
    }

    pub fn rotary_embedding_f32(
        input: &[f32],
        positions: &[usize],
        out: &mut [f32],
        shape: RotaryEmbeddingShape,
        theta: f32,
    ) {
        assert_eq!(input.len(), shape.tokens * shape.width());
        assert_eq!(positions.len(), shape.tokens);
        assert_eq!(out.len(), input.len());
        assert!(theta > 0.0);

        if shape.head_dim < 2 {
            out.copy_from_slice(input);
            return;
        }
        assert_eq!(shape.head_dim % 2, 0);

        let half_dim = shape.head_dim / 2;
        for (token, position) in positions.iter().copied().enumerate() {
            let position = position as f32;
            for head in 0..shape.heads {
                let head_start = (token * shape.heads + head) * shape.head_dim;
                for pair in 0..half_dim {
                    let exponent = (2 * pair) as f32 / shape.head_dim as f32;
                    let angle = position / libm::powf(theta, exponent);
                    let sin = libm::sinf(angle);
                    let cos = libm::cosf(angle);
                    let first_index = head_start + pair;
                    let second_index = head_start + half_dim + pair;
                    let first = input[first_index];
                    let second = input[second_index];
                    out[first_index] = first * cos - second * sin;
                    out[second_index] = second * cos + first * sin;
                }
            }
        }
    }

    pub fn causal_self_attention_f32(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        score_scratch: &mut [f32],
        out: &mut [f32],
        shape: CausalAttentionShape,
    ) {
        assert!(shape.query_heads >= shape.key_value_heads);
        assert_eq!(shape.query_heads % shape.key_value_heads, 0);
        assert_eq!(query.len(), shape.tokens * shape.query_width());
        assert_eq!(key.len(), shape.tokens * shape.key_value_width());
        assert_eq!(value.len(), shape.tokens * shape.key_value_width());
        assert_eq!(out.len(), query.len());
        assert!(score_scratch.len() >= shape.tokens);

        let scale = 1.0_f32 / libm::sqrtf(shape.head_dim as f32);
        for query_token in 0..shape.tokens {
            for query_head in 0..shape.query_heads {
                let key_value_head = query_head * shape.key_value_heads / shape.query_heads;
                for key_token in 0..=query_token {
                    let mut dot = 0.0_f32;
                    for dim in 0..shape.head_dim {
                        dot += query[query_offset(shape, query_token, query_head, dim)]
                            * key[key_value_offset(shape, key_token, key_value_head, dim)];
                    }
                    score_scratch[key_token] = dot * scale;
                }

                let weights = &mut score_scratch[..=query_token];
                softmax_in_place(weights);
                for dim in 0..shape.head_dim {
                    let mut acc = 0.0_f32;
                    for key_token in 0..=query_token {
                        acc += score_scratch[key_token]
                            * value[key_value_offset(shape, key_token, key_value_head, dim)];
                    }
                    out[query_offset(shape, query_token, query_head, dim)] = acc;
                }
            }
        }
    }

    pub fn row_topk_f32(
        input: &[f32],
        values: &mut [f32],
        indices: &mut [usize],
        rows: usize,
        cols: usize,
        k: usize,
    ) {
        assert!(k <= cols);
        assert_eq!(input.len(), rows * cols);
        assert_eq!(values.len(), rows * k);
        assert_eq!(indices.len(), rows * k);

        for row in 0..rows {
            let value_row = &mut values[row * k..(row + 1) * k];
            let index_row = &mut indices[row * k..(row + 1) * k];
            value_row.fill(f32::NEG_INFINITY);
            index_row.fill(usize::MAX);

            for col in 0..cols {
                insert_topk(input[row * cols + col], col, value_row, index_row);
            }
        }
    }

    /// Router helper: computes logits as `input @ router_weight.T`, keeps top-k experts per
    /// token, then softmaxes over the selected logits. `router_weight` is
    /// `[num_experts, hidden]`; outputs are `[tokens, top_k]`.
    pub fn moe_router_topk_f32(
        input: &[f32],
        router_weight: &[f32],
        routing_weights: &mut [f32],
        selected_experts: &mut [usize],
        tokens: usize,
        hidden: usize,
        num_experts: usize,
        top_k: usize,
    ) {
        assert!(top_k <= num_experts);
        assert_eq!(input.len(), tokens * hidden);
        assert_eq!(router_weight.len(), num_experts * hidden);
        assert_eq!(routing_weights.len(), tokens * top_k);
        assert_eq!(selected_experts.len(), tokens * top_k);

        for token in 0..tokens {
            let weights_row = &mut routing_weights[token * top_k..(token + 1) * top_k];
            let experts_row = &mut selected_experts[token * top_k..(token + 1) * top_k];
            weights_row.fill(f32::NEG_INFINITY);
            experts_row.fill(usize::MAX);

            for expert in 0..num_experts {
                let mut logit = 0.0_f32;
                for dim in 0..hidden {
                    logit += input[token * hidden + dim] * router_weight[expert * hidden + dim];
                }
                insert_topk(logit, expert, weights_row, experts_row);
            }

            softmax_in_place(weights_row);
        }
    }

    /// Softmax over row-local top-k logits. The `values` slice is overwritten with
    /// probabilities; `indices` is left unchanged.
    pub fn row_topk_softmax_f32(
        input: &[f32],
        values: &mut [f32],
        indices: &mut [usize],
        rows: usize,
        cols: usize,
        k: usize,
    ) {
        row_topk_f32(input, values, indices, rows, cols, k);
        for row in 0..rows {
            softmax_in_place(&mut values[row * k..(row + 1) * k]);
        }
    }

    /// Combines selected expert outputs laid out as `[tokens, top_k, hidden]` with
    /// routing weights `[tokens, top_k]` into `out` `[tokens, hidden]`.
    pub fn moe_combine_topk_f32(
        expert_outputs: &[f32],
        routing_weights: &[f32],
        out: &mut [f32],
        tokens: usize,
        top_k: usize,
        hidden: usize,
    ) {
        assert_eq!(expert_outputs.len(), tokens * top_k * hidden);
        assert_eq!(routing_weights.len(), tokens * top_k);
        assert_eq!(out.len(), tokens * hidden);

        for token in 0..tokens {
            for dim in 0..hidden {
                let mut acc = 0.0_f32;
                for slot in 0..top_k {
                    acc += routing_weights[token * top_k + slot]
                        * expert_outputs[(token * top_k + slot) * hidden + dim];
                }
                out[token * hidden + dim] = acc;
            }
        }
    }

    const fn query_offset(
        shape: CausalAttentionShape,
        token: usize,
        head: usize,
        dim: usize,
    ) -> usize {
        (token * shape.query_heads + head) * shape.head_dim + dim
    }

    const fn key_value_offset(
        shape: CausalAttentionShape,
        token: usize,
        head: usize,
        dim: usize,
    ) -> usize {
        (token * shape.key_value_heads + head) * shape.head_dim + dim
    }

    fn insert_topk(
        candidate_value: f32,
        candidate_index: usize,
        values: &mut [f32],
        indices: &mut [usize],
    ) {
        for slot in 0..values.len() {
            if candidate_value > values[slot] {
                for shift in (slot + 1..values.len()).rev() {
                    values[shift] = values[shift - 1];
                    indices[shift] = indices[shift - 1];
                }
                values[slot] = candidate_value;
                indices[slot] = candidate_index;
                break;
            }
        }
    }

    fn softmax_in_place(values: &mut [f32]) {
        if values.is_empty() {
            return;
        }

        let mut max_value = f32::NEG_INFINITY;
        for value in values.iter() {
            max_value = max_value.max(*value);
        }

        let mut sum = 0.0_f32;
        for value in values.iter_mut() {
            let exp = libm::expf(*value - max_value);
            *value = exp;
            sum += exp;
        }

        for value in values.iter_mut() {
            *value /= sum;
        }
    }

    fn silu_scalar(value: f32) -> f32 {
        value / (1.0_f32 + libm::expf(-value))
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

    #[test]
    fn computes_embedding_lookup_reference() {
        let token_ids = [2_usize, 0];
        let embedding_weight = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0_f32; 4];

        reference::embedding_lookup_f32(&token_ids, &embedding_weight, &mut output, 2, 3, 2);

        assert_eq!(output, [5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn computes_rms_norm_reference() {
        let input = [3.0_f32, 4.0];
        let weight = [1.0_f32, 2.0];
        let mut output = [0.0_f32; 2];

        reference::rms_norm_f32(&input, &weight, &mut output, 1, 2, 0.0);

        let inv_rms = 1.0_f32 / ((9.0_f32 + 16.0) / 2.0).sqrt();
        assert!((output[0] - 3.0 * inv_rms).abs() < 1.0e-6);
        assert!((output[1] - 8.0 * inv_rms).abs() < 1.0e-6);
    }

    #[test]
    fn computes_hf_style_linear_reference() {
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let bias = [0.5_f32, -1.0, 2.0];
        let mut output = [0.0_f32; 6];

        reference::linear_f32(&input, &weight, Some(&bias), &mut output, 2, 2, 3);

        assert_eq!(output, [1.5, 1.0, 5.0, 3.5, 3.0, 9.0]);
    }

    #[test]
    fn computes_swiglu_mlp_reference() {
        let input = [1.0_f32, 2.0];
        let gate_weight = [1.0_f32, 0.0, 0.0, 1.0];
        let up_weight = [2.0_f32, 0.0, 0.0, 3.0];
        let down_weight = [1.0_f32, 0.0, 0.0, 1.0];
        let mut scratch = [0.0_f32; 2];
        let mut output = [0.0_f32; 2];

        reference::swiglu_mlp_f32(
            &input,
            &gate_weight,
            &up_weight,
            &down_weight,
            &mut scratch,
            &mut output,
            reference::SwigluMlpShape::new(1, 2, 2),
        );

        let expected_0 = 1.0_f32 / (1.0 + (-1.0_f32).exp()) * 2.0;
        let expected_1 = 2.0_f32 / (1.0 + (-2.0_f32).exp()) * 6.0;
        assert!((output[0] - expected_0).abs() < 1.0e-6);
        assert!((output[1] - expected_1).abs() < 1.0e-6);
    }

    #[test]
    fn computes_rotary_embedding_reference() {
        let shape = reference::RotaryEmbeddingShape::new(2, 1, 2);
        let positions = [0_usize, 1];
        let input = [1.0_f32, 0.0, 1.0, 0.0];
        let mut output = [0.0_f32; 4];

        reference::rotary_embedding_f32(&input, &positions, &mut output, shape, 10_000.0);

        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 0.0);
        assert!((output[2] - 1.0_f32.cos()).abs() < 1.0e-6);
        assert!((output[3] - 1.0_f32.sin()).abs() < 1.0e-6);
    }

    #[test]
    fn computes_causal_self_attention_reference() {
        let shape = reference::CausalAttentionShape::new(2, 1, 1, 2);
        let query = [1.0_f32, 0.0, 0.0, 1.0];
        let key = [1.0_f32, 0.0, 0.0, 1.0];
        let value = [10.0_f32, 0.0, 0.0, 20.0];
        let mut scratch = [0.0_f32; 2];
        let mut output = [0.0_f32; 4];

        reference::causal_self_attention_f32(
            &query,
            &key,
            &value,
            &mut scratch,
            &mut output,
            shape,
        );

        assert_eq!(output[0], 10.0);
        assert_eq!(output[1], 0.0);
        assert!(output[2] > 0.0);
        assert!(output[3] > output[2]);
    }

    #[test]
    fn computes_router_topk_probabilities() {
        let input = [1.0_f32, 0.0, 0.0, 1.0];
        let router_weight = [1.0_f32, 0.0, 0.0, 1.0, 2.0, 2.0];
        let mut routing_weights = [0.0_f32; 4];
        let mut selected_experts = [usize::MAX; 4];

        reference::moe_router_topk_f32(
            &input,
            &router_weight,
            &mut routing_weights,
            &mut selected_experts,
            2,
            2,
            3,
            2,
        );

        assert_eq!(selected_experts, [2, 0, 2, 1]);
        assert!((routing_weights[0] + routing_weights[1] - 1.0).abs() < 1.0e-6);
        assert!((routing_weights[2] + routing_weights[3] - 1.0).abs() < 1.0e-6);
        assert!(routing_weights[0] > routing_weights[1]);
        assert!(routing_weights[2] > routing_weights[3]);
    }

    #[test]
    fn combines_topk_expert_outputs() {
        let expert_outputs = [1.0_f32, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0];
        let routing_weights = [0.25_f32, 0.75, 0.5, 0.5];
        let mut output = [0.0_f32; 4];

        reference::moe_combine_topk_f32(&expert_outputs, &routing_weights, &mut output, 2, 2, 2);

        assert_eq!(output, [7.75, 15.5, 16.5, 22.0]);
    }
}

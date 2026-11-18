/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <optional>
#include <tuple>

#include "pytorch_shim.h"

/*
 * From flash-attention
 */
std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor> mha_fwd(
    torch::stable::Tensor q,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    torch::stable::Tensor k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size,
                   // h_k, d) if there is page_table.
    torch::stable::Tensor v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages,
                   // page_size, h_k, dv) if there is page_table.
    std::optional<torch::stable::Tensor> k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
    std::optional<torch::stable::Tensor> v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
    std::optional<torch::stable::Tensor> q_v_,    // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    std::optional<torch::stable::Tensor> out_,    // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    std::optional<torch::stable::Tensor> cu_seqlens_q_,      // b+1
    std::optional<torch::stable::Tensor> cu_seqlens_k_,      // b+1
    std::optional<torch::stable::Tensor> cu_seqlens_k_new_,  // b+1
    std::optional<torch::stable::Tensor>
        seqused_q_,  // b. If given, only this many elements of each batch element's queries and outputs are used.
    std::optional<torch::stable::Tensor>
        seqused_k_,  // b. If given, only this many elements of each batch element's keys are used.
    std::optional<int64_t> max_seqlen_q_,
    // TODO: check if we need max_seqlen_k
    std::optional<int64_t> max_seqlen_k_,
    std::optional<torch::stable::Tensor> page_table_,      // (b_k, max_num_pages_per_seq)
    std::optional<torch::stable::Tensor> kv_batch_idx_,    // b. indices to index into the KV cache
    std::optional<torch::stable::Tensor> leftpad_k_,       // b
    std::optional<torch::stable::Tensor> rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<torch::stable::Tensor> rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<torch::stable::Tensor> seqlens_rotary_,  // b
    std::optional<torch::stable::Tensor> q_descale_,       // (b, h_k), not (b, h)
    std::optional<torch::stable::Tensor> k_descale_,       // (b, h_k)
    std::optional<torch::stable::Tensor> v_descale_,       // (b, h_k)
    std::optional<double> softmax_scale_,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t attention_chunk,
    double softcap,
    bool is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    std::optional<torch::stable::Tensor> scheduler_metadata_,  // (b + 1)
    int64_t num_splits,
    std::optional<bool> pack_gqa_,
    int64_t sm_margin,
    std::optional<const torch::stable::Tensor>& sinks_,      // (h)
    std::optional<torch::stable::Tensor> sparse_mask_fine_,  // [total_q, max_k_blocks, num_int32_per_block]
    bool only_qv);

/*
 * From flash-attention: get_scheduler_metadata
 * Precomputes tile scheduling metadata for FA3 so that the prepare_varlen_num_blocks
 * kernel does not need to run per-layer.
 */
torch::stable::Tensor mha_fwd_get_scheduler_metadata(
    int64_t batch_size,
    int64_t max_seqlen_q,
    int64_t max_seqlen_k,
    int64_t num_heads,
    int64_t num_heads_k,
    int64_t headdim,
    int64_t headdim_v,
    torch::headeronly::ScalarType qkv_dtype,
    torch::stable::Tensor seqused_k,
    std::optional<torch::stable::Tensor> cu_seqlens_q_,
    std::optional<torch::stable::Tensor> cu_seqlens_k_,
    std::optional<torch::stable::Tensor> cu_seqlens_k_new_,
    std::optional<torch::stable::Tensor> seqused_q_,
    std::optional<torch::stable::Tensor> leftpad_k_,
    std::optional<int64_t> page_size,
    int64_t max_seqlen_k_new,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    int64_t attention_chunk,
    bool has_softcap,
    int64_t num_splits,
    std::optional<bool> pack_gqa_,
    int64_t sm_margin);

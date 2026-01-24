/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct Mask {

    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");

    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const window_size_left, window_size_right, sink_token_length;
    cutlass::FastDivmod const attention_chunk_divmod;
    cutlass::FastDivmod const qhead_per_khead_divmod;

    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &attention_chunk_divmod,
         cutlass::FastDivmod const &qhead_per_khead_divmod)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , attention_chunk_divmod(attention_chunk_divmod)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
    {
    };

    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        if (!Seqlenk_mask && !Causal_mask && !Local_mask) { return; }

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        // We want to use the col indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first col index of this thread (get<Col>(tScS_rowcol(_0{}, _0{})))
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        if constexpr (!Causal_mask && !Local_mask) {
            if constexpr (Seqlenk_mask) {  // Just masking based on col
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        } else {  // mask based on both row and col
            if constexpr (!SwapAB) {
                // If PackGQA, we split the work of compute divmod among threads in the same row
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
                int mma_m_idx;
                // Might get OOB but it's ok since we'll check it later
                if constexpr (PackGQA) {
                    mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
                }
                int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + causal_row_offset
                            : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const local_row_offset_right = causal_row_offset + window_size_right;
                    int const local_row_offset_left = causal_row_offset - 1 - window_size_left;
                    int const col_limit_sink = sink_token_length - n_block * kBlockN;  // TODO: subtract thread_col_offset?
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int col_limit_right = !Seqlenk_mask
                            ? row_idx + local_row_offset_right
                            : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                        int col_limit_left = row_idx + local_row_offset_left;
                        if (attention_chunk_divmod.divisor > 0) {
                            int col_limit_left_chunk = flash::round_down(attention_chunk_divmod, row_idx + seqlen_k - seqlen_q) - n_block * kBlockN - thread_col_offset;
                            col_limit_left = std::max(col_limit_left, col_limit_left_chunk);
                            col_limit_right = std::min(col_limit_right, col_limit_left_chunk + attention_chunk_divmod.divisor);
                        }
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
                            if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            } else {
                // TODO: backward does not support attention_chunk yet
                int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
                int const causal_row_offset = seqlenk_col_limit - seqlen_q + m_block * kBlockM + thread_row_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset - window_size_right;
                        int const row_limit_bot = col0 < col_limit_sink ? kBlockM : col0 - causal_row_offset + window_size_left;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
                            if (row_idx < row_limit_top || row_idx > row_limit_bot) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            }
        }
    };

};

template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct SparseMask {
    static constexpr int kNumInt32PerBlock = (kBlockN + 31) / 32;

    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const max_k_blocks;
    cutlass::FastDivmod const qhead_per_khead_divmod;

    CUTLASS_DEVICE
    SparseMask(const int thread_idx, const int seqlen_q, const int seqlen_k,
               int const max_k_blocks,
               cutlass::FastDivmod const qhead_per_khead_divmod
            )
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , max_k_blocks(max_k_blocks)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
    {
    }

    // smem_mask_ptr -> current smem in stage
    // Reordered layout: bits are grouped by quad_lane for optimal GMMA access
    //   For kBlockN=128 (4 words, 32 bits per quad_lane):
    //     - word0: quad_lane 0, word1: quad_lane 1, word2: quad_lane 2, word3: quad_lane 3
    //   For kBlockN=64 (2 words, 16 bits per quad_lane):
    //     - word0 low 16: quad_lane 0, word0 high 16: quad_lane 1
    //     - word1 low 16: quad_lane 2, word1 high 16: quad_lane 3
    template <bool Seqlenk_mask=false, typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block,
               uint32_t const* __restrict__ smem_mask_ptr) const {

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});
        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));

        static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});

        // Compute quad_lane from thread_col_offset (compile-time derivable per thread)
        // quad_lane 0,1,2,3 have thread_col_offset 0,2,4,6 respectively
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const quad_lane = thread_col_offset / 2;

        // For seqlenk_mask: use thread0's column indices (compile-time known) with adjusted limit
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;

        const bool is_block_safe = n_block < max_k_blocks;

        #pragma unroll
        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
            int local_row = get<Row>(tScS_rowcol(m, _0{}));

            int global_row_idx;
            if constexpr (!PackGQA) {
                global_row_idx = local_row + m_block * kBlockM;
            } else {
                int mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + local_row);
                global_row_idx = __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
            }

            bool const is_row_safe = (global_row_idx < seqlen_q) && is_block_safe;
            if (!is_row_safe) {
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    tSrS_rowcol(m, n) = -INFINITY;
                }
                continue;
            }

            if constexpr (kBlockN == 128) {
                // Reordered layout: each quad_lane's 32 bits are in one word
                // Read only the word needed by this thread's quad_lane
                uint32_t const mask_word = smem_mask_ptr[local_row * 4 + quad_lane];

                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    // In reordered layout, bit index = n (compile-time known!)
                    bool const mask_bit = (mask_word >> n) & 1;

                    if constexpr (Seqlenk_mask) {
                        // Use thread0's column index (compile-time known) with adjusted limit
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        bool const is_valid = col0 < seqlenk_col_limit;
                        bool const final_keep = is_valid && mask_bit;
                        if (!final_keep) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    } else {
                        if (!mask_bit) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    }
                }
            } else if constexpr (kBlockN == 64) {
                // Reordered layout: quad_lane 0,1 share word0, quad_lane 2,3 share word1
                // Each quad_lane gets 16 bits (low or high half)
                int const word_idx = quad_lane / 2;
                int const half_idx = quad_lane % 2;
                uint32_t const full_word = smem_mask_ptr[local_row * 2 + word_idx];
                uint32_t const mask_word = half_idx ? (full_word >> 16) : full_word;

                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    // In reordered layout, bit index = n (compile-time known!)
                    bool const mask_bit = (mask_word >> n) & 1;

                    if constexpr (Seqlenk_mask) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        bool const is_valid = col0 < seqlenk_col_limit;
                        bool const final_keep = is_valid && mask_bit;
                        if (!final_keep) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    } else {
                        if (!mask_bit) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    }
                }
            } else {
                // General case for other kBlockN values (fallback to original logic)
                // This assumes the mask is NOT reordered for non-standard sizes
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    int const token_in_block = int(get<Col>(tScS_rowcol(m, n)));
                    int const word_idx = token_in_block / 32;
                    int const bit_idx = token_in_block % 32;

                    bool const mask_bit = (smem_mask_ptr[local_row * kNumInt32PerBlock + word_idx] >> bit_idx) & 1;

                    if constexpr (Seqlenk_mask) {
                        bool const is_valid = (n_block * kBlockN + token_in_block < seqlen_k);
                        bool const final_keep = is_valid && mask_bit;
                        if (!final_keep) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    } else {
                        if (!mask_bit) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    }
                }
            }
        }
    }
};

} // namespace flash

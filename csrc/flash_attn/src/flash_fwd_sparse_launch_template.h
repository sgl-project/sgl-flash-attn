/******************************************************************************
 * Copyright (c) 2024, PAI, Alibaba Cloud.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include "flash_fwd_launch_template.h"
#include "flash_fwd_sparse_kernel.h"
#include "flash_sparse.h"

namespace FLASH_NAMESPACE {

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        FLASH_NAMESPACE::compute_sparse_attn<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_sparse_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
            SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                constexpr bool IsEvenMNConst = false;
                constexpr bool Is_local = false;
                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                // If Is_local, set Is_causal to false
                auto kernel = &flash_fwd_sparse_kernel<Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap>;
                if (smem_size >= 48 * 1024) {
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim160(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 160;
    run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim224(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 224;
    run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal>(params, stream);
}

} // namespace FLASH_NAMESPACE
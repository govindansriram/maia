#pragma once

#include <cutlass/arch/config.h>

#define KL_HOST_DEVICE __host__ __device__ __forceinline__
#define KL_DEVICE __device__ __forceinline__

// Defined in device passes that may emit sm_100-family instructions (tcgen05, TMEM, CLC).
#if defined(CUTLASS_ARCH_MMA_SM100A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM100F_ENABLED) || \
    defined(CUTLASS_ARCH_MMA_SM103A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM103F_ENABLED)
#define KL_ARCH_SM100 1
#endif

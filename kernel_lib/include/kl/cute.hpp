#pragma once

// The only place kl includes CuTe / CUTLASS headers. tools/check_includes.py fails
// the build if anything here pulls in the atom, collective or pipeline layers.

#include <cute/tensor_impl.hpp>
#include <cute/pointer.hpp>
#include <cute/pointer_flagged.hpp>
#include <cute/pointer_swizzle.hpp>
#include <cute/swizzle_layout.hpp>
#include <cute/arch/util.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/arch/cluster_sm90.hpp>     // elect_one_sync
#include <cute/arch/copy_sm90_desc.hpp>   // TMA descriptor prefetch, cache hints
#include <cute/arch/copy_sm90_tma.hpp>    // cp.async.bulk.tensor loads and stores

#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>          // mbarrier

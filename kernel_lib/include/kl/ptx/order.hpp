#pragma once

// Memory-ordering choices shared by the ptx/ wrappers (mbarrier.hpp, cluster.hpp).
namespace kl {

// Memory semantics of an arrive: release publishes this thread's earlier writes along with the
// arrive; relaxed is only a signal. (Waits are always acquire.)
enum class Sem { release, relaxed };

// Which threads a release / acquire is guaranteed to be visible to: this CTA, or the cluster.
enum class Scope { cta, cluster };

}  // namespace kl

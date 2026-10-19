# mbarrier and fence patterns

The common ways to use `kl::Mbarrier`, `kl::PeerMbarrier` (`mbarrier.hpp`), the fences
(`fence.hpp`) and the cluster barrier (`cluster.hpp`). Each pattern is run on hardware by the
test named under it, in `tests/device/test_barrier_patterns.cu` unless noted. The snippets are
trimmed from the code between the `examples.md` markers in those tests.

## Rules

- **A barrier lives in the CTA that waits on it.** PTX can only init and wait on a barrier in
  the caller's own CTA. Other CTAs can only arrive on it, through `bar.peer(rank)`.
- **Sem** (peer arrives only) says what the arrive publishes. `Sem::relaxed` is just a signal.
  `Sem::release` also makes this thread's earlier ordinary stores visible to the waiters.
- **Scope** (waits only) says whose writes the wait acquires. `Scope::cta` (the default) is this
  CTA's. `Scope::cluster` also covers a peer that published with `Sem::release`.
- **TMA bytes need neither.** Data that TMA writes is tracked by the barrier's transaction count.
  When the phase completes, the bytes are visible to every waiter.
- **Cost.** A cluster release (`Sem::release`, `fence_release_cluster`) emits a GPU-wide fence
  (`MEMBAR.ALL.GPU`). A cluster acquire (`wait<Scope::cluster>`) invalidates L1 (`CCTL.IVALL`).
  Pay for them only where data actually crosses CTAs.

| Situation | Use | Section |
|---|---|---|
| Hand data to another warp of this CTA | `bar.arrive()` / `bar.wait(phase)` | 1 |
| Wait but stay free to do other work | `bar.try_wait(phase)` (or `try_wait<Scope::cluster>`) | 2, 7 |
| TMA load into this CTA | `bar.arrive_expect_tx(bytes)` + `tma_load` | 3 |
| Threads wrote a tile, TMA stores it | every writer: `fence_view_async_shared()`, then sync | 4 |
| Barriers that peers arrive on | `fence_mbarrier_init()` + `cluster_arrive<Sem::relaxed>()` + `cluster_wait()`, then `cluster_sync()` at exit | 5 |
| Tell a peer "done" / "slot free" | `peer(r).arrive<Sem::relaxed>()` | 6 |
| Hand data you stored to one peer | `peer(r).arrive<Sem::release>()` + `wait<Scope::cluster>` | 7 |
| Hand data to several peers | `fence_release_cluster()` once + `arrive<Sem::relaxed>()` each | 8 |
| Arm a peer's barrier for TMA bytes | `peer(r).arrive_expect_tx<Sem::relaxed>(bytes)` | 9 |
| ... and hand it data you stored | `peer(r).arrive_expect_tx<Sem::release>(bytes)` + `wait<Scope::cluster>` | 10 |

`kl::peer_ptr(&x, rank)` (`cluster.hpp`) in the snippets is a plain C++ pointer to `x` in the CTA
with that rank, so `*kl::peer_ptr(&x, rank) = v` is an ordinary store into another CTA's shared
memory.

## 1. Local handoff: `arrive` / `wait`

A producer warp fills a buffer and the consumer warps read it, using two barriers: `full` (data is
ready) and `empty` (the buffer can be reused). Each round flips the phase, so round `i` waits on
parity `i % 2`. A local `arrive` is `release.cta`, which is free, and the `__syncwarp` lets one
elected lane's arrive cover the whole warp's stores. Any number of warps can wait on the same
phase; only arrivals are counted (`empty.init(2)`: one per consumer warp).

```cpp
s.full.init(1);   // the producer's arrive
s.empty.init(2);  // one arrive per consumer warp

if (warp == 0) {
  for (int i = 0; i < kRounds; ++i) {
    if (i > 0) s.empty.wait((i - 1) % 2);  // consumers are done with round i - 1
    s.buf[lane] = 100 * i + lane;
    __syncwarp();
    if (kl::elect_one()) s.full.arrive();
  }
} else {
  for (int i = 0; i < kRounds; ++i) {
    s.full.wait(i % 2);
    out[...] = s.buf[lane];
    __syncwarp();
    if (kl::elect_one()) s.empty.arrive();
  }
}
```

Tested by `BarrierPatterns.LocalHandoff`.

## 2. Polling: `try_wait`

`try_wait` returns whether the phase is done. It suspends the thread for a hardware-defined
while first, so a retry loop doesn't burn issue slots. Use it when the waiter has other work to
interleave. Otherwise `wait` is the same loop written in PTX, and it gets a faster first check.

```cpp
while (!s.full.try_wait(i % 2)) {
  // free to do other work here
}
```

Tested by `BarrierPatterns.LocalHandoff` (the second consumer warp). For `try_wait` at cluster
scope, see section 7.

## 3. TMA load into this CTA: `arrive_expect_tx`

One instruction records the tile's byte count and then arrives. The TMA counts those bytes down
as they land, and the phase completes when the last one does. The order matters: an arrive
before the expect could complete the phase before any data has landed, and the fused form makes
that impossible. Bytes may also land before the arm: the transaction count goes briefly negative,
and the phase stays open because the arrival is still pending. Waiters need no cluster scope even
though TMA wrote the data. The appendix walks through both cases.

```cpp
if (kl::elect_one()) s.bar.arrive_expect_tx(Tile::bytes);
kl::tma_load(s.tile, src, c, s.bar);
s.bar.wait(0);
```

Tested by `TmaCopy.*` (`tests/device/test_tma_copy.cu`).

## 4. Threads write a tile, then TMA stores it: `fence_view_async_shared`

TMA reads shared memory through the async proxy, which doesn't see ordinary stores until they are
fenced. **Every thread that wrote** fences its own writes, then everyone syncs, then one warp
issues the store. This fence is not cumulative like a release, so one thread can't fence for the
others. `kl::tma_store` fences its own warp again, which is harmless.

```cpp
view(r, k) = ...;               // all 128 threads write the tile
kl::fence_view_async_shared();  // every thread that wrote
__syncthreads();
if (kl::warp_id() == 0) {
  kl::tma_store(dst, kl::TileCoord<2>{0, 0}, s.tile);
  kl::tma_store_wait<0>();
}
```

Tested by `BarrierPatterns.TmaStoreAfterThreadWrites`. (A missing fence is a race, so this test
can pass without it. It shows the pattern; it doesn't prove the fence is needed.)

## 5. Cluster setup and teardown: `cluster_arrive` / `cluster_wait` / `cluster_sync`

The cluster barrier (`ptx/cluster.hpp`) is `__syncthreads()` for the whole cluster: every thread
of every CTA arrives, and `cluster_wait()` returns once all of them have. Whole warps call it.

At setup, the wait makes sure no CTA arrives on a peer's barrier before that peer has initialized
it. `fence_mbarrier_init()` already publishes the init to the cluster, so the arrive can be
relaxed and skip the GPU-wide fence a release arrive would add (the PTX docs give this exact
sequence). At exit, the full `cluster_sync()` keeps every CTA alive while others may still arrive
on its barriers or store into its shared memory.

```cpp
if (kl::elect_one()) {
  s.bar.init(count);
  kl::fence_mbarrier_init();
}
kl::cluster_arrive<kl::Sem::relaxed>();  // fence_mbarrier_init() already published the init
kl::cluster_wait();                      // every CTA's barriers exist before anyone arrives remotely
// ... the kernel ...
kl::cluster_sync();  // no CTA exits while another may still arrive on its barriers
```

Use the plain `cluster_arrive()` (release) when the setup also wrote data other CTAs will read
after the wait. `kl::cluster_rank()` gives this CTA's rank, as `peer()` takes it.

Tested by every cluster test below. A single-CTA kernel uses `__syncthreads()` instead and needs
nothing at exit.

## 6. Signal a peer: `arrive<Sem::relaxed>`

A pure signal with no data attached, e.g. "my copy of this multicast stage is free" sent back to
the CTA that issues the multicast. Relaxed is a bare remote arrive with no fences. CUTLASS's
default remote arrive (`release.cta`) compiles to the same instruction. `peer(own rank)` is
allowed and arrives on your own barrier.

```cpp
if (kl::elect_one()) s.from_peer.peer(other).arrive<kl::Sem::relaxed>();
s.from_peer.wait(0);  // nothing to acquire: Scope::cta
```

Tested by `BarrierPatterns.PeerSignal`.

## 7. Publish to one peer: `arrive<Sem::release>` + `wait<Scope::cluster>`

You stored data into a peer's shared memory (or into global memory it will read), and the arrive
tells it the data is there. The release waits until your stores have landed before the arrive
goes out. The peer's cluster-scope wait pairs with it. Here every lane publishes its own store,
so the barrier counts 32 arrivals.

```cpp
s.ready.init(32);  // one arrive per lane of the peer

*kl::peer_ptr(&s.data[lane], other) = peer_value(rank, lane);
s.ready.peer(other).arrive<kl::Sem::release>();

if (rank == 0) {
  s.ready.wait<kl::Scope::cluster>(0);
} else {
  while (!s.ready.try_wait<kl::Scope::cluster>(0)) {
  }
}
out[rank * 32 + lane] = s.data[lane];
```

`try_wait` takes the same `Scope` as `wait`, so a poller can acquire a peer's release too (CTA 1
here).

Tested by `BarrierPatterns.PublishToOnePeer`.

## 8. Publish to several peers: one `fence_release_cluster`, then relaxed arrives

Each `Sem::release` arrive pays for its own GPU-wide fence, and ptxas does not merge them. One
`fence_release_cluster()` followed by relaxed arrives is a release on every one of them (the PTX
memory model's "release pattern"). The `__syncwarp` lets one elected lane fence and arrive for
the whole warp, because a release also covers writes that synced with it first.

```cpp
s.ready.init(kFanOut - 1);  // one arrive per other CTA

for (uint32_t p = 0; p < kFanOut; ++p)
  if (p != rank) *kl::peer_ptr(&s.data[rank][lane], p) = peer_value(rank, lane);
__syncwarp();  // the elected lane's fence now covers every lane's stores
if (kl::elect_one()) {
  kl::fence_release_cluster();
  for (uint32_t p = 0; p < kFanOut; ++p)
    if (p != rank) s.ready.peer(p).arrive<kl::Sem::relaxed>();
}
s.ready.wait<kl::Scope::cluster>(0);
```

Tested by `BarrierPatterns.PublishToSeveralPeers` (a cluster of 4).

## 9. Arm a peer's barrier for TMA bytes: `arrive_expect_tx<Sem::relaxed>`

The usual expect on a peer. The issuing CTA knows how many bytes a TMA will deliver into another
CTA (here, a multicast into both CTAs), so it arms each receiver's barrier itself. The TMA's bytes
are tracked by the transaction count, so the arming thread has nothing to publish: relaxed.
Receivers wait with `Scope::cta`.

```cpp
s.full.init(1);  // CTA 0's arm

// CTA 0, one thread:
for (uint32_t r = 0; r < 2; ++r)
  s.full.peer(r).arrive_expect_tx<kl::Sem::relaxed>(McTile::bytes);
cute::SM90_TMA_LOAD_MULTICAST::copy(src.get(), s.full.ptr(), uint16_t(0b11), ...);

// every CTA:
s.full.wait(0);
```

Tested by `BarrierPatterns.ArmPeerForTmaBytes`. (In most multicast kernels each CTA arms its own
barrier with a local `arrive_expect_tx` instead; this is for when only the issuer knows the
count.)

## 10. Arm and publish: `arrive_expect_tx<Sem::release>`

The rare case: the arming thread also stored ordinary data the peer will read, e.g. a small header
or scale next to the tile. Release makes the one arrive cover both. The receivers must then wait
with `Scope::cluster` to see the stored data. The TMA bytes would be visible either way.

```cpp
*kl::peer_ptr(&s.note, r) = note_value(r);
s.full.peer(r).arrive_expect_tx<kl::Sem::release>(McTile::bytes);

s.full.wait<kl::Scope::cluster>(0);  // acquire what CTA 0 stored
```

Tested by `BarrierPatterns.ArmPeerAndPublish`.

## Appendix: why the expect comes before the arrive

Each phase has two counters, and it completes the moment **both are zero**:

- **pending arrivals** start at the `init` count. Each arrive subtracts 1.
- **the transaction (tx) count** starts at 0. `expect_tx(B)` adds B, and each byte TMA delivers
  subtracts 1.

The barrier has no idea a TMA is coming until an expect tells it. The pending arrival is what holds
the phase open in the meantime, so the arrive has to be the last thing: it means "this phase is
set up, let the waiters go once the counters reach zero". With `init(1)` and a tile of B bytes:

**Expect, then arrive (what `arrive_expect_tx` does):** the phase waits for the data.

| step | pending | tx | |
|---|---|---|---|
| start | 1 | 0 | |
| expect B | 1 | B | |
| arrive | 0 | B | waiting for bytes |
| TMA delivers B bytes | 0 | 0 | phase completes, tile is in shared memory |

**Arrive, then expect (two separate instructions, wrong order):** the phase completes early.

| step | pending | tx | |
|---|---|---|---|
| start | 1 | 0 | |
| arrive | 0 | 0 | **phase completes with no data**: waiters read garbage |
| expect B | | | too late |

**Bytes land before the arm (e.g. another CTA's multicast, section 9):** still correct.

| step | pending | tx | |
|---|---|---|---|
| start | 1 | 0 | |
| TMA delivers B bytes | 1 | −B | negative is allowed; the pending arrival keeps the phase open |
| expect B, arrive | 0 | 0 | phase completes, tile is in shared memory |

So this is not an arbitrary PTX rule; it follows from "complete when both counters are zero".
PTX defines the fused `mbarrier.arrive.expect_tx` to apply the expect first and then the arrive.
Separate `expect_tx` and `arrive` instructions issued in that order would also be correct; the
fused form makes the wrong order impossible to write, in one instruction.

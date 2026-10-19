# maia

GPU kernels for datacenter Blackwell (sm_100 / sm_103: B200, B300).

- `kernel_lib/`: **kl**, a header-only C++20 kernel library (`include/kl/`) and its tests.
- `PLAN.md`: the design and roadmap.
- `scripts/remote.sh`: build and test on a remote B200.
- `third_party/cutlass`: CUTLASS submodule. kl uses only an allow-listed subset of its headers.

## Requirements

- CUDA 12.8+ (nvcc and the driver API), CMake 3.28+, a C++20 host compiler, Python 3
- The CUTLASS submodule: `git submodule update --init`

## Build and test

From the repo root:

```bash
cmake -S kernel_lib -B kernel_lib/build      # Release by default: assert() is compiled out
cmake --build kernel_lib/build -j
ctest --test-dir kernel_lib/build --output-on-failure
```

To build with `assert()` checks on, use Debug, in a separate build directory:

```bash
cmake -S kernel_lib -B kernel_lib/build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build kernel_lib/build-debug -j
ctest --test-dir kernel_lib/build-debug --output-on-failure
```

What runs where:

- **Unit tests** (`kl_unit_tests`) are built for sm_89 and sm_100a (`KL_TEST_CUDA_ARCHITECTURES`),
  so they also run on a local Ada GPU.
- **Device tests** (`kl_device_tests`) use TMA, mbarriers and clusters, so they are built for
  sm_100a only and skip on other GPUs.
- **`kl_include_allowlist`** needs no GPU: it checks kl's headers pull in no banned CUTLASS
  headers.

## Remote B200

`scripts/remote.sh` copies the repo to a B200 machine and builds there in Release for sm_100a:

```bash
scripts/remote.sh test                         # sync, build, run all tests
scripts/remote.sh test -R BarrierPatterns      # extra args go to ctest
scripts/remote.sh run <cmd...>                 # run a command from the remote repo root
scripts/remote.sh shell                        # interactive shell on the remote
```

Point it at a machine with `KL_REMOTE_HOST`, `KL_REMOTE_PORT` and `KL_REMOTE_DIR`. It uses SSH key
authentication.

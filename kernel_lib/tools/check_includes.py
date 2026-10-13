#!/usr/bin/env python3
"""Fail if kl's public headers pull in parts of CUTLASS that kl deliberately avoids.

Runs `nvcc -M` on a translation unit that includes <kl/kl.hpp> and checks the full
transitive include graph against BANNED.
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BANNED = [
    r"cute/atom/",
    r"cute/tensor\.hpp$",
    r"cute/algorithm/(copy|gemm|cooperative_copy|cooperative_gemm|prefetch)\.hpp$",
    r"cutlass/gemm/",
    r"cutlass/epilogue/",
    r"cutlass/pipeline/",
    r"cutlass/conv/",
]


def include_graph(nvcc: str, includes: list[str], arch: str) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "all.cu"
        src.write_text("#include <kl/kl.hpp>\n")
        cmd = [nvcc, "-std=c++20", "--expt-relaxed-constexpr", f"-arch={arch}", "-M", str(src)]
        for inc in includes:
            cmd += ["-I", inc]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"nvcc -M failed:\n{res.stderr}")
        return [tok for tok in res.stdout.replace("\\\n", " ").split() if tok.endswith((".h", ".hpp"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvcc", default="nvcc")
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--arch", default="sm_100a")
    args = ap.parse_args()

    headers = include_graph(args.nvcc, args.include, args.arch)
    pattern = re.compile("|".join(f"(?:{p})" for p in BANNED))
    offenders = sorted({h for h in headers if pattern.search(h)})
    cutlass = [h for h in headers if re.search(r"/(cute|cutlass)/", h)]

    print(f"kl.hpp pulls in {len(cutlass)} CuTe/CUTLASS headers")
    if offenders:
        print("banned headers in kl's include graph:")
        for h in offenders:
            print(f"  {h}")
        return 1
    print("OK: no banned headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())

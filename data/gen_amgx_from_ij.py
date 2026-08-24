#!/usr/bin/env python3
"""
Convert a HyPre IJ system (<base>.00000 + <base>_rhs.00000) into the AMGX
MatrixMarket format (<base>.amgx), so AMGX solves the EXACT same matrix + RHS
that HyPre (and, via the augmented .mtx, AC) solve.

AMGX format (general = full matrix, 1-indexed, RHS appended), per AMGX's own
writeSystemMatrixMarket:
    %%MatrixMarket matrix coordinate real general
    %%NVAMG 1 1 rhs
    <N> <N> <nnz>
    i j val            (1-indexed, FULL matrix -- both triangles + diagonal)
    ...
    <N>
    b_1
    ...
    b_N

The HyPre .00000 is already the FULL matrix in 0-indexed COO (gen_chimera_sweep.jl
save_ij writes every nonzero), so conversion is just a +1 reindex and a re-banner.

Usage:
    python3 gen_amgx_from_ij.py <base> [<base> ...]      # <base> = path without extension
    python3 gen_amgx_from_ij.py <dir>                    # all <base>.00000 in a directory
"""
import sys, os, glob

def convert(base):
    mtx_in, rhs_in, out = base + ".00000", base + "_rhs.00000", base + ".amgx"
    if not (os.path.exists(mtx_in) and os.path.exists(rhs_in)):
        print(f"  SKIP {base}: missing .00000 or _rhs.00000"); return False
    # --- matrix ---
    with open(mtx_in) as f:
        h = f.readline().split()
        N = int(h[1]) + 1            # "ilower iupper" -> N = iupper+1
        f.readline()                 # second header line (column range)
        rows = f.read().splitlines() # remaining lines: "r c v" (0-indexed)
    nnz = len(rows)
    # --- rhs ---
    with open(rhs_in) as f:
        f.readline()                 # header
        b = [ln.split()[1] for ln in f.read().splitlines() if ln.strip()]
    if len(b) != N:
        print(f"  WARN {base}: rhs has {len(b)} entries, matrix N={N}")
    # --- write ---
    with open(out, "w") as o:
        o.write("%%MatrixMarket matrix coordinate real general\n")
        o.write("%%NVAMG 1 1 rhs\n")
        o.write(f"{N} {N} {nnz}\n")
        for ln in rows:
            r, c, v = ln.split()
            o.write(f"{int(r)+1} {int(c)+1} {v}\n")
        o.write(f"{N}\n")
        o.write("\n".join(b))
        o.write("\n")
    print(f"  {os.path.basename(base)}: N={N} nnz={nnz} -> {os.path.basename(out)}")
    return True

def main(argv):
    if not argv:
        print(__doc__); sys.exit(1)
    bases = []
    for a in argv:
        if os.path.isdir(a):
            bases += [f[:-6] for f in sorted(glob.glob(os.path.join(a, "*.00000")))
                      if not f.endswith("_rhs.00000")]
        else:
            bases.append(a[:-6] if a.endswith(".00000") else a)
    n = sum(convert(b) for b in bases)
    print(f"converted {n}/{len(bases)}")

if __name__ == "__main__":
    main(sys.argv[1:])

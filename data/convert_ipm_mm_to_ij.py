#!/usr/bin/env python3
"""Convert an ipmMat MatrixMarket (.mm, real symmetric SDDM) matrix to HYPRE IJ
format, with the paper-correct right-hand side b = M g / ||M g|| (random Gaussian g).

This reproduces the RHS recipe from Gao-Kyng-Spielman (arXiv:2303.00709, p.24),
which the alternating-+/-1 RHS in convert_mtx_to_ij.py does NOT match.

Output (matches the format the `ij` driver's -fromfile/-rhsfromfile already read):
  <out>.00000        line1: "0 n-1"  (ilower iupper)
                     line2: "0 n-1"  (jlower jupper)
                     then:  "i j v"  (0-indexed, BOTH triangles written)
  <out>_rhs.00000    line1: "0 n-1"
                     then:  "idx val"

Pure stdlib (no numpy/scipy) so it runs in the base env. Intended for the small
uni_chimera matrices (n=1e5, ~1M nnz); for the multi-100M-nnz spielman files use a
numpy/scipy path instead.

Usage:
  python3 convert_ipm_mm_to_ij.py <in.mm> <out_basepath> [seed]
    -> writes <out_basepath>.00000 and <out_basepath>_rhs.00000
"""
import sys
import math
import random

def main():
    if len(sys.argv) < 3:
        sys.exit("usage: convert_ipm_mm_to_ij.py <in.mm> <out_basepath> [seed]")
    infile = sys.argv[1]
    outbase = sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    n = None
    declared_nnz = None
    # store entries as parsed (one per stored (i>=j) pair, 1-indexed in file)
    rows = []
    cols = []
    vals = []

    with open(infile) as f:
        for line in f:
            if not line or line[0] == '%':
                continue
            parts = line.split()
            if n is None:
                n = int(parts[0])
                declared_nnz = int(parts[2]) if len(parts) >= 3 else None
                continue
            i = int(parts[0]) - 1   # to 0-indexed
            j = int(parts[1]) - 1
            v = float(parts[2])
            rows.append(i); cols.append(j); vals.append(v)

    if n is None:
        sys.exit("ERROR: no MatrixMarket size header found in %s" % infile)

    # ---- RHS: b = M g / ||M g|| with random Gaussian g (seeded, reproducible) ----
    rng = random.Random(seed)
    g = [rng.gauss(0.0, 1.0) for _ in range(n)]
    b = [0.0] * n
    for i, j, v in zip(rows, cols, vals):
        b[i] += v * g[j]
        if i != j:                 # symmetric: the (j,i) entry equals (i,j)
            b[j] += v * g[i]
    nrm = math.sqrt(sum(x * x for x in b))
    if nrm == 0.0:
        sys.exit("ERROR: ||M g|| == 0 (degenerate); try a different seed")
    inv = 1.0 / nrm
    b = [x * inv for x in b]

    # ---- write IJ matrix (both triangles) ----
    mfile = outbase + ".00000"
    with open(mfile, "w") as out:
        out.write("0 %d\n" % (n - 1))
        out.write("0 %d\n" % (n - 1))
        w = out.write
        for i, j, v in zip(rows, cols, vals):
            w("%d %d %.16e\n" % (i, j, v))
            if i != j:
                w("%d %d %.16e\n" % (j, i, v))

    # ---- write RHS ----
    rfile = outbase + "_rhs.00000"
    with open(rfile, "w") as out:
        out.write("0 %d\n" % (n - 1))
        w = out.write
        for idx in range(n):
            w("%d %.16e\n" % (idx, b[idx]))

    off = sum(1 for i, j in zip(rows, cols) if i != j)
    full_nnz = len(vals) + off
    print("Done: n=%d, stored=%d (declared %s), full_nnz=%d, seed=%d"
          % (n, len(vals), declared_nnz, full_nnz, seed))
    print("  matrix: %s" % mfile)
    print("  rhs:    %s" % rfile)

if __name__ == "__main__":
    main()

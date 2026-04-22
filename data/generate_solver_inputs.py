"""
Generate matrix and RHS files for hypre (IJ format) and AMGX (MTX format).

Physics matrices: used as-is (positive diagonal, negative off-diagonal).
  Exception: SPE matrices have flipped signs and need negation.

Graph matrices: build Laplacian from adjacency (pattern) files.
  - Off-diagonal = -1.0
  - Diagonal = degree (sum of abs of off-diagonals)

RHS: random zero-mean vector using mt19937 with seed=0, matching the C++ code
in gpu_implementation/solver.hpp:generate_zero_sum_vector.
"""

import sys
import os
import struct
import numpy as np
from collections import defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
PHYSICS_DIR = os.path.join(DATA_DIR, "..", "physics")


def generate_zero_sum_rhs(n, seed=0):
    """Match the C++ generate_zero_sum_vector: mt19937(seed), uniform_real[0,1), subtract mean."""
    rng = np.random.MT19937(seed)
    gen = np.random.Generator(rng)
    vec = gen.random(n, dtype=np.float64)
    vec -= vec.mean()
    return vec


def read_mtx_physics(filepath):
    """Read a real symmetric coordinate MTX file. Returns n, list of (i,j,v) 1-indexed."""
    entries = []
    n = None
    with open(filepath) as f:
        for line in f:
            if line.startswith('%'):
                continue
            parts = line.strip().split()
            if n is None:
                n = int(parts[0])
                continue
            i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
            entries.append((i, j, v))
    return n, entries


def read_mtx_graph(filepath):
    """Read a graph MTX file (pattern or integer/real symmetric).
    Returns n, list of (i, j, weight) 1-indexed.
    For pattern files, weight=1. For valued files, weight=abs(value).
    Diagonal entries are skipped (they'll be recomputed as degree)."""
    edges = []
    n = None
    is_pattern = False
    with open(filepath) as f:
        for line in f:
            if line.startswith('%%'):
                is_pattern = 'pattern' in line.lower()
                continue
            if line.startswith('%'):
                continue
            parts = line.strip().split()
            if n is None:
                n = int(parts[0])
                continue
            i, j = int(parts[0]), int(parts[1])
            if i == j:
                continue
            if is_pattern:
                w = 1.0
            else:
                w = abs(float(parts[2]))
            edges.append((i, j, w))
    return n, edges


def write_hypre_ij(filepath, n, entries_0indexed):
    """Write hypre IJ format. entries are (i, j, v) 0-indexed, both triangles."""
    with open(filepath + '.00000', 'w') as f:
        f.write(f'0 {n-1}\n')
        f.write(f'0 {n-1}\n')
        for i, j, v in entries_0indexed:
            f.write(f'{i} {j} {v:.15e}\n')


def write_hypre_rhs(filepath, rhs):
    """Write hypre IJ vector format."""
    n = len(rhs)
    with open(filepath + '.00000', 'w') as f:
        f.write(f'0 {n-1}\n')
        for i, v in enumerate(rhs):
            f.write(f'{i} {v:.15e}\n')


def write_amgx_mtx(filepath, n, nnz_lower, header_lines, entries_1indexed, rhs):
    """Write AMGX MTX format with embedded RHS.
    entries_1indexed: lower triangle only (i >= j), 1-indexed.
    """
    with open(filepath, 'w') as f:
        # Header
        f.write('%%MatrixMarket matrix coordinate real symmetric\n')
        f.write('%%AMGX rhs\n')
        for line in header_lines:
            f.write(line)
        f.write(f'{n} {n} {nnz_lower}\n')
        for i, j, v in entries_1indexed:
            f.write(f'{i} {j} {v:.15e}\n')
        # RHS
        f.write(f'{n}\n')
        for v in rhs:
            f.write(f'{v:.15e}\n')


def process_physics(name, mtx_path, outdir_hypre, outdir_amgx, negate=False):
    """Process a physics matrix (already has values, symmetric)."""
    print(f'Processing physics: {name}')
    n, entries = read_mtx_physics(mtx_path)

    if negate:
        entries = [(i, j, -v) for i, j, v in entries]

    # Generate RHS
    rhs = generate_zero_sum_rhs(n)

    # For hypre: need both triangles, 0-indexed
    hypre_entries = []
    for i, j, v in entries:
        hypre_entries.append((i-1, j-1, v))
        if i != j:
            hypre_entries.append((j-1, i-1, v))

    write_hypre_ij(os.path.join(outdir_hypre, name), n, hypre_entries)
    write_hypre_rhs(os.path.join(outdir_hypre, name + '_rhs'), rhs)

    # For AMGX: lower triangle, 1-indexed
    amgx_entries = []
    for i, j, v in entries:
        if i >= j:
            amgx_entries.append((i, j, v))
        else:
            amgx_entries.append((j, i, v))

    # Deduplicate (in case both (i,j) and (j,i) appear)
    seen = {}
    for i, j, v in amgx_entries:
        seen[(i, j)] = v
    amgx_entries = [(i, j, v) for (i, j), v in sorted(seen.items())]

    write_amgx_mtx(
        os.path.join(outdir_amgx, name + '.mtx'),
        n, len(amgx_entries), [], amgx_entries, rhs
    )

    print(f'  {name}: n={n}, nnz_lower={len(amgx_entries)}')


def process_graph(name, mtx_path, outdir_hypre, outdir_amgx):
    """Process a graph (pattern/integer/real) matrix into a Laplacian.
    Off-diagonal = -weight, diagonal = sum of weights per row."""
    print(f'Processing graph: {name}')
    n, edges = read_mtx_graph(mtx_path)

    # Compute weighted degree: diagonal = sum of abs(off-diagonal weights)
    degree = defaultdict(float)
    for i, j, w in edges:
        degree[i] += w
        degree[j] += w

    # Generate RHS
    rhs = generate_zero_sum_rhs(n)

    # For hypre: both triangles + diagonal, 0-indexed
    hypre_entries = []
    for node in range(1, n+1):
        deg = degree.get(node, 0.0)
        hypre_entries.append((node-1, node-1, deg))
    for i, j, w in edges:
        hypre_entries.append((i-1, j-1, -w))
        hypre_entries.append((j-1, i-1, -w))

    write_hypre_ij(os.path.join(outdir_hypre, name), n, hypre_entries)
    write_hypre_rhs(os.path.join(outdir_hypre, name + '_rhs'), rhs)

    # For AMGX: lower triangle + diagonal, 1-indexed
    amgx_entries = []
    for node in range(1, n+1):
        deg = degree.get(node, 0.0)
        amgx_entries.append((node, node, deg))
    for i, j, w in edges:
        if i > j:
            amgx_entries.append((i, j, -w))
        else:
            amgx_entries.append((j, i, -w))

    amgx_entries.sort()
    nnz_lower = len(amgx_entries)

    write_amgx_mtx(
        os.path.join(outdir_amgx, name + '.mtx'),
        n, nnz_lower, [], amgx_entries, rhs
    )

    print(f'  {name}: n={n}, edges={len(edges)}, nnz_lower={nnz_lower}')


def main():
    # Output directories
    outdir_hypre = os.path.join(DATA_DIR, 'hypre')
    outdir_amgx = os.path.join(DATA_DIR, 'amgx')
    os.makedirs(outdir_hypre, exist_ok=True)
    os.makedirs(outdir_amgx, exist_ok=True)

    # =========================================================================
    # Physics matrices (positive diagonal, negative off-diagonal)
    # =========================================================================
    physics_matrices = [
        ('parabolic_fem', os.path.join(PHYSICS_DIR, 'parabolic_fem/parabolic_fem.mtx'), False),
        ('ecology1',      os.path.join(PHYSICS_DIR, 'ecology1/ecology1.mtx'), False),
        ('ecology2',      os.path.join(PHYSICS_DIR, 'ecology2/ecology2.mtx'), False),
        ('apache2',       os.path.join(PHYSICS_DIR, 'apache2/apache2.mtx'), False),
        ('G3_circuit',    os.path.join(PHYSICS_DIR, 'G3_circuit/G3_circuit.mtx'), False),
        # SPE: has negative diagonal and positive off-diagonal, need to negate
        ('spe16m',        os.path.join(PHYSICS_DIR, 'spe16m/spe16m.mtx'), True),
    ]

    for name, path, negate in physics_matrices:
        if os.path.exists(path):
            process_physics(name, path, outdir_hypre, outdir_amgx, negate=negate)
        else:
            print(f'  WARNING: {path} not found, skipping {name}')

    # =========================================================================
    # Graph matrices (pattern -> Laplacian)
    # =========================================================================
    graph_matrices = [
        ('GAP-road',       os.path.join(DATA_DIR, 'GAP-road/GAP-road.mtx')),
        ('com-LiveJournal', os.path.join(DATA_DIR, 'com-LiveJournal/com-LiveJournal.mtx')),
        ('delaunay_n24',   os.path.join(DATA_DIR, 'delaunay_n24/delaunay_n24.mtx')),
        ('venturiLevel3',  os.path.join(DATA_DIR, 'venturiLevel3/venturiLevel3.mtx')),
        ('europe_osm',    os.path.join(DATA_DIR, 'europe_osm/europe_osm.mtx')),
        ('belgium_osm',   os.path.join(DATA_DIR, 'belgium_osm/belgium_osm.mtx')),
    ]

    for name, path in graph_matrices:
        if os.path.exists(path):
            process_graph(name, path, outdir_hypre, outdir_amgx)
        else:
            print(f'  WARNING: {path} not found, skipping {name}')

    # =========================================================================
    # Note: uniform_3D, aniso_contrast_3D, contrast_3D_laplace are generated
    # by Julia scripts, not downloaded. Add them here once generated.
    # =========================================================================
    julia_physics = [
        ('uniform_3D',          os.path.join(PHYSICS_DIR, 'uniform_3D/uniform_3D.mtx')),
        ('aniso_contrast_3D',   os.path.join(PHYSICS_DIR, 'aniso_contrast_3D/aniso_contrast_3D.mtx')),
        ('poisson_contrast_3D', os.path.join(PHYSICS_DIR, 'poisson_contrast_3D/poisson_contrast_3D.mtx')),
    ]

    for name, path in julia_physics:
        if os.path.exists(path):
            process_physics(name, path, outdir_hypre, outdir_amgx)
        else:
            print(f'  NOTE: {path} not found (generate with Julia first), skipping {name}')

    print('\nDone. Output in:')
    print(f'  hypre: {outdir_hypre}/')
    print(f'  amgx:  {outdir_amgx}/')


if __name__ == '__main__':
    main()

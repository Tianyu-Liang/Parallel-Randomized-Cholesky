"""Convert a real symmetric MTX Laplacian to hypre IJ format.
Assumes the MTX already has diagonal + off-diagonal entries (not pattern)."""
import sys

infile = sys.argv[1]
outfile = sys.argv[2]

entries = []
n = None

with open(infile) as f:
    for line in f:
        if line.startswith('%'):
            continue
        if n is None:
            parts = line.strip().split()
            n = int(parts[0])
            continue
        parts = line.strip().split()
        i, j, v = int(parts[0]), int(parts[1]), float(parts[2])
        entries.append((i - 1, j - 1, v))  # convert to 0-indexed

# Write IJ format
with open(outfile, 'w') as out:
    out.write(f'0 {n - 1}\n')
    out.write(f'0 {n - 1}\n')
    for i, j, v in entries:
        out.write(f'{i} {j} {v}\n')
        if i != j:  # symmetric: write both triangles
            out.write(f'{j} {i} {v}\n')

# Write RHS (alternating +1/-1, sum=0)
with open(outfile + '.rhs', 'w') as out:
    out.write(f'0 {n - 1}\n')
    for node in range(n):
        if n % 2 == 1 and node == n - 1:
            val = 0.0
        elif node % 2 == 0:
            val = 1.0
        else:
            val = -1.0
        out.write(f'{node} {val}\n')

print(f'Done: {n} rows, wrote {outfile} and {outfile}.rhs')

import sys
from collections import defaultdict

infile = sys.argv[1]   # original pattern mtx
outfile = sys.argv[2]  # output IJ format

# First pass: read edges and compute degrees
edges = []
degree = defaultdict(int)
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
        i, j = int(parts[0]), int(parts[1])
        edges.append((i, j))
        degree[i] += 1
        if i != j:
            degree[j] += 1

# Write IJ format (0-indexed)
# Matrix file
with open(outfile, 'w') as out:
    out.write(f'0 {n - 1}\n')
    out.write(f'0 {n - 1}\n')
    # Diagonal + off-diagonal (both lower and upper triangle, 0-indexed)
    for node in range(1, n + 1):
        deg = degree.get(node, 0)
        out.write(f'{node - 1} {node - 1} {float(deg)}\n')
    for i, j in edges:
        if i != j:
            out.write(f'{i - 1} {j - 1} -1.0\n')
            out.write(f'{j - 1} {i - 1} -1.0\n')  # symmetric: store both

# Write RHS file (alternating +1/-1, sum=0)
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

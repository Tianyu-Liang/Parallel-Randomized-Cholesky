import sys
from collections import defaultdict

infile = sys.argv[1]
outfile = sys.argv[2]

# First pass: read edges and compute degrees
edges = []
degree = defaultdict(int)
header_lines = []
size_line = None

with open(infile) as f:
    for line in f:
        if line.startswith('%'):
            header_lines.append(line)
            continue
        if size_line is None:
            size_line = line
            parts = line.strip().split()
            n = int(parts[0])
            continue
        parts = line.strip().split()
        i, j = int(parts[0]), int(parts[1])
        edges.append((i, j))
        degree[i] += 1
        if i != j:  # symmetric: j also gets degree contribution
            degree[j] += 1

# New nnz = original edges (off-diag, stored as lower triangle) + n diagonal entries
n_diag = n
new_nnz = len(edges) + n_diag

with open(outfile, 'w') as out:
    # Write header, changing 'pattern' to 'real'
    for line in header_lines:
        if line.startswith('%%MatrixMarket'):
            out.write(line.replace('pattern', 'real'))
            out.write('%%AMGX rhs\n')
        elif line.startswith('%%'):
            out.write(line)
        else:
            out.write(line)
    # Write size line with new nnz
    parts = size_line.strip().split()
    out.write(f'{parts[0]} {parts[1]} {new_nnz}\n')
    
    # Write diagonal entries first
    for node in range(1, n + 1):
        deg = degree.get(node, 0)
        out.write(f'{node} {node} {float(deg)}\n')
    
    # Write off-diagonal entries as -1.0
    for i, j in edges:
        if i != j:
            out.write(f'{i} {j} -1.0\n')

    # Append RHS vector perpendicular to all-ones vector (sum = 0)
    # Use alternating +1/-1; if n is odd, set last entry to 0
    out.write(f'{n}\n')
    for node in range(1, n + 1):
        if n % 2 == 1 and node == n:
            out.write('0.0\n')
        elif node % 2 == 1:
            out.write('1.0\n')
        else:
            out.write('-1.0\n')

print(f'Done: {n} rows, {len(edges)} off-diag edges, {new_nnz} total nnz')

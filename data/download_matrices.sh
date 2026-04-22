#!/bin/bash
# Download matrices for graph sparsification experiments.
# Run from the data/ directory: bash download_matrices.sh
#
# Matrices are split into three categories:
#   1. Graph Laplacians (from SuiteSparse, pattern format - need Laplacian conversion)
#   2. Physics matrices (from SuiteSparse, already have values)
#   3. SPE matrix (from Dropbox)
#
# After downloading, use the Julia scripts in cpu_implementation/ to generate
# Laplacian versions with AMD/nnz-sort orderings.

set -e
cd "$(dirname "$0")"
DATA_DIR=$(pwd)

echo "=== Downloading matrices to $DATA_DIR ==="

###############################################################################
# 1. Graph Laplacians from SuiteSparse (pattern symmetric)
#    These are adjacency graphs. Use write_graph.jl to convert to Laplacians.
#    Already downloaded: belgium_osm, europe_osm, com-LiveJournal, GAP-road,
#                        delaunay_n24, venturiLevel3
###############################################################################

echo ""
echo "--- Graph matrices (already present, skipping) ---"
for mat in belgium_osm europe_osm com-LiveJournal GAP-road delaunay_n24 venturiLevel3; do
    if [ -d "$mat" ] && ls "$mat"/*.mtx &>/dev/null; then
        echo "  $mat: already exists, skipping"
    else
        echo "  WARNING: $mat directory missing or has no .mtx files"
    fi
done

###############################################################################
# 2. Physics matrices from SuiteSparse
#    These already have real values (stiffness/FEM matrices).
#    Downloaded to physics/ directory, then processed by produce_physics_amd.jl
###############################################################################

PHYSICS_DIR="../physics"

echo ""
echo "--- Physics matrices from SuiteSparse ---"

# ecology1 - McRae/ecology1
if [ ! -f "$PHYSICS_DIR/ecology1/ecology1.mtx" ]; then
    echo "Downloading ecology1..."
    wget -q https://suitesparse-collection-website.herokuapp.com/MM/McRae/ecology1.tar.gz -O ecology1.tar.gz
    tar xzf ecology1.tar.gz -C "$PHYSICS_DIR/"
    rm ecology1.tar.gz
    echo "  ecology1 done"
else
    echo "  ecology1: already exists, skipping"
fi

# ecology2 - McRae/ecology2
if [ ! -f "$PHYSICS_DIR/ecology2/ecology2.mtx" ]; then
    echo "Downloading ecology2..."
    wget -q https://suitesparse-collection-website.herokuapp.com/MM/McRae/ecology2.tar.gz -O ecology2.tar.gz
    tar xzf ecology2.tar.gz -C "$PHYSICS_DIR/"
    rm ecology2.tar.gz
    echo "  ecology2 done"
else
    echo "  ecology2: already exists, skipping"
fi

# apache2 - GHS_psdef/apache2
if [ ! -f "$PHYSICS_DIR/apache2/apache2.mtx" ]; then
    echo "Downloading apache2..."
    wget -q https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/apache2.tar.gz -O apache2.tar.gz
    tar xzf apache2.tar.gz -C "$PHYSICS_DIR/"
    rm apache2.tar.gz
    echo "  apache2 done"
else
    echo "  apache2: already exists, skipping"
fi

# G3_circuit - AMD/G3_circuit
if [ ! -f "$PHYSICS_DIR/G3_circuit/G3_circuit.mtx" ]; then
    echo "Downloading G3_circuit..."
    wget -q https://suitesparse-collection-website.herokuapp.com/MM/AMD/G3_circuit.tar.gz -O G3_circuit.tar.gz
    tar xzf G3_circuit.tar.gz -C "$PHYSICS_DIR/"
    rm G3_circuit.tar.gz
    echo "  G3_circuit done"
else
    echo "  G3_circuit: already exists, skipping"
fi

# parabolic_fem - Wissgott/parabolic_fem
if [ ! -f "$PHYSICS_DIR/parabolic_fem/parabolic_fem.mtx" ]; then
    echo "Downloading parabolic_fem..."
    wget -q https://suitesparse-collection-website.herokuapp.com/MM/Wissgott/parabolic_fem.tar.gz -O parabolic_fem.tar.gz
    tar xzf parabolic_fem.tar.gz -C "$PHYSICS_DIR/"
    rm parabolic_fem.tar.gz
    echo "  parabolic_fem done"
else
    echo "  parabolic_fem: already exists, skipping"
fi

###############################################################################
# 3. SPE matrix (from Dropbox)
#    Downloaded to physics/spe/
###############################################################################

echo ""
echo "--- SPE matrix from Dropbox ---"

if [ ! -f "$PHYSICS_DIR/spe16m/spe16m.mtx" ]; then
    echo "Downloading spe.zip from Dropbox..."
    wget -q "https://www.dropbox.com/scl/fi/hio4eifrxpfoaduv1l922/spe.zip?rlkey=uzi1uujiqy9nmj058tjgxsu3m&e=1&dl=1" -O spe.zip
    unzip -o spe.zip -d "$PHYSICS_DIR/spe/"
    # Move each .mm file into its own directory as .mtx
    for f in "$PHYSICS_DIR/spe/"*.mm; do
        name=$(basename "$f" .mm)
        mkdir -p "$PHYSICS_DIR/$name"
        cp "$f" "$PHYSICS_DIR/$name/$name.mtx"
    done
    rm spe.zip
    echo "  spe done (split into spe0.5m, spe2m, spe4m, spe8m, spe16m)"
else
    echo "  spe: already exists, skipping"
fi

###############################################################################
# Summary
###############################################################################

echo ""
echo "=== Download complete ==="
echo ""
echo "Matrices ready in physics/:"
for d in parabolic_fem ecology1 ecology2 apache2 G3_circuit; do
    if [ -f "$PHYSICS_DIR/$d/$d.mtx" ]; then
        echo "  $d: OK"
    else
        echo "  $d: MISSING"
    fi
done
echo ""
echo "SPE:"
for s in spe0.5m spe2m spe4m spe8m spe16m; do
    if [ -f "$PHYSICS_DIR/$s/$s.mtx" ]; then
        echo "  $s: OK"
    else
        echo "  $s: MISSING"
    fi
done
echo ""
echo "Graph matrices in data/:"
for d in belgium_osm europe_osm com-LiveJournal GAP-road delaunay_n24 venturiLevel3; do
    if ls "$DATA_DIR/$d/"*.mtx &>/dev/null; then
        echo "  $d: OK"
    else
        echo "  $d: MISSING"
    fi
done
echo ""
echo "Note: uniform_3D, aniso_contrast_3D, and contrast_3D_laplace are generated"
echo "by the Julia scripts (produce_physics_amd.jl), not downloaded."

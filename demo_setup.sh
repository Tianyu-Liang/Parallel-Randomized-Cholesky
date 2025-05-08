#!/bin/bash 

mkdir physics
cd physics
DATASETS=("parabolic_fem" "ecology1" "ecology2" "apache2" "G3_circuit" "uniform_3D" "poisson_contrast_3D" "aniso_contrast_3D")
LINKS=("https://suitesparse-collection-website.herokuapp.com/MM/Wissgott/parabolic_fem.tar.gz" "https://suitesparse-collection-website.herokuapp.com/MM/McRae/ecology1.tar.gz" "https://suitesparse-collection-website.herokuapp.com/MM/McRae/ecology2.tar.gz" "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/apache2.tar.gz" "https://suitesparse-collection-website.herokuapp.com/MM/AMD/G3_circuit.tar.gz")
for i in {0..7}
do
    dataset=${DATASETS[i]}
    mkdir $dataset
    if(($i < 5)); then
        cd $dataset
        link=${LINKS[i]}
        wget $link
        filename1="${dataset}.tar.gz"
        filename2="${dataset}.mtx"
        tar -xzvf $filename1
        mv $dataset/$filename2 .
        rm $filename1
        #rm -rf $dataset
        cd ..
        #echo $i
    fi 
done

cd ../cpu_implementation
echo "datasets downloaded. now processing them for demo experiment"
julia produce_physics_nnz_sort.jl

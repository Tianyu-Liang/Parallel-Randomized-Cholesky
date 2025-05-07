using SparseArrays
using MatrixMarket
using LinearAlgebra
using AMD
using Random
using Laplacians

function remove_diagonal(G)
    I = Vector{Int64}()
    J = Vector{Int64}()
    V = Vector{Float64}()
    for i = 1 : length(G.colptr) - 1
        for j = G.colptr[i] : G.colptr[i + 1] - 1
            if G.rowval[j] != i
                push!(I, G.rowval[j])
                push!(J, i)
                push!(V, G.nzval[j])
            end
        end
    end
    G = sparse(I,J,V)
    return G
end



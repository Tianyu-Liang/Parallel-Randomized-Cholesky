using SparseArrays
using MatrixMarket
using LinearAlgebra
using AMD
using Random
using Laplacians
using Metis
#include("../gen_mat/julia_gen.jl")

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

function graph_produce(path::String, method::String)
    
    G = MatrixMarket.mmread(path * ".mtx")
    G = SparseMatrixCSC{Float64, Int64}(G)
    G = remove_diagonal(G)
    if !issymmetric(G)
        println("not symmetric, warning")
        G = tril(G) + tril(G)'
    end
    #G.nzval[:] .= 1.0 # set all nonzeros to 1
    G.nzval .= -abs.(G.nzval)
    
    graph_share(path, G, method)
   
end

function graph_share(path::String, G::SparseMatrixCSC, method::String)

    G = spdiagm(-sum(G, dims=1)[:]) + G
    if method == "nnz-sort"
        p = randperm(size(G, 1))
        G = G[p, p]

        #G.nzval .= abs.(G.nzval)
        nnz_col = Vector{Int64}()
        for i = 1 : size(G, 1)
            push!(nnz_col, G.colptr[i + 1] - G.colptr[i])
        end
        #p = 1 : size(G, 1)
        t1 = time()
        p = sortperm(nnz_col)
        println("sort time: ", time() - t1)
        #p = amd(G)
        G_new = G[p, p]
    elseif method == "amd"
        t1 = time()
        p = amd(G)
        println("amd time: ", time() - t1)
        G_new = G[p, p]
    elseif method == "random"
        t1 = time()
        p = randperm(size(G, 1))
        println("random time: ", time() - t1)
        G_new = G[p, p]
    elseif method == "nd"
        t1 = time()
        p, ip = Metis.permutation(G)
        println("nd time: ", time() - t1)
        G_new = G[p, p]
    else
        throw("specified permutation not implemented")
    end

    if method == "nnz-sort"
        MatrixMarket.mmwrite(path * "-nnz-sorted.mtx", G_new)
    elseif method == "amd"

        MatrixMarket.mmwrite(path * "-amd.mtx", G_new)
    elseif method == "random"
        MatrixMarket.mmwrite(path * "-rand.mtx", G_new)
    elseif method == "nd"
        MatrixMarket.mmwrite(path * "-nd.mtx", G_new)
    else
        throw("specified permutation not implemented")
    end

    println(path * " finished")
    GC.gc()
end


function physics_produce(path::String, method::String; negate::Bool=false)
    G = MatrixMarket.mmread(path * ".mtx")
    G = SparseMatrixCSC{Float64, Int64}(G)
    # G = remove_diagonal(G)
    # if !issymmetric(G)
    #     G = tril(G) + tril(G)'
    # end
    #G.nzval[:] .= 1.0 # set all nonzeros to 1

    if negate
        G .= G .* -1
    end


    if method == "nnz-sort"
        p = randperm(size(G, 1))
        G = G[p, p]

        #G.nzval .= abs.(G.nzval)
        nnz_col = Vector{Int64}()
        for i = 1 : size(G, 1)
            push!(nnz_col, G.colptr[i + 1] - G.colptr[i])
        end
        #p = 1 : size(G, 1)
        t1 = time()
        
        
        p = sortperm(nnz_col)
        println("sort time: ", time() - t1)
        #p = amd(G)
        G_new = G[p, p]
    elseif method == "amd"
        t1 = time()
        p = amd(G)
        println("amd time: ", time() - t1)
        G_new = G[p, p]
    elseif method == "random"
        t1 = time()
        p = randperm(size(G, 1))
        println("random time: ", time() - t1)
        G_new = G[p, p]
    elseif method == "nd"
        t1 = time()
        p, ip = Metis.permutation(G)
        println("nd time: ", time() - t1)
        G_new = G[p, p]
    else
        throw("specified permutation not implemented")
    end

    check_sum = sum(G_new)
    if check_sum < 0 && abs(check_sum) > 1e-9
        println("not diagonally dominant")
        @assert false
    end
    if check_sum > 1e-9
        println("not directly laplacian, append to make laplacian")
        col_append = -sum(G_new, dims=1)
        idx = findall(x -> x > 0, col_append)
        col_append[idx] .= 0
        

        G_new = vcat(G_new, col_append)
        last_sum = -sum(col_append)
        G_new = hcat(G_new, [col_append'; last_sum])
    end

    if negate
        G_new .= G_new .* -1
    end


    if method == "nnz-sort"
        MatrixMarket.mmwrite(path * "-nnz-sorted.mtx", G_new)
    elseif method == "amd"

        MatrixMarket.mmwrite(path * "-amd.mtx", G_new)
    elseif method == "random"
        MatrixMarket.mmwrite(path * "-rand.mtx", G_new)
    elseif method == "nd"
        MatrixMarket.mmwrite(path * "-nd", G_new)
    else
        throw("specified permutation not implemented")
    end
    
end

function physics_produce_no_append(path::String, method::String; negate::Bool=false)
    G = MatrixMarket.mmread(path * ".mtx")
    G = SparseMatrixCSC{Float64, Int64}(G)
    # G = remove_diagonal(G)
    # if !issymmetric(G)
    #     G = tril(G) + tril(G)'
    # end
    #G.nzval[:] .= 1.0 # set all nonzeros to 1




    if method == "nnz-sort"
        p = randperm(size(G, 1))
        G = G[p, p]

        #G.nzval .= abs.(G.nzval)
        nnz_col = Vector{Int64}()
        for i = 1 : size(G, 1)
            push!(nnz_col, G.colptr[i + 1] - G.colptr[i])
        end
        #p = 1 : size(G, 1)
        t1 = time()
        
        
        p = sortperm(nnz_col)
        println("sort time: ", time() - t1)
        #p = amd(G)
        G_new = G[p, p]
    elseif method == "amd"
        t1 = time()
        p = amd(G)
        println("amd time: ", time() - t1)
        G_new = G[p, p]
    elseif method == "random"
        t1 = time()
        p = randperm(size(G, 1))
        println("random time: ", time() - t1)
        G_new = G[p, p]
    elseif method == "nd"
        t1 = time()
        p, ip = Metis.permutation(G)
        println("nd time: ", time() - t1)
        G_new = G[p, p]
    else
        throw("specified permutation not implemented")
    end

    



    if method == "nnz-sort"
        MatrixMarket.mmwrite(path * "-bare-nnz-sorted.mtx", G_new)
    elseif method == "amd"

        MatrixMarket.mmwrite(path * "-bare-amd.mtx", G_new)
    elseif method == "random"
        MatrixMarket.mmwrite(path * "-bare-rand.mtx", G_new)
    elseif method == "nd"
        MatrixMarket.mmwrite(path * "-bare-nd.mtx", G_new)
    else
        throw("specified permutation not implemented")
    end
    
end




function uniform_produce(path::String, n, method::String; append_decision::Bool=true)
    G = uniform_grid_sddm(n)
    # G = remove_diagonal(G)
    # if !issymmetric(G)
    #     G = tril(G) + tril(G)'
    # end
    #G.nzval[:] .= 1.0 # set all nonzeros to 1


    physics_share(path, G, method; append_last=append_decision)
    
end



# Anisotropic coefficient 3D cube with fixed discretization and variable weight
function poisson_contrast_produce(path::String, n, grid_len, constrast_factor, method::String; append_decision::Bool=true)

    G = checkered_grid_sddm(n, grid_len, grid_len, grid_len, constrast_factor)
    # G = remove_diagonal(G)
    # if !issymmetric(G)
    #     G = tril(G) + tril(G)'
    # end
    #G.nzval[:] .= 1.0 # set all nonzeros to 1


    physics_share(path, G, method; append_last=append_decision)

end



# Anisotropic coefficient 3D cube with fixed discretization and variable weight
function aniso_contrast_produce(path::String, n, constrast_factor, method::String; append_decision::Bool=true)
    G = wgrid_sddm(n, constrast_factor)
    # G = remove_diagonal(G)
    # if !issymmetric(G)
    #     G = tril(G) + tril(G)'
    # end
    #G.nzval[:] .= 1.0 # set all nonzeros to 1

    physics_share(path, G, method; append_last=append_decision)
  
end

# Anisotropic coefficient 3D cube with fixed discretization and variable weight
function uniform_chimera(path::String, n, index, method::String)
    G = uni_chimera(n, index)
    # G = remove_diagonal(G)
    # if !issymmetric(G)
    #     G = tril(G) + tril(G)'
    # end
    #G.nzval[:] .= 1.0 # set all nonzeros to 1
    G.nzval .= -abs.(G.nzval)

    graph_share(path, G, method)
  
end

function sine_cos(path::String, n, method::String)
    G = gen_variable_coefficient(sqrt(n));

    physics_share(path, G, method)

end

function poisson_squared(path::String, n, method::String)
    G = gen_squared_coefficient(sqrt(n));

    physics_share(path, G, method)

end

function physics_share(path::String, G::SparseMatrixCSC, method::String; append_last::Bool=true)

    println("size: ", size(G, 1))


    if method == "nnz-sort"
        p = randperm(size(G, 1))
        G = G[p, p]

        #G.nzval .= abs.(G.nzval)
        nnz_col = Vector{Int64}()
        for i = 1 : size(G, 1)
            push!(nnz_col, G.colptr[i + 1] - G.colptr[i])
        end
        #p = 1 : size(G, 1)
        t1 = time()
        p = sortperm(nnz_col)
        println("sort time: ", time() - t1)
        #p = amd(G)
        G_new = G[p, p]
    elseif method == "amd"
        t1 = time()
        p = amd(G)
        println("amd time: ", time() - t1)
        G_new = G[p, p]
    elseif method == "random"
        t1 = time()
        p = randperm(size(G, 1))
        println("random time: ", time() - t1)
        G_new = G[p, p]
    elseif method == "nd"
        t1 = time()
        p, ip = Metis.permutation(G)
        println("nd time: ", time() - t1)
        G_new = G[p, p]
    else
        throw("specified permutation not implemented")
    end

    if(append_last)
        check_sum = sum(G_new)
        if check_sum < 0 && abs(check_sum) > 1e-4
            println("not diagonally dominant")
            @assert false
        end
        if check_sum > 1e-9
            println("not directly laplacian, append to make laplacian")
            col_append = -sum(G_new, dims=1)
            idx = findall(x -> x > 0, col_append)
            col_append[idx] .= 0
            

            G_new = vcat(G_new, col_append)
            last_sum = -sum(col_append)
            G_new = hcat(G_new, [col_append'; last_sum])
        end
    end

    if(append_last)
        if method == "nnz-sort"
            MatrixMarket.mmwrite(path * "-nnz-sorted.mtx", G_new)
        elseif method == "amd"

            MatrixMarket.mmwrite(path * "-amd.mtx", G_new)
        elseif method == "random"
            MatrixMarket.mmwrite(path * "-rand.mtx", G_new)
        elseif method == "nd"
            MatrixMarket.mmwrite(path * "-nd.mtx", G_new)
        else
            throw("specified permutation not implemented")
        end
    else
        if method == "nnz-sort"
            MatrixMarket.mmwrite(path * "-bare-nnz-sorted.mtx", G_new)
        elseif method == "amd"

            MatrixMarket.mmwrite(path * "-bare-amd.mtx", G_new)
        elseif method == "random"
            MatrixMarket.mmwrite(path * "-bare-rand.mtx", G_new)
        elseif method == "nd"
            MatrixMarket.mmwrite(path * "-bare-nd.mtx", G_new)
        else
            throw("specified permutation not implemented")
        end
    end

end
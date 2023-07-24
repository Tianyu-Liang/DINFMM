__precompile__()
module kernel_2D
using Distributed
@everywhere using Distributed
@everywhere using DistributedArrays
@everywhere using Cubature
@everywhere using VectorizedRoutines
@everywhere using LinearAlgebra
@everywhere using FFTW
@everywhere using LowRankApprox
@everywhere using Morton
@everywhere using LoopVectorization
@everywhere using Dates
@everywhere using ThreadPools
@everywhere using IterativeSolvers

@everywhere const BOXES_PER_NODE = 4
@everywhere const DIMENSION = 2
@everywhere const PRECISION = Float64
@everywhere const ERROR = 1e-6

# the following can potentially change at each layer
@everywhere global TOTAL_PTS = 0



include("rsfactor.jl")
using .rsfactor
# include("ano.jl")
# using .ano

@everywhere function evaluate_diagonal(N)
    n = sqrt(N)
    h = 1 / n
    return 4 * hcubature(x -> -1 / (2 * pi) * log(sqrt(x[1]^2 + x[2]^2)), [0, 0],[h / 2, h / 2])[1]
end

# Defines the pointwise interaction kernel
@everywhere function form_kernel!(x1::AbstractArray{T, 2}, y1::AbstractArray{T, 2}, A::AbstractArray{T, 2}) where T <: AbstractFloat
    # copy into x and y to avoid time penalty of accessing distributed array

    x = zeros(size(x1))
    y = zeros(size(y1))
    x .= x1
    y .= y1

    x_size = size(x, 2)
    y_size = size(y, 2)


    coefficient = -1 / (2 * pi)
    # can use multithread here
    # @inbounds for c = 1 : y_size
    #     @inbounds @avx for r = 1 : x_size
    #         A[r, c] = coefficient * log(sqrt((x[1, r] - y[1, c])^2  + (x[2, r] - y[2, c])^2))
    #     end
    # end
    @inbounds @avx for c in 1 : y_size, r in 1 : x_size
        A[r, c] = coefficient * log(sqrt((x[1, r] - y[1, c])^2  + (x[2, r] - y[2, c])^2))
    end

    #return @. -1/(2*pi).*log.(sqrt.((x[1,:] - y[1,:]')^2 + (x[2,:] - y[2,:]')^2));
end

@everywhere function troll()
    @time @inbounds @sync for c = 1 : 10
        Threads.@spawn println(c, " on thread: ", Threads.threadid())
    end

    @time @inbounds Threads.@threads for c = 1 : 10
        println(c, " on thread: ", Threads.threadid())
    end

end



# internal function for evaluate kernel matrices at the given X index and Y index
@everywhere function kfun_internal!(points::AbstractArray{T, 2}, idx::Union{AbstractArray{Int64, 1}, AbstractArray{T,}}, idy::Union{AbstractArray{Int64, 1}, AbstractArray{T,}}, diagonal::AbstractFloat, A::AbstractArray{T, 2}) where T<:AbstractFloat
    @assert typeof(idx) <: AbstractRange || typeof(idx) <: Array{Int64, 1} || typeof(idx) <: Array{Float64,}
    @assert typeof(idy) <: AbstractRange || typeof(idy) <: Array{Int64, 1} || typeof(idy) <: Array{Float64,}
    N = size(points, 2)
    local_points = localpart(points)
    local_range = localindices(points)[2]
    sub_points_x = []
    sub_points_y = []

    # if local range is empty (which likely means function is called from master worker), use a place holder
    if isempty(local_range)
        local_range = [-1]
    end
    if !isempty(idx)
        # check to see if points are on current processer
        # THIS ASSUMES THAT IDX OR IDY ONLY COMES FROM ONE NODE
        # OTHERWISE THE FIRST POINT WOULDN'T BE INDICATIVE OF THE WHOLE IDX ARRAY
        if typeof(idx) <: Array{Float64,}
            sub_points_x = idx
        else
            if idx[1] >= local_range[1] && idx[1] <= local_range[end] && idx[end] >= local_range[1] && idx[end] <= local_range[end]
                sub_points_x = local_points[:, idx .- local_range[1] .+ 1]
            else
                sub_points_x = points[:, idx]
            end
        end
    else
        sub_points_x = zeros(DIMENSION, 0)
    end

    if !isempty(idy)
        if typeof(idy) <: Array{Float64,}
            sub_points_y = idy
        else
            if idy[1] >= local_range[1] && idy[1] <= local_range[end] && idy[end] >= local_range[1] && idy[end] <= local_range[end]
                sub_points_y = local_points[:, idy .- local_range[1] .+ 1]
            else
                sub_points_y = points[:, idy]
            end
        end
    else
        sub_points_y = zeros(DIMENSION, 0)
    end


    form_kernel!(sub_points_x, sub_points_y, A)
    #A = form_kernel(points[:, idx], points[:, idy])
    #A .= A ./ N

    @inbounds @avx for i in eachindex(A)
        A[i] = A[i] / TOTAL_PTS
    end
    # set diagonal to precomputed integral value
    if !(typeof(idx) <: Array{Float64,}) && !(typeof(idy) <: Array{Float64,})
        @inbounds for c = 1 : length(idy)
            @inbounds for r = 1 : length(idx)

                if idx[r] == idy[c]
                    A[r, c] = diagonal
                end
            end
        end
    end


end


function kfun1_internal(points, idx, idy, diagonal)
    N = size(points)[2];
    A = -1/(2*pi)*log.(sqrt.(broadcast(-, points[1, idx], points[1, idy]').^2 + broadcast(-, points[2, idx], points[2, idy]') .^2)) / N;
    X, Y = meshgrid(idx, idy);
    A[X .== Y] .= diagonal;
    return A;
end


function kfun2_internal(points, idx, idy, diagonal)
    N = size(points)[2]
    A = zeros(length(idx), length(idy))
    @inbounds for (c_index, c_value) in enumerate(idy)
        @inbounds for (r_index, r_value) in enumerate(idx)

            A[r_index, c_index] = -log(sqrt((points[1, r_value] - points[1, c_value])^2  + (points[2, r_value] - points[2, c_value])^2)) / (N * 2 * pi)
            # set to precomputed integral value
            if r_value == c_value
                A[r_index, c_index] = diagonal
            end
        end
    end

    return A
end


@everywhere function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end


# internal function for evaluating proxy
@everywhere function pfun_internal!(pxy_points::Array{T, 2}, index::Array{Int64, 1}, points::AbstractArray{T, 2}, A::Matrix) where T <: AbstractFloat

    # A = zeros(size(pxy_points, 2), length(index))
    # @inbounds for c = 1 : length(index)
    #     @inbounds for r = 1 : size(pxy_points, 2)
    #         A[r, c] = -1 / (2 * pi) * log(sqrt((pxy_points[1, r] - points[1, index[c]])^2  + (pxy_points[2, r] - points[2, index[c]])^2)) / N
    #     end
    # end
    local_points = localpart(points)
    local_range = localindices(points)[2]
    sub_points = []
    if !isempty(index)
        # check to see if points are on current processer
        # THIS ASSUMES THAT IDX OR IDY ONLY COMES FROM ONE NODE
        # OTHERWISE THE FIRST POINT WOULDN'T BE INDICATIVE OF THE WHOLE IDX ARRAY
        if index[1] >= local_range[1] && index[1] <= local_range[end]
            sub_points = local_points[:, index .- local_range[1] .+ 1]
        else
            sub_points = points[:, index]
        end
    else
        sub_points = zeros(DIMENSION, 0)
    end


    form_kernel!(pxy_points, sub_points, A)
    @inbounds @avx for i in eachindex(A)
        A[i] = A[i] / TOTAL_PTS
    end

end

#=
problem specific fast multiplication using FFT, can later be used to
verify answers or used in PCG
=#
@everywhere function fast_mul(points::AbstractArray{T, 2}, diagonal::AbstractFloat) where T <: AbstractFloat
    N = size(points, 2)
    n = Int64(sqrt(N))
    A = zeros(N, 1)
    kfun_internal!(points, 1:N, [1], diagonal, A)
    a = reshape(A, n, n)
    B = Matrix{Complex{Float64}}(undef, 2 * n - 1, 2 * n - 1)
    B .= 0
    B[1 : n, 1 : n] .= a
    B[1 : n, n + 1 : end] .= a[:, 2 : n]
    B[n + 1 : end, 1 : n] .= a[2 : n, :]
    B[n + 1 : end, n + 1 : end] .= a[2 : n, 2 : n]
    B[:, n + 1 : end] .= reverse(B[:, n + 1 : end], dims=2)
    B[n + 1 : end, :] .= reverse(B[n + 1 : end, :], dims=1)
    fft!(B)
    fft_mv(x) = fft_mv_(B, x)
    return fft_mv
end

@everywhere function fft_mv_(F::AbstractArray{T1, 2}, x::Vector{T2} where T2 <: AbstractFloat) where T1 <: Number
    N = length(x)
    n = Int64(sqrt(N))
    larger = Matrix{Complex{Float64}}(undef, 2 * n - 1, 2 * n - 1)
    larger .= 0
    larger[1 : n, 1 : n] .= reshape(x, n, n)
    fft!(larger)
    larger .= F .* larger
    ifft!(larger)
    y = reshape(larger[1 : n, 1 : n], N, 1)
    return y
end

@everywhere import LinearAlgebra.mul!
@everywhere function mul!(y::AbstractVector, A::Function, x::AbstractVector)
    temp = real.(A(x))
    temp = reshape(temp, length(temp))
    y .= temp
end

@everywhere import LinearAlgebra.ldiv!
@everywhere function ldiv!(y::AbstractVector, A::Function, x::AbstractVector)
    y .= A(x)
end

@everywhere function Base.:(size)(A::Function, dim::Int64)
    return TOTAL_PTS
end

#=
Storing unpermuted points in a distributed fashion, which can later be used
as a part of fft. Again, this is kernel specific
=#
@everywhere function default_assignment(tup, n)
    idx_range = tup[2]
    tup_len = length(idx_range)
    points = zeros(DIMENSION, tup_len)

    # find starting location
    left_bound = idx_range[1]
    left_num = floor(left_bound / n)
    left_remainder = mod(left_bound, n)
    if left_remainder != 0
        left_num += 1
    else
        left_remainder = n
    end


    # going 1 by 1 to fill in the points array
    for i = 1 : tup_len
        points[2, i] = left_num / n
        points[1, i] = left_remainder / n

        if left_remainder == n
            left_remainder = 1
            left_num += 1
        else
            left_remainder += 1
        end
    end

    return points

end


#=
distribute points to different processes based on process id
again, this is kernel specific
=#
@everywhere function init_coordinates(n::Int64, min_val::AbstractFloat, max_val::AbstractFloat, worker_id::Int64, total_workers::Int64)

    worker_coordinate = morton2cartesian(worker_id)
    box_length = (max_val - min_val) / (total_workers ^ (1 / DIMENSION)) # length of each box
    x_coordinate = worker_coordinate[1]
    y_coordinate = worker_coordinate[2]
    # box range
    box_x_range = [(x_coordinate - 1) * box_length, x_coordinate * box_length] .+ min_val
    box_y_range = [(y_coordinate - 1) * box_length, y_coordinate * box_length] .+ min_val

    # assume the default (1:n)/n point coordinates
    # identify points in x range
    x_left_bound = Int64(ceil(box_x_range[1] * n))
    x_right_bound = Int64(floor(box_x_range[2] * n))
    if box_x_range[1] * n == ceil(box_x_range[1] * n) && x_coordinate != 1
        x_left_bound += 1
    end


    # identify points in y box_x_range
    y_left_bound = Int64(ceil(box_y_range[1] * n))
    y_right_bound = Int64(floor(box_y_range[2] * n))
    if box_y_range[1] * n == ceil(box_y_range[1] * n) && y_coordinate != 1
        y_left_bound += 1
    end


    sect_X = (x_left_bound : x_right_bound) ./ n
    sect_Y = (y_left_bound : y_right_bound) ./ n
    X, Y = meshgrid(sect_X, sect_Y)
    points = zeros(DIMENSION, length(X))
    points .= cat(X[:], Y[:], dims = 2)'

    return points
end

#calculate the perumutation that goes from original_coordinates to default_points
@everywhere function point_permutation_recovery(n::Int64, min_val::AbstractFloat, max_val::AbstractFloat, worker_id::Int64, total_workers::Int64)
    worker_coordinate = morton2cartesian(worker_id)
    box_length = (max_val - min_val) / (total_workers ^ (1 / DIMENSION)) # length of each box
    x_coordinate = worker_coordinate[1]
    y_coordinate = worker_coordinate[2]
    # box range
    box_x_range = [(x_coordinate - 1) * box_length, x_coordinate * box_length] .+ min_val
    box_y_range = [(y_coordinate - 1) * box_length, y_coordinate * box_length] .+ min_val

    # assume the default (1:n)/n point coordinates
    # identify points in x range
    x_left_bound = Int64(ceil(box_x_range[1] * n))
    x_right_bound = Int64(floor(box_x_range[2] * n))
    if box_x_range[1] * n == ceil(box_x_range[1] * n) && x_coordinate != 1
        x_left_bound += 1
    end


    # identify points in y box_x_range
    y_left_bound = Int64(ceil(box_y_range[1] * n))
    y_right_bound = Int64(floor(box_y_range[2] * n))
    if box_y_range[1] * n == ceil(box_y_range[1] * n) && y_coordinate != 1
        y_left_bound += 1
    end

    ret_p = zeros(Int64, (y_right_bound - y_left_bound + 1) * (x_right_bound - x_left_bound + 1))
    counter = 1
    for y = y_left_bound : y_right_bound
        for x = x_left_bound : x_right_bound
            ret_p[counter] = (y - 1) * n + x
            counter += 1
        end
    end

    return ret_p
    
end

@everywhere function rand_right(points::DArray)
    return rand(size(localpart(points), PART_TWO))
end


@everywhere function kernel_init(N::Int64, probes, use_MV::Bool=false)
    n = sqrt(N)
    @assert n * n == N
    n = Int64(n)
    intgrl = evaluate_diagonal(N)
    if length(probes) == 0
        probes = 1
    end
    array_parts = Array{Future, 1}(undef, length(probes))
    @sync for p in eachindex((probes))
        @async array_parts[p] = remotecall_wait(init_coordinates, probes[p], n, 1 / n, 1.0, p, length(probes))
    end
    array_parts = reshape(array_parts, (1, length(probes)))
    points = DArray(array_parts)
    
    default_kfun = kfun_internal! #used for FFT because it's not permuted
    kfun = kfun_internal!
    pfun = pfun_internal!

    MV = Nothing
    recovered_point_permutation = Nothing
    
    if use_MV == true
        default_points = default_assignment((1:2, 1:N), n)
        MV = fast_mul(distribute(default_points, procs=[PART_ONE]), intgrl)
        array_parts = Array{Future, 1}(undef, length(probes))
        @sync for p in eachindex((probes))
            @async array_parts[p] = remotecall_wait(point_permutation_recovery, probes[p], n, 1 / n, 1.0, p, length(probes))
        end
        recovered_point_permutation = DArray(array_parts)
        recovered_point_permutation = convert(Array{Int64, 1}, recovered_point_permutation)
    end

    return kfun, pfun, MV, points, default_kfun, recovered_point_permutation
end



function start_point()
    N = 64^2
    kfun, pfun, MV, points, default_kfun, default_points = kernel_init(N)
    @time result = kfun(1:3,1:3)
    @time result = kfun(1:100,1:100)
    @time result = kfun(1:N,1:N)
    @time result1 = default_kfun(1:N, 1:N)

    #@time result = kfun(1:N, 1:N)
    #@time result1 = kfun1(points, 1:N, 1:N, intgrl)
    #@time result1 = kfun1(points, 1:N, 1:N, intgrl)
    #@time result1 = kfun1(points, 1:N, 1:N, intgrl)
    #@time result = kfun(1:N, 1:N)
    #@time result = kfun(points, [1, 4, 7], [2, 4, 8], intgrl)
    #println(pfun([0.0134 0.0115 0.31; 0.52 0.61 0.244], [3,6,9], points))
    #println(result)
    #@time pfun(rand(2, 10000), 1:N, points);
    #@time pfun(rand(2, 10000), 1:N, points);

    x = rand(N)
    y = result1 * x
    println(norm(y - MV(x)) / norm(y))

end

@everywhere function permute_solution(factor_darray::DArray, v::DArray, rightside::DArray)

    factor_nodes = localpart(factor_darray)
    rightside_nodes = localpart(v)
    num_points = length(localpart(rightside))
    # this offset is used to index into the local parts, IMPORTANT TO REMEMBER
    ret_vec = zeros(num_points)
    for i = 1 : length(rightside_nodes)
        f_node = factor_nodes[i]
        v_node = rightside_nodes[i]
        ret_vec[f_node.ske_index .- f_node.idx_range[1] .+ 1] = v_node.right_side[f_node.relative_ske_index]
        ret_vec[f_node.red_index .- f_node.idx_range[1] .+ 1] = v_node.right_side[f_node.relative_red_index]
    end

    return ret_vec
end

@everywhere function get_size(arr::DArray, other_arr::DArray)
    loc_arr = localpart(arr)
    loc_other_arr = localpart(other_arr)
    tot_size = 0
    for i = 1 : length(loc_arr)
        node = loc_arr[i]
        other_node = loc_other_arr[i]
       
        tot_size += (length(node.ske_index) * 8)
        tot_size += (length(node.red_index) * 8)
        tot_size += (length(node.relative_ske_index) * 8)
        tot_size += (length(node.relative_red_index) * 8)
        tot_size += (length(node.neighbor_idx) * 8)
    
        tot_size += size(other_node.X_RR, 1) * size(other_node.X_RR, 2) * 8
        tot_size += (length(other_node.X_RS) * 8)
        tot_size += (length(other_node.X_RN) * 8)
        tot_size += (length(other_node.X_SR) * 8)
        tot_size += (length(other_node.X_NR) * 8)
        tot_size += (length(other_node.T) * 8)
        
    end

    return tot_size
end

@everywhere function comm_reset()
    global comm_counter = 1
    global comm_solve = true
end

@everywhere function comm_set()
    global comm_counter = 1
    global comm_solve = false
end

@everywhere function announce_factorization_level(current_layer)
    println("-------------------------------- The current layer for factorization is: ", current_layer)
end

@everywhere function announce_forward_solve_level(current_layer)
    println("-------------------------------- The current layer for forward solve is: ", current_layer)
end

@everywhere function announce_backward_solve_level(current_layer)
    println("-------------------------------- The current layer for backward solve is: ", current_layer)
end

@everywhere function count_ske_points(cur_level, tree_darray, factor_darray, on_boundary)
    factor_nodes = localpart(factor_darray)
    tree_nodes = localpart(tree_darray)
    pt_count = 0
    box_count = 0
    for i in eachindex(factor_nodes)

        f_node = factor_nodes[i]
        t_node = tree_nodes[i]

        if t_node.boundary[1] == on_boundary
            pt_count += length(f_node.ske_index)
            box_count += 1
        end
    end
    # println("ske points count: ", pt_count)
    # println("box count: ", box_count)
    # println("average points per box: ", pt_count / box_count)
    return [pt_count; box_count]
end

@everywhere function factorize_and_solve(N::Int64, layer::Int64, min_box_per_machine=4, worker_list=workers(); run_solve::Bool=false)
    bt = time()
    #=
    hosts = []
    pids = []
    for i in worker_list
        host, pid = fetch(@spawnat i (gethostname(), getpid()))
        push!(hosts, host)
        push!(pids, pid)
    end
    println(gethostname())
    println(hosts)
    println(pids)
    =#

    # assume min coordinate and max coordinate are given
    #N = 512^2
    #layer = 6



    ON_BOUNDARY = 1
    OFF_BOUNDARY = 0
    min_point = 1 / (N ^ (1 / DIMENSION))
    max_point = 1.0
    use_MV = run_solve
    multiply_verify = run_solve


    # get worker information
    probes = worker_list

    if length(probes) == 0
        probes = [1]
    end
    num_probes = length(probes)

    pts = N
    global TOTAL_PTS = pts
    @sync for p in probes
        @async remotecall_wait(eval, p, :(global TOTAL_PTS = $pts))
    end
    #@everywhere const TOTAL_PTS = eval($pts)
    kfun, pfun, MV, original_coordinates, default_kfun, recovered_point_permutation = kernel_init(N, probes, use_MV)

    coordinates = original_coordinates
    intgrl = evaluate_diagonal(N)
    box_per = BOXES_PER_NODE
    println("set up time: ", time() - bt)

    @sync for p in eachindex(probes)
        @async remotecall_wait(comm_set, probes[p])
    end

    # deepest level for both problem layer and number of processes layer
    bt = time()
    probe_layer = Int64(log(box_per, num_probes))


    # create a vector of futures that contain individual tree parts
    layer_level = [0; cumsum(box_per .^ (0 : layer))]
    tree_parts = Array{DArray, 1}(undef, 0)
    factor_nodes = Array{DArray, 1}(undef, 0)
    modified_nodes = Array{DArray, 1}(undef, 0)
    x_matrix_nodes = Array{DArray, 1}(undef, 0)
    communication_nodes = Array{DArray, 1}(undef, 0)
    rightside_nodes = Array{DArray, 1}(undef, 0)
    store_size = Array{DArray, 1}(undef, 0)

    temp_right = Array{Future, 1}(undef, length(probes))
    @sync for p in eachindex(probes)
        @async temp_right[p] = remotecall_wait(rand_right, probes[p], original_coordinates)
    end
    generated_right = DArray(temp_right)

    # for cur_level = length(probe_level) : -1 : 2
    #     left_bound = probe_level[cur_level - 1] + 1
    #     right_bound = probe_level[cur_level]
    #     cur_num_probes = Int64(box_per ^ (cur_level - 2))
    #     tree_parts[left_bound : right_bound] = [@spawnat probes[p] create_tree([1 / (N ^ (1 / DIMENSION)), 1], p, layer - (length(probe_level) - cur_level), layer, cur_num_probes) for p = 1 : cur_num_probes]
    # end

    #tree_parts = [@spawnat probes[p] create_tree([1 / (N ^ (1 / DIMENSION)), 1], p, layer, layer, num_probes) for p = 1 : num_probes]

    # build tree and distribute it across different processes
    probe_num_progression = []
    for cur_level = 0 : layer
        cur_num_probes = max(box_per ^ (cur_level) / min_box_per_machine, 1)
        cur_num_probes = min(Int64(box_per ^ floor(log(box_per, cur_num_probes))), num_probes)
        push!(probe_num_progression, cur_num_probes)
        step_size = Int64(num_probes / cur_num_probes)
        probe_range = range(1, step=step_size, length=cur_num_probes)
        temp_future = Array{Future, 1}(undef, length(probe_range))
        @sync for p_idx in eachindex(probe_range)
            p = probe_range[p_idx]
            @async temp_future[p_idx] = remotecall_wait(create_tree, probes[p], [min_point, max_point], p_idx, cur_level, layer, cur_num_probes)
        end
        temp_future = reshape(temp_future, (1, length(temp_future)))
        temp_array = [DArray(temp_future)]
        append!(tree_parts, temp_array)
    end

    println(probe_num_progression)
    # build factor information tree
    for cur_level = 0 : layer
        cur_num_probes = max(box_per ^ (cur_level) / min_box_per_machine, 1)
        cur_num_probes = min(Int64(box_per ^ floor(log(box_per, cur_num_probes))), num_probes)
        step_size = Int64(num_probes / cur_num_probes)
        probe_range = range(1, step=step_size, length=cur_num_probes)
        temp_future = Array{Future, 1}(undef, length(probe_range))
        #1
        @sync for p_idx in eachindex(probe_range)
            p = probe_range[p_idx]
            @async temp_future[p_idx] = remotecall_wait(classify_points, probes[p], coordinates, p_idx, cur_level, layer,
            cur_num_probes, tree_parts[cur_level + 1][1].box_length[1], min_point * ones(DIMENSION))
        end
        temp_future = reshape(temp_future, (1, length(temp_future)))
        temp_array = [DArray(temp_future)]
        append!(factor_nodes, temp_array)
        #2
        @sync for p_idx in eachindex(probe_range)
            p = probe_range[p_idx]
            @async temp_future[p_idx] = remotecall_wait(create_other_info, probes[p], cur_level, cur_num_probes, 2)
        end
        temp_future = reshape(temp_future, (1, length(temp_future)))
        temp_array = [DArray(temp_future)]
        append!(modified_nodes, temp_array)
        #3
        @sync for p_idx in eachindex(probe_range)
            p = probe_range[p_idx]
            @async temp_future[p_idx] = remotecall_wait(create_other_info, probes[p], cur_level, cur_num_probes, 3)
        end
        temp_future = reshape(temp_future, (1, length(temp_future)))
        temp_array = [DArray(temp_future)]
        append!(x_matrix_nodes, temp_array)

        # create rightside
        @sync for p_idx in eachindex(probe_range)
            p = probe_range[p_idx]
            @async temp_future[p_idx] = remotecall_wait(generate_rightside, probes[p], factor_nodes[end], cur_level, layer, generated_right)
        end
        temp_future = reshape(temp_future, (1, length(temp_future)))
        temp_array = [DArray(temp_future)]
        append!(rightside_nodes, temp_array)

        # create array for storing size
        @sync for p_idx in eachindex(probe_range)
            p = probe_range[p_idx]
            @async temp_future[p_idx] = remotecall_wait(zeros, probes[p], Int64, 1, 1)
        end
        temp_future = reshape(temp_future, (1, length(temp_future)))
        temp_array = [DArray(temp_future)]
        append!(store_size, temp_array)
    end


    # build communication information tree
    for cur_level = 0 : layer
        cur_num_probes = max(box_per ^ (cur_level) / min_box_per_machine, 1)
        cur_num_probes = min(Int64(box_per ^ floor(log(box_per, cur_num_probes))), num_probes)
        step_size = Int64(num_probes / cur_num_probes)
        probe_range = range(1, step=step_size, length=cur_num_probes)
        temp_future = Array{Future, 1}(undef, length(probe_range))
        @sync for p_idx in eachindex(probe_range)
            p = probe_range[p_idx]
            @async temp_future[p_idx] = remotecall_wait(create_communication_struct, probes[p], cur_level, cur_num_probes)
        end
        temp_future = reshape(temp_future, (1, length(temp_future)))
        temp_array = [DArray(temp_future)]
        append!(communication_nodes, temp_array)
    end

    #GC.enable(false)
    println("build stuff time: ", time() - bt)

    let
        coordinates = original_coordinates
        returned_points = []
        prev_cur_num_probes = -1
        prev_probe_range = -1

        factorize_time = @timed for cur_level = layer : -1 : 0
            
            
            cur_num_probes = probe_num_progression[cur_level + 1]
            step_size = Int64(num_probes / cur_num_probes)
            probe_range = range(1, step=step_size, length=cur_num_probes)
            total_factor_nodes = [factor_nodes[cur_level + 1], modified_nodes[cur_level + 1], x_matrix_nodes[cur_level + 1]]
            
            # number of boxes in each machine at the current layer

            # if not at bottom level, apply transition first
            if cur_level != layer
                store_temp = store_size[cur_level + 2]
                nodes_below = [factor_nodes[cur_level + 2], modified_nodes[cur_level + 2], x_matrix_nodes[cur_level + 2]]
                tree_below = tree_parts[cur_level + 2]
                num_next_level = Int64(prev_cur_num_probes / cur_num_probes)
            #    println("aaaaaaaaaaaa", range(1, step=Int64(max(step_size / box_per, 1)), length=num_next_level))
            #    println("vvvvvvvvvvvv", probe_range)
                t1 = time()
                
                first_message = Array{Future, 1}(undef, length(probe_range))
                second_message = Array{Future, 1}(undef, length(prev_probe_range))
                if prev_cur_num_probes == cur_num_probes
                    second_message = zeros(length(prev_probe_range))
                end
                @sync for (p_idx, p) in enumerate(probe_range)
                    temp_probe_list = probes[range(p, step=Int64(max(step_size / box_per, 1)), length=num_next_level)]
                    rp_range = (p_idx - 1) * box_per + 1 : (p_idx - 1) * box_per + box_per
                    @async first_message[p_idx] = remotecall_wait(prefetch_points_level_transition, probes[p], tree_parts[cur_level + 1], nodes_below[PART_ONE], coordinates)
                    
                    if prev_cur_num_probes != cur_num_probes
                        for (j_idx, j) in enumerate(temp_probe_list)
                            @async second_message[rp_range[j_idx]] = remotecall_wait(prefetch_points, j, tree_below, nodes_below[PART_ONE], coordinates, ON_BOUNDARY)
                        end
                    end
                end

                println("transition fetch time: ", time() - t1)

                events = Array{Future, 1}(undef, length(probe_range))
                @sync for p_idx in eachindex(probe_range)
                    p = probe_range[p_idx]
                    rp_range = (p_idx - 1) * box_per + 1 : (p_idx - 1) * box_per + box_per
                    if prev_cur_num_probes == cur_num_probes
                        rp_range = p_idx
                    end
                    @async events[p_idx] = remotecall_wait(level_transition!, probes[p], tree_parts[cur_level + 1], tree_below, total_factor_nodes, nodes_below, cur_level, kfun,
                    probes[range(p, step=Int64(max(step_size / box_per, 1)), length=num_next_level)], coordinates, 
                        intgrl, returned_points[:, rp_range], store_temp, p_idx, prev_cur_num_probes != cur_num_probes, (first_message[p_idx], second_message[rp_range]))
                end

                returned_points = []
                #level_transition!(tree_parts[cur_level + 1], total_factor_nodes, nodes_below, cur_level, kfun, probes[1 : 1 + lower_range - 1], coordinates, intgrl)
                for fut in events
                    wait(fut)
                end
                # if there is a change in number of processes between levels, points need to be reorganized
                if prev_cur_num_probes != cur_num_probes
                    #close(coordinates)
                    
                    coordinates = DArray(reshape(events, (1, length(events))))
                    
                end

                # clear up memory
                @sync for p_idx in eachindex(prev_probe_range)
                    p = prev_probe_range[p_idx]
                    @async remotecall_wait(clear_auxilliary!, probes[p], communication_nodes[cur_level + 2], nodes_below)
                end
                println("transition total time: ", time() - t1)
            end
            prev_cur_num_probes = cur_num_probes
            prev_probe_range = probe_range


            if cur_level != 0
                @sync for p in eachindex(probes)
                    @async remotecall_wait(announce_factorization_level, probes[p], cur_level)
                end
                # pick out processes according to 4 coloring or 8 coloring
                subset_probes = Int64(max(cur_num_probes / box_per, 1))
                returned_points = Array{Future, 2}(undef, min(box_per, cur_num_probes), subset_probes)

                t1 = time()
                # prefetch points
                #prefetch_list = Array{Future, 2}(undef, min(box_per, cur_num_probes), subset_probes)
                #for coloring = 1 : min(box_per, cur_num_probes)
                #    launch_idx = range(coloring, stop=cur_num_probes, step=box_per)
                #    to_wait = [@spawnat probes[probe_range[launch_idx[p]]] prefetch_points(tree_parts[cur_level + 1], total_factor_nodes[1], coordinates, OFF_BOUNDARY) for p = 1 : subset_probes]
                #    for fut in to_wait
                #        wait(fut) # synchronization point
                #    end
                #    prefetch_list[coloring, :] .= to_wait
                #end
                #prefetch_list = reshape(prefetch_list, (subset_probes * min(box_per, cur_num_probes), ))
                prefetch_list = Array{Future, 1}(undef, length(probe_range))
                @sync for p_idx in eachindex(probe_range)
                    p = probe_range[p_idx]
                    @async prefetch_list[p_idx] = remotecall_wait(prefetch_points, probes[p], tree_parts[cur_level + 1], factor_nodes[cur_level + 1], coordinates, OFF_BOUNDARY)
                end
                println("interior fetch time: ", time() - t1)

                store_temp = store_size[cur_level + 1]
                #fetch(@spawnat 2 perform_factorization!(tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], cur_level, kfun, pfun, OFF_BOUNDARY, coordinates, intgrl, store_temp))

                # @sync begin
                #    for p in probe_range
                #        @async remotecall_wait(perform_factorization!, probes[p], tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], cur_level, kfun, pfun, OFF_BOUNDARY, coordinates, intgrl, store_temp, prefetch_list[p])
                #    end
                # end

                # for (p_idx, p) in enumerate(probe_range)
                #     event = @spawnat probes[p] perform_factorization!(tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], cur_level, kfun, pfun, OFF_BOUNDARY, coordinates, intgrl, store_temp, prefetch_list[p_idx])
                #     fetch(event)
                # end
                #perform_factorization!(tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], cur_level, kfun, pfun, OFF_BOUNDARY, coordinates, intgrl)

                @time @sync for (p_idx, p) in enumerate(probe_range)
                    @async remotecall_wait(perform_factorization!, probes[p], tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], cur_level, kfun, pfun, OFF_BOUNDARY, coordinates, intgrl, store_temp, prefetch_list[p_idx])
                end
                
                println("interior factorization time for level ", cur_level, " : ", time() - t1)

                ske_points_counter = Array{Future, 1}(undef, length(probe_range))
                pt_box_ct = zeros(2)
                @sync for (p_idx, p) in enumerate(probe_range)
                    @async ske_points_counter[p_idx] = remotecall_wait(count_ske_points, probes[p], cur_level, tree_parts[cur_level + 1], total_factor_nodes[PART_ONE], OFF_BOUNDARY)
                end
                for p in eachindex(ske_points_counter)
                    count_info = fetch(ske_points_counter[p])
                    pt_box_ct[1] += count_info[1]
                    pt_box_ct[2] += count_info[2]
                end
                println("total interior boxes: ", pt_box_ct[2])
                println("total ske points: ", pt_box_ct[1])
                println("average pts per box: ", pt_box_ct[1] / pt_box_ct[2])

                for p in probe_range
                    #remotecall_fetch(GC.gc, p)
                end

                s1 = 0
                s2 = 0
                for coloring = 1 : min(box_per, cur_num_probes)
                    launch_idx = range(coloring, stop=cur_num_probes, step=box_per)
                    t1 = time()
                    prefetch_list = Array{Future, 1}(undef, length(probe_range))
                    @sync for p = 1 : subset_probes
                        @async prefetch_list[p] = remotecall_wait(prefetch_points, probes[probe_range[launch_idx[p]]], tree_parts[cur_level + 1], factor_nodes[cur_level + 1], coordinates, ON_BOUNDARY)
                    end
                    println("exterior fetch time: ", time() - t1)
                    to_wait = Array{Future, 1}(undef, subset_probes)
                    @time @sync for p = 1 : subset_probes
                        @async to_wait[p] = remotecall_wait(perform_factorization!, probes[probe_range[launch_idx[p]]], tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], cur_level, kfun, pfun, ON_BOUNDARY, coordinates, intgrl, store_temp, prefetch_list[p])
                    end
                    #perform_factorization!(tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], cur_level, kfun, pfun, ON_BOUNDARY, coordinates, intgrl)
                    s1 += time() - t1


                    ske_points_counter = Array{Future, 1}(undef, subset_probes)
                    pt_box_ct = zeros(2)
                    @sync for p = 1 : subset_probes
                        @async ske_points_counter[p] = remotecall_wait(count_ske_points, probes[probe_range[launch_idx[p]]], cur_level, tree_parts[cur_level + 1], total_factor_nodes[PART_ONE], ON_BOUNDARY)
                    end
                    for p in eachindex(ske_points_counter)
                        count_info = fetch(ske_points_counter[p])
                        pt_box_ct[1] += count_info[1]
                        pt_box_ct[2] += count_info[2]
                    end
                    println("total exterior boxes: ", pt_box_ct[2])
                    println("total ske points: ", pt_box_ct[1])
                    println("average pts per box: ", pt_box_ct[1] / pt_box_ct[2])


                    returned_points[coloring, :] .= to_wait
                    # accumulate the updates immediately after each color had been processed
                    t2 = time()
                    t1 = time()
                    current_layer_num_box = Int64((box_per ^ cur_level) / cur_num_probes)
                    determine_machine = [current_layer_num_box, coloring]
                    #prefetch_candidate = Array{Future, 1}(undef, length(probe_range))
                    #for (p_idx, p) in enumerate(probe_range)
                    #    prefetch_candidate[p_idx] = wait(@spawnat probes[p] prefetch_communication_nodes(tree_parts[cur_level + 1], communication_nodes[cur_level + 1], determine_machine))
                    #end
                    prefetch_candidate = Array{Future, 1}(undef, length(probe_range))
                    @sync for p_idx in eachindex(probe_range)
                        p = probe_range[p_idx]
                        @async prefetch_candidate[p_idx] = remotecall_wait(prefetch_communication_nodes, probes[p], tree_parts[cur_level + 1], communication_nodes[cur_level + 1], determine_machine)
                    end
                #   println("breakkkkkkkkkkkkkkkkkk")
                    prefetch_bigger_list = Array{Future, 1}(undef, length(probe_range))
                    @sync for p_idx in eachindex(probe_range)
                        p = probe_range[p_idx]
                        @async prefetch_bigger_list[p_idx] = remotecall_wait(prefetch_bigger, probes[p], prefetch_candidate[p_idx], factor_nodes[cur_level + 1], coordinates)
                    end
                    println("accumulate fetch time: ", time() - t1)
                    @sync for p_idx in eachindex(probe_range)
                        p = probe_range[p_idx]
                        @async remotecall_wait(accumulate_update!, probes[p], tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1],
                            coordinates, intgrl, determine_machine, [prefetch_candidate[p_idx]; prefetch_bigger_list[p_idx]])
                    end
                    #for inside_coloring = 1 : min(box_per, cur_num_probes)
                    #    inside_launch_idx = range(inside_coloring, stop=cur_num_probes, step=box_per)
                    #    inside_to_wait = [@spawnat probes[probe_range[inside_launch_idx[p]]] accumulate_update!(tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], coordinates, intgrl) for p = 1 : subset_probes]
                    #    for fut in inside_to_wait
                    #        wait(fut) # synchronization point
                    #    end
                    #end
                    s2 += time() - t2
                    # clear communication to avoid double update
                    t3 = time()
                    @sync for p = 1 : subset_probes
                        @async remotecall_wait(clear_communication!, probes[probe_range[launch_idx[p]]], communication_nodes[cur_level + 1])
                    end
                    println("delete time: ", time() - t3)
                end
                # this consists of all the ske_index of the next layer
                returned_points = reshape(returned_points, (1, subset_probes * min(box_per, cur_num_probes)))
                #println(returned_points)
                # for coloring = 1 : min(box_per, cur_num_probes)
                #     launch_idx = range(coloring, stop=cur_num_probes, step=box_per)
                #     to_wait = [@spawnat probes[launch_idx[p]] accumulate_update!(tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], coordinates, intgrl) for p = 1 : subset_probes]
                #     for fut in to_wait
                #         fetch(fut) # synchronization point
                #     end
                # end
                println("exterior factorization time for level ", cur_level, " : ", s1)
                println("accumulate update time for level ", cur_level, " : ", s2)
                
            end

        end
        println("total time spent factorizing: ", factorize_time.time)
    end

    GC.enable(true)

    # Cholesky factor of matrix at top level
    fetch(@spawnat probes[1] x_matrix_nodes[1][1].X_RR = cholesky(Hermitian(modified_nodes[1][1].modified[1])));




    # fetch(@spawnat 1 perform_factorization!(tree_parts[3], factor_nodes[3], communication_nodes[3], 2, kfun, pfun, 0))
    # fetch(@spawnat 1 perform_factorization!(tree_parts[3], factor_nodes[3], communication_nodes[3], 2, kfun, pfun, 1))
    # fetch(@spawnat 1 level_transition!(tree_parts[2], factor_nodes[2], factor_nodes[3], 1, kfun, [1]))
    # fetch(@spawnat 1 perform_factorization!(tree_parts[2], factor_nodes[2], communication_nodes[2], 1, kfun, pfun, 0))
    #fetch(@spawnat 1 perform_factorization!(tree_parts[2], factor_nodes[2], communication_nodes[2], 1, kfun, pfun, 1))
    # gg = [@spawnat 1 perform_factorization!(tree_parts[2], factor_nodes[2], communication_nodes[2], 1, kfun, pfun, 1)]
    # wait(gg[1])

    # perform_factorization!(tree_parts[3], factor_nodes[3], communication_nodes[3], 2, kfun, pfun, 0)
    # perform_factorization!(tree_parts[3], factor_nodes[3], communication_nodes[3], 2, kfun, pfun, 1)
    # level_transition!(tree_parts[2], factor_nodes[2], factor_nodes[3], 1, kfun, [1])
    # perform_factorization!(tree_parts[2], factor_nodes[2], communication_nodes[2], 1, kfun, pfun, 0)
    # perform_factorization!(tree_parts[2], factor_nodes[2], communication_nodes[2], 1, kfun, pfun, 1)


    # aa = factor_nodes[3]
    # println(length(aa[13].ske_index))
    # println(length(aa[14].ske_index))
    # println(length(aa[15].ske_index))
    # println(length(aa[16].ske_index))
    # aa = factor_nodes[2]
    # println(length(aa[4].ske_index))
    aa = factor_nodes[1]
    println(length(aa[1].ske_index))

    #d_closeall()



    if multiply_verify == true
        let
            # temp_future = [@spawnat probes[p] permute_solution(factor_nodes[layer + 1], rightside_nodes[layer + 1], generated_right) for p = 1 : Int64(max(Float64(box_per) ^ (probe_layer), 1.0))]
            # # for fut in temp_future
            # #     fetch(fut)
            # # end
            # #permute_solution(factor_nodes[layer + 1], rightside_nodes[layer + 1], coordinates)
            # leftside = DArray(temp_future)
            
            # println(typeof(leftside))
            # verify = []
            
            if use_MV
                # get the transpose of the permutation
                recovered_point_permutation_t = zeros(Int64, length(recovered_point_permutation))
                recovered_point_permutation_t[recovered_point_permutation] = 1 : length(recovered_point_permutation)
                actual = convert(Vector, generated_right)
                actual = actual[recovered_point_permutation_t]
                sol = []
                for i = 1 : 5
                    #@time sol = solver!(probes, probe_num_progression, tree_parts, factor_nodes, communication_nodes, modified_nodes, x_matrix_nodes, rightside_nodes, recovered_point_permutation, generated_right, actual)
                    # @sync for p in eachindex(probes)
                    #     @async remotecall_wait(comm_reset, probes[p])
                    # end
                    # @time sol = solver!(probes, probe_num_progression, tree_parts, factor_nodes, communication_nodes, modified_nodes, x_matrix_nodes, rightside_nodes, recovered_point_permutation, generated_right, generated_right)
                end
                
                #verify = MV(sol)
                
                #println(norm(verify - actual) / norm(actual))
            else
                # A = zeros(N, N)
                # kfun(original_coordinates, 1 : N, 1 : N, intgrl, A)
                # verify = A * sol
                #result_future = [@spawnat probes[p] verify_residual(original_coordinates, leftside, intgrl) for p = 1 : Int64(max(Float64(box_per) ^ (probe_layer), 1.0))]
                #result = DArray(result_future)
            end
            
        end
        
        preconditioner_operator(x) = solver!(probes, probe_num_progression, tree_parts, factor_nodes, communication_nodes, modified_nodes, 
        x_matrix_nodes, rightside_nodes, recovered_point_permutation, generated_right, x)
        x_left = zeros(N)
        y_right = rand(N)
        y_right = y_right ./ norm(y_right)
        info = cg!(x_left, MV, y_right, reltol=1e-12, Pl=preconditioner_operator, log=true, verbose=true)
        println(info[PART_TWO])
        println("relative residual is: ", norm(MV(x_left) - y_right) / norm(y_right))
        println(info[PART_TWO][:resnorm])
        

    end


    total_storage_in_bytes = 0
    size_calculation = Array{Future, 1}(undef, length(probes))
    @time for cur_level = layer : -1 : 0
        
        @sync for p_idx in eachindex(probes)
            @async size_calculation[p_idx] = remotecall_wait(get_size, probes[p_idx], factor_nodes[cur_level + 1], x_matrix_nodes[cur_level + 1])
        end 

        for p_idx in eachindex(probes)
            total_storage_in_bytes += fetch(size_calculation[p_idx])
        end
        #[@spawnat p println("size of factor on level ", cur_level, " is ", get_size(factor_nodes[cur_level + 1], 1) + get_size(x_matrix_nodes[cur_level + 1], 2)) for p in probes]
    end

    println("total size in bytes: ", total_storage_in_bytes)
    println()
    println()


    d_closeall()
    GC.gc()
    println()
    println()
    println()

    end
end

# function loop_over_global(x)
#
#     for i = 1 : length(x)
#         x[i] = x[i] + 1
#     end
# end
#
# function simple_sum(x)
#
#     x[:] = .+(x, 1)
#     #x[:] .+ 1
# end
#
# x = rand(100000)
#
#
# @time simple_sum(x)
# @time simple_sum(x)


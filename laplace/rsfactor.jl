__precompile__()

module rsfactor
using Distributed
@everywhere using Distributed
@everywhere using DistributedArrays
@everywhere import Base.sizeof
@everywhere const PART_ONE = 1
@everywhere const PART_TWO = 2
@everywhere const PART_THREE = 3




@everywhere in_range(x::Int64, idx_range::Array{Int64, 1}) = (x >= idx_range[1] && x <= idx_range[2])

# data structure for storing the quad/oct tree
@everywhere struct tree_node{T1<:Array{Int64, 1}, T2<:Int64, T3<:Array{Float64, 1}}
    boundary::T1   # check to see if node is on boundary
    children::T1
    close::T1
    interact::T1
    parent::T2
    box_center::T3
    box_length::T3
end



# includes factorization info, which can be used to solve linear systems
@everywhere mutable struct factor_info_1{T1<:Array{Int64, 1}}
    ske_index::T1
    red_index::T1
    relative_ske_index::T1
    relative_red_index::T1
    neighbor_idx::T1
    idx_range::T1
end

@everywhere mutable struct factor_info_2{T1<:Array{Int64, 1}, T2<:Array{Array, 1}}
    modified::T2
    update_close::T1
end

@everywhere mutable struct factor_info_3{T2<:Array{PRECISION, 2}}
    X_RR::Factorization
    X_RS::T2
    X_RN::T2
    X_SR::T2
    X_NR::T2
    T::T2
end

@everywhere struct rightside_block{T1<:Array{PRECISION, 1}}
    right_side::T1
end

# used for boundary communication purposes for both factorization and solve
@everywhere mutable struct communication_node{T1<:Array{Int64, 1}, T2<:Array{Array, 1}}
    smaller_idx::T1
    bigger_idx::T1
    sidx_array::T2
    bidx_array::T2
end

@everywhere function sizeof(a::communication_node)
    track = 0
    for i in eachindex(a.sidx_array)
        track += sizeof(a.sidx_array[i])
    end
    for i in eachindex(a.bidx_array)
        track += sizeof(a.bidx_array[i])
    end
    total = sizeof(a.smaller_idx) + sizeof(a.bigger_idx) + track
    return total
end

@everywhere function sizeof(a::rightside_block)
    total = sizeof(a.right_side)
    return total
end

@everywhere function sizeof(a::factor_info_3)
    total = sizeof(a.X_RR) + sizeof(a.X_RS) + sizeof(a.X_RN) + sizeof(a.X_SR) + sizeof(a.X_NR) + sizeof(a.T)
    return total
end

@everywhere function sizeof(a::factor_info_2)
    track = 0
    for i in eachindex(a.modified)
        track += sizeof(a.modified[i])
    end
    total = track + sizeof(a.update_close)
    return total
end


@everywhere function sizeof(a::factor_info_1)
    total = sizeof(a.ske_index) + sizeof(a.red_index) + sizeof(a.relative_ske_index) + 
        sizeof(a.relative_red_index) + sizeof(a.neighbor_idx) + sizeof(a.idx_range)
    return total
end


@everywhere function sizeof(a::tree_node)
    total = sizeof(a.boundary) + sizeof(a.children) + sizeof(a.close) + 
        sizeof(a.interact) + sizeof(a.parent) + sizeof(a.box_center) + sizeof(a.box_length)
    return total
end

include("optimize_communication.jl")


# build the tree data structure
@everywhere function create_tree(coordinate_range::Array{T, 1}, worker_id::Int64, layer::Int64, max_layer::Int64, probes_length::Int64) where T <: AbstractFloat
    box_per = BOXES_PER_NODE
    prob_dim = DIMENSION
    total_boxes = Int64((box_per ^ layer) / probes_length) # num of boxes this process gets
    level = [0; cumsum(box_per .^ (0 : max_layer))] # can find start index of every level

    # declare tree
    tree = Array{tree_node, 2}(undef, 1, total_boxes)

    # figure out the actual box index range relative to current layer
    box_range = ((total_boxes * (worker_id - 1) + 1) : (total_boxes * worker_id))


    for i = 1 : length(box_range)
        layer_idx = box_range[i] # index relative to layer

        # calculate children boxes
        children = zeros(Int64, box_per)
        if layer != max_layer
            # absolute index
            # first_child = level[layer + 2] + (layer_idx - 1) * box_per + 1
            # children .= first_child : (first_child + box_per - 1)

            # relative index
            children .= layer_idx * box_per - (box_per - 1) : layer_idx * box_per
        end

        # calculate neighbor boxes
        neighbor = get_neighbor(level[layer + 1] + layer_idx, layer, prob_dim)


        interact = zeros(Int64, 0)
        parent = 0
        if layer != 0
            # calculate parent
            # parent = level[layer + 1] - (box_per ^ (layer - 1) - Int64(ceil(layer_idx / 4)))
            parent = Int64(ceil(layer_idx / box_per))

            # calculate interaction list
            parent_neighbor = get_neighbor(level[layer + 1] - (box_per ^ (layer - 1) - parent), layer - 1, prob_dim)
            candidates = zeros(Int64, length(parent_neighbor) * box_per)
            count = 1
            for each_neighbor in parent_neighbor
                # for each parent's neighbor, get all their children filled in
                # first_child = level[layer + 1] + (each_neighbor - level[layer] - 1) * box_per + 1
                first_child = each_neighbor * box_per - (box_per - 1)
                candidates[count : count + box_per - 1] = first_child : (first_child + box_per - 1)
                count += box_per
            end
            # if candidate is current node's parent's neighbor's child and is not a neighbor of current node

            for candidate in candidates
                if !(candidate in neighbor)
                    push!(interact, candidate)
                end
            end
        end

        # box_length
        box_length = (coordinate_range[2] - coordinate_range[1]) / 2 ^ layer

        # box_center
        box_center = (morton2cartesian(layer_idx) .- 0.5) .* box_length .+ coordinate_range[1]

        # filter out the interaction list, leaving only boxes on proxy
        interact = get_proxy_box(interact, box_length, box_center, coordinate_range[1])

        # build tree node, initialize boundary to 0, which will be set later
        tree[i] = tree_node{Array{Int64, 1}, Int64, Array{Float64, 1}}([0], children, neighbor, interact, parent, box_center, [box_length])
    end

    boundary = mark_boundary(box_range, worker_id, Int64(total_boxes ^ (1 / prob_dim)))
    for i in boundary
        tree[i - (worker_id - 1) * total_boxes].boundary[1] = 1
    end

    return tree
end

# filter out the non proxy boxes from interaction list
@everywhere function get_proxy_box(interact::Array{Int64, 1}, box_length::AbstractFloat, box_center::Array, min_point)
    proxy_list = zeros(Int64, 0)
    for i in interact
        num_box = ((morton2cartesian(i) .- 0.5) * box_length .- box_center .+ min_point) ./ box_length
        conditional_bool = true
        for j in num_box
            if abs(j) > 2 && !isapprox(abs(j), 2)
                conditional_bool = false
            end
        end
        if conditional_bool == true
            push!(proxy_list, i)
        end
    end
    return proxy_list
end


# find neighbor of node, where node is index with respect to the entire tree
@everywhere function get_neighbor(node::Int64, layer::Int64, prob_dim::Int64)
    if prob_dim == 2
        # 2D problem
        return get_neighbor_square(layer, node)
    else
        # 3D problem, TO BE IMPLEMENTED LATER
        return zeros(Int64, 0)
    end
end

# function for getting neighbors of i for 2D square
@everywhere function get_neighbor_square(level::Int64, i::Int64)
    list = zeros(Int64, 0);

    if i < 1
        return list;
    end

    # cumulative number of blocks in all previous levels
    offset = Int64((4 ^ level - 1) / 3) # ADD TO MORTON IF ONE WANTS ABSOLUTE INDEX WITHIN WHOLE TREE
    side = 2^level
    # location relative to layer, not to entire tree
    location = morton2cartesian(i - offset)
    x = location[1]
    y = location[2]

    # x - 1, y - 1
    if x - 1 >= 1 && y - 1 >= 1
        push!(list, cartesian2morton([x - 1, y - 1]))
    end

    # x, y - 1
    if y - 1 >= 1
        push!(list, cartesian2morton([x, y - 1]))
    end

    # x + 1, y - 1
    if x + 1 <= side && y - 1 >= 1
        push!(list, cartesian2morton([x + 1, y - 1]))
    end

    # x - 1, y
    if x - 1 >= 1
        push!(list, cartesian2morton([x - 1, y]))
    end

    # x + 1, y
    if x + 1 <= side
        push!(list, cartesian2morton([x + 1, y]))
    end

    # x - 1, y + 1
    if x - 1 >= 1 && y + 1 <= side
        push!(list, cartesian2morton([x - 1, y + 1]))
    end

    # x, y + 1
    if y + 1 <= side
        push!(list, cartesian2morton([x, y + 1]))
    end

    # x + 1, y + 1
    if x + 1 <= side && y + 1 <= side
        push!(list, cartesian2morton([x + 1, y + 1]))
    end

    return list
end

@everywhere function mark_boundary(index_range::AbstractArray, region::Int64, side_length::Int64)
    boundary_index = zeros(Int64, 0)
    if DIMENSION == 2
        # 2D problem
        region_coordinate = morton2cartesian(region)
        x_left = (region_coordinate[1] - 1) * side_length + 1
        x_right = region_coordinate[1] * side_length
        y_left = (region_coordinate[2] - 1) * side_length + 1
        y_right = (region_coordinate[2]) * side_length
        @inbounds for i in index_range
            coordinate = morton2cartesian(i)
            # if on boundary, add it to list
            if coordinate[1] == x_left || coordinate[1] == x_right || coordinate[2] == y_left || coordinate[2] == y_right
                push!(boundary_index, i)
            end
        end
    else
        # 3D problem, TO BE IMPLEMENTED

    end

    return boundary_index
end

# for the bottom layer, first classify the points into the boxes, also builds factor tree (contain index info)
@everywhere function classify_points(points::DArray{T, 2}, worker_id::Int64, layer::Int64, max_layer::Int64, probes_length::Int64, box_length::AbstractFloat, min_point::Array{PRECISION, 1}) where {T <: AbstractFloat}
    loc_points = localpart(points)
    loc_indices = localindices(points)[2]
    box_per = BOXES_PER_NODE
    prob_dim = DIMENSION
    total_boxes = Int64((box_per ^ layer) / probes_length) # num of boxes this process gets
    # declare factor information tree
    factor_node = Array{factor_info_1, 2}(undef, 1, total_boxes)
    for i = 1 : length(factor_node)
        factor_node[i] = factor_info_1(zeros(Int64, 0), zeros(Int64, 0), zeros(Int64, 0), zeros(Int64, 0), zeros(Int64, 0), [loc_indices[1]; loc_indices[end]])
    end

    if layer != max_layer
        return factor_node
    end


    # figure out the offset to add relative to current layer
    box_offset = (total_boxes * (worker_id - 1))

    for idx = 1 : size(loc_points, 2)
        point = loc_points[:, idx] - min_point
        point ./= box_length

        nu_error = round.(point)
        for c = 1 : length(point)
            if isapprox(nu_error[c], point[c])
                point[c] = nu_error[c]
            end
        end
        
        rounded_point = Int64.(ceil.(point))
        for c = 1 :  length(rounded_point)
            if rounded_point[c] == 0
                rounded_point[c] = 1
            end
        end
        # convert point to the morton Number, add index(relative to current layer) to factor
       
        push!(factor_node[cartesian2morton(rounded_point) - box_offset].ske_index, loc_indices[idx])
    end

    return factor_node
end

# contains the modified and update_close parameter of the factor_info
@everywhere function create_other_info(layer::Int64, probes_length::Int64, which_part::Int64)
    box_per = BOXES_PER_NODE
    prob_dim = DIMENSION
    total_boxes = Int64((box_per ^ layer) / probes_length) # num of boxes this process gets

    # declare factor information tree
    factor_node = []
    if which_part == 3
        temp = cholesky(0)
        factor_node = Array{factor_info_3, 2}(undef, 1, total_boxes)
        for i = 1 : length(factor_node)
            factor_node[i] = factor_info_3(temp, zeros(PRECISION, 0, 0),
                zeros(PRECISION, 0, 0), zeros(PRECISION, 0, 0), zeros(PRECISION, 0, 0), zeros(PRECISION, 0, 0))
        end
    else
        factor_node = Array{factor_info_2, 2}(undef, 1, total_boxes)
        for i = 1 : length(factor_node)
            factor_node[i] = factor_info_2(Array{Array, 1}(undef, 0), zeros(Int64, 0))
        end
    end
    return factor_node
end


@everywhere function accumulate_update!(tree_darray::DArray, factor_darray::Array, communication_darray::DArray, coordinates::DArray{Float64, 2}, intgrl::Float64, determine_machine::Array, fetched_data::Array)
    
    tree_nodes = localpart(tree_darray)
    local_idx = localindices(tree_darray)[2]
    # this offset is used to index into the local parts, IMPORTANT TO REMEMBER
    offset = local_idx[1] - 1
    # find the ones not on current machine, which will be the ones that need to resort to the communication method in the first place
    
    accumulate_update!(fetched_data, factor_darray, communication_darray, [local_idx[1]; local_idx[end]], coordinates, intgrl)
end

# function for transmitting updates between boundaries of processes
@everywhere function accumulate_update!(fetched_data::Array, factor_darray::Array, communication_darray::DArray, idx_range::Array{Int64, 1}, coordinates::DArray{Float64, 2}, intgrl::Float64)
    offset = idx_range[1] - 1
    factor_nodes = localpart(factor_darray[PART_ONE])
    factor_nodes_second = localpart(factor_darray[PART_TWO])
    candidate_list = fetch(fetched_data[PART_ONE])
    f_node_dict = []
    points_dict = []
    if !isempty(candidate_list)
        f_node_dict, points_dict = fetch(fetched_data[PART_TWO])
    end

    for candidate in eachindex(candidate_list)
        c_node = candidate_list[candidate]

        relevant_idx = findall(x->in_range(x, idx_range), c_node.smaller_idx)
        smaller_list = c_node.smaller_idx[relevant_idx]
        bigger_list = c_node.bigger_idx[relevant_idx]
        # update all in smaller_list
        for i = 1 : length(smaller_list)
            f_node = factor_nodes[smaller_list[i] - offset]
            f_node_second = factor_nodes_second[smaller_list[i] - offset]
            update = f_node_second.update_close
            up_idx = findall(x->x==bigger_list[i], update)
            mat_a = c_node.sidx_array[relevant_idx[i]]
            mat_b = c_node.bidx_array[relevant_idx[i]]
            # if both ends are the same, then only 1 part is stored
            if smaller_list[i] == bigger_list[i]
                mat_b = mat_a
            end



            other_f_node = []
            if in_range(bigger_list[i], idx_range)
                other_f_node = factor_nodes[bigger_list[i] - offset]
            else
                other_f_node = get(f_node_dict, bigger_list[i], -1)
                @assert other_f_node != -1
            end
            if !isempty(up_idx)
                # up_idx = up_idx[1]
                # col_idx = 1 : size(f_node_second.modified[idx], 1);
                # row_idx = 1 : size(f_node_second.modified[idx], 2);
                # if length(col_idx) != size(change_matrix, 1)
                #     col_idx = f_node.relative_ske_index;
                # end
                #
                # if length(row_idx) != size(change_matrix, 2)
                #     row_idx = f_node.relative_ske_index;
                # end

                idx = up_idx[1]

                if size(f_node_second.modified[idx], 1) > length(f_node.ske_index) && size(f_node_second.modified[idx], 2) > length(other_f_node.ske_index)
                    f_node_second.modified[idx] = f_node_second.modified[idx][f_node.relative_ske_index, other_f_node.relative_ske_index]
                elseif size(f_node_second.modified[idx], 1) > length(f_node.ske_index)
                    f_node_second.modified[idx] = f_node_second.modified[idx][f_node.relative_ske_index, :]
                elseif size(f_node_second.modified[idx], 2) > length(other_f_node.ske_index)
                    f_node_second.modified[idx] = f_node_second.modified[idx][:, other_f_node.relative_ske_index]
                end

                if size(mat_a, 1) > length(f_node.ske_index)
                    mat_a = mat_a[f_node.relative_ske_index, :]
                end
                if size(mat_b, 1) > length(other_f_node.ske_index)
                    mat_b = mat_b[other_f_node.relative_ske_index, :]
                end



                BLAS.gemm!('N', 'T', -1.0, mat_a, mat_b, 1.0, f_node_second.modified[idx])

            else
                mapped_points = get(points_dict, bigger_list[i], -1)
                if mapped_points == -1
                    mapped_points = other_f_node.ske_index
                end

                # add the new interact box to update
                push!(update, bigger_list[i])

                A = zeros(length(f_node.ske_index), length(other_f_node.ske_index))
                kfun_internal!(coordinates, f_node.ske_index, mapped_points, intgrl, A)
                if size(mat_a, 1) != length(f_node.ske_index)
                    mat_a = mat_a[f_node.relative_ske_index, :]
                end
                if size(mat_b, 1) != length(other_f_node.ske_index)
                    mat_b = mat_b[other_f_node.relative_ske_index, :]
                end
                BLAS.gemm!('N', 'T', -1.0, mat_a, mat_b, 1.0, A)
                #A .-= change_matrix
                push!(f_node_second.modified, A)
            end

        end

    end

end

# factorization for non boundary nodes
@everywhere function perform_factorization!(tree_darray::DArray, factor_darray::Array, communication_darray::DArray, current_layer::Int64, kfun::Function,
    pfun::Function, on_boundary::Int64, coordinates::DArray{Float64, 2}, intgrl::Float64, store_size::DArray, fetched_map_future = Future[])

  
    # set tolerance and define proxy
    err = ERROR
    num_proxy_points = 64
    theta = (1 : num_proxy_points) * 2 * pi / num_proxy_points
    proxy = [cos.(theta)'; sin.(theta)']
    loc_coordinates = localpart(coordinates)
    loc_store = localpart(store_size)
    new_coordinates_size = 0
    # 3D proxy to be implemented

    tree_nodes = localpart(tree_darray)
    factor_nodes = localpart(factor_darray[PART_ONE])
    factor_nodes_second = localpart(factor_darray[PART_TWO])
    factor_nodes_third = localpart(factor_darray[PART_THREE])
    communication_nodes = localpart(communication_darray)
    local_idx = localindices(tree_darray)[2]
    # this offset is used to index into the local parts, IMPORTANT TO REMEMBER
    offset = local_idx[1] - 1
    idx_range = [local_idx[1]; local_idx[end]]
    check_repeat = Dict()


    s1 = 0
    s2 = 0
    s3 = 0
    tot_stat = 0
    id_mat_len = 0
    processed_num = 0
    own_block_len = 0
    T_len = (0, 0)
    A_SR_len = (0, 0)
    pre_allocate_mat = zeros(100, 500)

    # prefetch some stuff from other machines to speed up access
    
    if fetched_map_future != Future[]
        fetch_time = time()
        fetched_map = fetch(fetched_map_future)
        println("fetch time for perform_factorization: ", time() - fetch_time)
    else
	    t1 = time()
        fetched_map = prefetch_points(tree_darray, factor_darray[PART_ONE], coordinates, on_boundary)
	    println("factorization color fetch time: ", time() - t1)
    end
    
    f_nodes_dict = fetched_map[1]
    points_dict = fetched_map[2]

    #println("gooby")
    GC.enable(false)
    @time for i in eachindex(factor_nodes)


        f_node = factor_nodes[i]
        f_node_third = factor_nodes_third[i]
        t_node = tree_nodes[i]
        c_node = communication_nodes[i]
        # if it's not part of the current boundary or no points, skip
        if isempty(f_node.ske_index) || t_node.boundary[1] != on_boundary
            if on_boundary == 1
                new_coordinates_size += length(f_node.ske_index)
            end
            continue
        end


        # if factoring boundary, accumulate updates first

        if on_boundary == 1
            #accumulate_update!(i + offset, t_node, factor_darray, communication_darray, idx_range, check_repeat, coordinates, intgrl)
        end

        # get interact list and close list
        interaction_list = t_node.interact
        close_list = t_node.close

        #------------------------------- do interpolative decomposition using proxy points

        nlist = interaction_list
        #nlist = filter(x -> in_range(x, idx_range), interaction_list)
        proxy_mat = []
        if current_layer > 1
            adjusted_circle = t_node.box_center .+ 2.5 * t_node.box_length[1] * proxy
            proxy_mat = zeros(size(adjusted_circle, 2), length(f_node.ske_index))
            pfun(adjusted_circle, f_node.ske_index, coordinates, proxy_mat)
            proxy_mat = proxy_mat'
        else
            # if at layer 1 or 0, then no far field boxes, do near compression
            nlist = close_list
            adjusted_circle = zeros(0, length(f_node.ske_index))
            proxy_mat = zeros(length(f_node.ske_index), 0)
        end

        t1 = @timed begin
        pre_allocate_mat, len_list_far = build_matrix(factor_darray, tree_nodes, communication_nodes, proxy_mat, nlist, i, kfun, offset, idx_range, coordinates, intgrl, pre_allocate_mat, fetched_map)
        end


        opts = LRAOptions(rtol=err)
        len_of_qr = sum(len_list_far) + size(proxy_mat, 2)

        @views S_idx, R_idx, T = id(pre_allocate_mat[1 : length(f_node.ske_index), 1 : len_of_qr]', opts);


        add_or_not = true
        for gg in nlist
            if !in_range(gg, idx_range)
                add_or_not = true
            end
        end
        if add_or_not == true
            s1 += t1.time
        end


        #S_idx, R_idx, T = id(pre_allocate_mat', opts);
        # sort it so that indices are in order, which can later be accessed contiguously
        # s_perm = sortperm(S_idx)
        # r_perm = sortperm(R_idx)
        # T = T[s_perm, r_perm]
        # S_idx = S_idx[s_perm]
        # R_idx = R_idx[r_perm]
        id_mat_len += len_of_qr
        processed_num += 1
        own_block_len += length(f_node.ske_index)

        #=
        _, R, p = qr(pre_allocate_mat[1 : length(f_node.ske_index), 1 : len_of_qr]', Val(true));
        temp = findall(x->x < abs(R[1, 1]) * err, abs.(diag(R)));

        if isempty(temp)
            f_node.red_index = []
            f_node.relative_ske_index = 1 : length(f_node.ske_index)
            f_node.relative_red_index = []
            cutoff = min(size(R, 2), size(R, 1));
            continue;
        else
            cutoff = temp[1] - 1;
        end
        if(cutoff < 1)
            cutoff = 1;
        end
        # change the permutation to make the order work out according to QR
        # its position relative to the submatrix is 1 : length(p)
        tempidx = 1 : length(p);
        tempidx = tempidx[p];
        reverseid = [cutoff + 1 : length(tempidx); 1 : cutoff];
        tempidx = tempidx[reverseid];

        # perform reduction and calculation on submatrix
        S_idx = tempidx[end - cutoff + 1 : end];
        R_idx = tempidx[1 : end - cutoff];
        s_perm = sortperm(S_idx)
        r_perm = sortperm(R_idx)
        R = R[:, [s_perm; r_perm .+ length(S_idx)]];

        T = (R[1 : cutoff, 1 : cutoff]) \ R[1 : cutoff, cutoff + 1 : end];
        # store T
        f_node_third.T = T;
        S_idx = S_idx[s_perm]
        R_idx = R_idx[r_perm]
        =#

        tot = @timed begin
        t2 = time()

        if on_boundary == 1
            new_coordinates_size += length(S_idx)
        end
        if isempty(R_idx)
            f_node.red_index = f_node.ske_index[R_idx]
            f_node.ske_index = f_node.ske_index[S_idx]
            f_node.relative_ske_index = S_idx
            f_node.relative_red_index = R_idx
            continue;
        end
        f_node_third.T = T
        # if current_layer == 2
        #     println(size(qrmat))
        #     println(length(S_idx))
        #     println(S_idx)
        #     te = qrmat'
        #     println(norm(te[:, S_idx] * cat(I(size(T, 1)), T, dims=2) - te[:, [S_idx; R_idx]]))
        # end


        # ---------------------------- builds the RR, RS, RN matrix
        # nlist includes itself and neighbors unless at the top level and doing near field compression
        nlist = [i + offset; close_list]
        if current_layer <= 1
            nlist = [i + offset]
        end
        # construct submatrix matrix A_(R)(S, N)

        pre_allocate_mat, lenlist = build_matrix(factor_darray, tree_nodes, communication_nodes,
            zeros(length(f_node.ske_index), 0), nlist, i, kfun, offset, idx_range, coordinates, intgrl, pre_allocate_mat, fetched_map)
        pre_allocate_mat = pre_allocate_mat'
        len_near = sum(lenlist)



        # change to redundant and compute results for f_node

        f_node.neighbor_idx = [0; cumsum(lenlist[2 : end])]
        f_node.red_index = f_node.ske_index[R_idx]
        f_node.ske_index = f_node.ske_index[S_idx]
        f_node.relative_ske_index = S_idx
        f_node.relative_red_index = R_idx
        A_SS = pre_allocate_mat[S_idx, S_idx]
        X_SR = pre_allocate_mat[S_idx, R_idx];
        A_RR = pre_allocate_mat[R_idx, R_idx];


        T_len = T_len .+ size(T)
        A_SR_len = A_SR_len .+ size(X_SR)

        T_ASR = T' * X_SR
        ASS_T = A_SS * T
        F = cholesky(Hermitian(A_RR .- T_ASR .- T_ASR' .+ T' * ASS_T))
        upper_mat = Array(F.U)
        #X_SR = (A_SR .- A_SS * T) / F.U
        BLAS.gemm!('N', 'N', -1.0, A_SS, T, 1.0, X_SR)
        BLAS.trsm!('R', 'U', 'N', 'N', 1.0, upper_mat, X_SR)

        len_p = length(R_idx) + length(S_idx)

        X_NR = pre_allocate_mat[len_p + 1 : len_near, R_idx]
        BLAS.gemm!('N', 'N', -1.0, pre_allocate_mat[len_p + 1 : len_near, S_idx], T, 1.0, X_NR)
        BLAS.trsm!('R', 'U', 'N', 'N', 1.0, upper_mat, X_NR)

        #X_NR = (submatrix[len_p + 1 : end, R_idx] - submatrix[len_p + 1 : end, S_idx] * T) / F.U;
        f_node_third.X_RR = F
        f_node_third.X_SR = X_SR
        f_node_third.X_NR = X_NR
        s2 += (time() - t2)
        pre_allocate_mat = pre_allocate_mat' # revert it back so this array maintains horizontal
        #------------------------------------- perform updates between each nlist boxes

        # get rid of the length of the non skeleton part
        lenlist = [0; cumsum(lenlist[2 : end])]
        t3 = @timed begin
        for (k, k1) in enumerate(nlist)
            for (c, k2) in enumerate(nlist)
                # assuming symmetry, only store 1 pair, skip otherwise unless edge case
                k1_f_node = get(f_nodes_dict, k1, -1)
                k2_f_node = get(f_nodes_dict, k2, -1)
                if k1_f_node == -1
                    k1_f_node = factor_nodes[k1 - offset]
                end
                if k2_f_node == -1
                    k2_f_node = factor_nodes[k2 - offset]
                end

                # if k1 > k2, skip for most circumstances unless a special case on boundary
                if k1 > k2
                    if on_boundary == 1
                        # WANT TO CHECK IF K1 AND K2 ON SAME MACHINE
                        if k1_f_node.idx_range == k2_f_node.idx_range
                            continue;
                        elseif k1_f_node.idx_range != k2_f_node.idx_range && !in_range(k1, idx_range)
                            push!(c_node.smaller_idx, k1)
                            push!(c_node.bigger_idx, k2)
                            push!(c_node.sidx_array, X_NR[lenlist[k - 1] + 1 : lenlist[k], :])
                            if c != 1
                                push!(c_node.bidx_array, X_NR[lenlist[c - 1] + 1 : lenlist[c], :])
                            else
                                # if the other guy is located at index 1, use X_SR
                                push!(c_node.bidx_array, X_SR)
                            end
                            continue;
                        end
                    else
                        continue;
                    end
                end

                if in_range(k1, idx_range)

                    # assuming that k1 is on the current machine
                    k1_f_node_second = factor_nodes_second[k1 - offset]


                    update = k1_f_node_second.update_close
                    idx = findall(x -> x == k2, update)


                    # change_matrix = []
                    # if k1 == k2
                    #     # if k1 == k2, force matrix to be symmetric
                    #     if k1 == i + offset
                    #         change_matrix = X_SR * X_SR'
                    #     else
                    #         @views change_matrix = X_NR[lenlist[k - 1] + 1 : lenlist[k], :] * X_NR[lenlist[k - 1] + 1 : lenlist[k], :]'
                    #     end
                    #
                    # else
                    #     if k1 == i + offset
                    #         @views change_matrix = X_SR * X_NR[lenlist[c - 1] + 1 : lenlist[c], :]'
                    #     elseif k2 == i + offset
                    #         @views change_matrix = X_NR[lenlist[k - 1] + 1 : lenlist[k], :] * X_SR'
                    #     else
                    #         @views change_matrix = X_NR[lenlist[k - 1] + 1 : lenlist[k], :] * X_NR[lenlist[c - 1] + 1 : lenlist[c], :]';
                    #     end
                    # end


                    if !isempty(idx)

                        # idx = idx[1]
                        # col_idx = 1 : size(k1_f_node_second.modified[idx], 1);
                        # row_idx = 1 : size(k1_f_node_second.modified[idx], 2);
                        # if length(col_idx) != size(change_matrix, 1)
                        #     col_idx = k1_f_node.relative_ske_index;
                        # end
                        # if length(row_idx) != size(change_matrix, 2)
                        #     row_idx = k2_f_node.relative_ske_index;
                        # end
                        # if size(k1_f_node_second.modified[idx]) == (length(col_idx), length(row_idx))
                        #     # if the size didn't change
                        #     k1_f_node_second.modified[idx] .-= change_matrix
                        # else
                        #     @views k1_f_node_second.modified[idx] = k1_f_node_second.modified[idx][col_idx, row_idx] .- change_matrix;
                        # end



                        idx = idx[1]
                        if length(k1_f_node.ske_index) != size(k1_f_node_second.modified[idx], 1) && length(k2_f_node.ske_index) != size(k1_f_node_second.modified[idx], 2)
                            k1_f_node_second.modified[idx] = k1_f_node_second.modified[idx][k1_f_node.relative_ske_index, k2_f_node.relative_ske_index]
                        elseif length(k1_f_node.ske_index) != size(k1_f_node_second.modified[idx], 1)
                            k1_f_node_second.modified[idx] = k1_f_node_second.modified[idx][k1_f_node.relative_ske_index, :]
                        elseif length(k2_f_node.ske_index) != size(k1_f_node_second.modified[idx], 2)
                            k1_f_node_second.modified[idx] = k1_f_node_second.modified[idx][:, k2_f_node.relative_ske_index]
                        end

                        # if (k1 != k2) && (k1 == i + offset) && lenlist[c] - lenlist[c - 1] != size(k1_f_node_second.modified[idx], 2)
                        #     println("k11111111111111111111111111111111111: ", k1)
                        #     println("k22222222222222222222222222222222222: ", k2)
                        #     println("fffffffffffffffffffffffffffffff: ", length(k2_f_node.ske_index))
                        #     println("originnnnnnnnnnnnnnnnnnnnnnnnnn: ", keys(f_nodes_dict))
                        # end


                        if k1 == k2
                            # if k1 == k2, force matrix to be symmetric
                            if k1 == i + offset
                                @views BLAS.gemm!('N', 'T', -1.0, X_SR, X_SR, 1.0, k1_f_node_second.modified[idx])
                            else
                                @views BLAS.gemm!('N', 'T', -1.0, X_NR[lenlist[k - 1] + 1 : lenlist[k], :], X_NR[lenlist[k - 1] + 1 : lenlist[k], :], 1.0, k1_f_node_second.modified[idx])
                            end

                        else
                            if k1 == i + offset
                                @views BLAS.gemm!('N', 'T', -1.0, X_SR, X_NR[lenlist[c - 1] + 1 : lenlist[c], :], 1.0, k1_f_node_second.modified[idx])
                            elseif k2 == i + offset
                                @views BLAS.gemm!('N', 'T', -1.0, X_NR[lenlist[k - 1] + 1 : lenlist[k], :], X_SR, 1.0, k1_f_node_second.modified[idx])
                            else
                                @views BLAS.gemm!('N', 'T', -1.0, X_NR[lenlist[k - 1] + 1 : lenlist[k], :], X_NR[lenlist[c - 1] + 1 : lenlist[c], :], 1.0, k1_f_node_second.modified[idx])
                            end
                        end



                    else

                        # add the new interact box to update
                        push!(update, k2)
                        mapped_points_k1 = get(points_dict, k1, -1)
                        mapped_points_k2 = get(points_dict, k2, -1)
                        if mapped_points_k1 == -1
                            mapped_points_k1 = k1_f_node.ske_index
                        end
                        if mapped_points_k2 == -1
                            mapped_points_k2 = k2_f_node.ske_index
                        end
                        A = zeros(length(k1_f_node.ske_index), length(k2_f_node.ske_index))
                        kfun(coordinates, mapped_points_k1, mapped_points_k2, intgrl, A)

                        if k1 == k2
                            # if k1 == k2, force matrix to be symmetric
                            if k1 == i + offset
                                @views BLAS.gemm!('N', 'T', -1.0, X_SR, X_SR, 1.0, A)
                            else
                                @views BLAS.gemm!('N', 'T', -1.0, X_NR[lenlist[k - 1] + 1 : lenlist[k], :], X_NR[lenlist[k - 1] + 1 : lenlist[k], :], 1.0, A)
                            end

                        else
                            if k1 == i + offset
                                @views BLAS.gemm!('N', 'T', -1.0, X_SR, X_NR[lenlist[c - 1] + 1 : lenlist[c], :], 1.0, A)
                            elseif k2 == i + offset
                                @views BLAS.gemm!('N', 'T', -1.0, X_NR[lenlist[k - 1] + 1 : lenlist[k], :], X_SR, 1.0, A)
                            else
                                @views BLAS.gemm!('N', 'T', -1.0, X_NR[lenlist[k - 1] + 1 : lenlist[k], :], X_NR[lenlist[c - 1] + 1 : lenlist[c], :], 1.0, A)
                            end
                        end

                        #A .-= change_matrix
                        push!(k1_f_node_second.modified, A)


                    end

                else
                    #@assert false
                    # THE CASE WHERE K1 IS NOT ON CURRENT MACHINE, THIS IS THE HARD PART
                    # we know that if k1 = offset + i (in other words, it's the current node), then it can never reach here
                    # storing k1 into communication struct pointed to by c_node, which can be used by the machine k1 is on
                    #println("k1 :", k1, " k: ", k, " k2: ", k2, " c: ", c)
                    push!(c_node.smaller_idx, k1)
                    push!(c_node.bigger_idx, k2)
                    push!(c_node.sidx_array, X_NR[lenlist[k - 1] + 1 : lenlist[k], :])
                    # if k1 == k2, then only 1 part is needed
                    if k1 != k2
                        if c != 1
                            push!(c_node.bidx_array, X_NR[lenlist[c - 1] + 1 : lenlist[c], :])
                        else
                            # if the other guy is located at index 1, use X_SR
                            push!(c_node.bidx_array, X_SR)
                        end
                    else
                        push!(c_node.bidx_array, [])
                    end

                end
            end
        end
        end
        s3 += t3.time
        end
        tot_stat += tot.gctime
        #----------------------------------- update neighbor idx
        # if on_boundary == 0
        #
        #     neighbor_length = 0
        #     for k1 in nlist
        #         k1_f_node = factor_nodes[k1 - offset]
        #         neighbor_length = neighbor_length + length(k1_f_node.ske_index);
        #     end
        #     neighbor_idx = zeros(Int64, neighbor_length);
        #     neighbor_length = 1;
        #     for k1 in nlist
        #         k1_f_node = factor_nodes[k1 - offset]
        #         temp_length = length(k1_f_node.ske_index);
        #         neighbor_idx[neighbor_length : neighbor_length + temp_length - 1] = k1_f_node.ske_index;
        #         neighbor_length = neighbor_length + temp_length;
        #     end
        #     # set the current f_node's neighbor list
        #     f_node.neighbor_idx = neighbor_idx;
        # end
    end
    GC.enable(true)
    GC.gc()
    if (myid() == workers()[1] || length(workers()) == 1) && false

    # if on_boundary == 0
    #println("-------------------------------------------------------------- average own block length: ", own_block_len / processed_num)
    #println("-------------------------------------------------------------- average id mat length: ", id_mat_len / processed_num)
    #println("-------------------------------------------------------------- average T length: ", T_len ./ processed_num)
    #println("-------------------------------------------------------------- average A_SR length: ", A_SR_len ./ processed_num)
    println("-------------------------------------------------------------- num boxes processed: ", processed_num)
    println("-------------------------------------------------------------- id section time: ", s1)
    println("-------------------------------------------------------------- calculate/modify closefield time: ", s2)
    println("-------------------------------------------------------------- store update byte: ", s3)
    println("-------------------------------------------------------------- total byte: ", tot_stat)

    # end
    end

    new_coordinates = zeros(DIMENSION, 0)
    count = 1
    if on_boundary == 1
        new_coordinates = zeros(DIMENSION, new_coordinates_size)
        for i in eachindex(factor_nodes)
            f_node = factor_nodes[i]
            temp_len = length(f_node.ske_index)
            new_coordinates[:, count : count + temp_len - 1] = loc_coordinates[:, f_node.ske_index .- f_node.idx_range[1] .+ 1]
            count += temp_len
        end
        loc_store[1] = size(new_coordinates, 2)
    end


    return new_coordinates
end

# transition between layers for factorization
@everywhere function level_transition!(tree_darray::DArray, tree_below::DArray, factor_darray::Array, factor_level_below::Array, current_layer::Int64, kfun::Function, probe_list::Array{Int64, 1}, coordinates::DArray{Float64, 2}, intgrl::Float64,
    new_points::Array, store_size::DArray{Int64, 2}, machine_idx::Int64, changed_or_not::Bool, fetched_messages)
    tree_nodes = localpart(tree_darray)
    factor_nodes = localpart(factor_darray[PART_ONE])
    local_idx = localindices(factor_level_below[PART_ONE])[2]
    my_idx = localindices(tree_darray)[2]
    offset = my_idx[1] - 1
    idx_range = [my_idx[1]; my_idx[end]]
    box_per = BOXES_PER_NODE

    #@timev begin
    # check if machine number changed between level, if it did, then create the points for the level above
    ske_offset = -1
    next_level_pts = zeros(DIMENSION, 0)
    if changed_or_not # size(store_size, 2) > 1 check to see if number of machines involved changed between level
        cutoff = (machine_idx - 1) * box_per
        loc_store = zeros(Int64, 1, cutoff)
        loc_store .= store_size[:, 1 : cutoff]
        ske_offset = 0
        if !isempty(loc_store)
            ske_offset = cumsum(loc_store, dims=2)[end]
        end
        idx_array = zeros(Int64, 1, length(new_points))

        idx_array[:, :] .= store_size[:, cutoff + 1 : cutoff + box_per]
        idx_array = reshape(idx_array, (length(new_points, )))
        next_level_pts = zeros(DIMENSION, sum(idx_array))
        idx_array = [0; cumsum(idx_array)]
        for fut in eachindex(new_points)
            next_level_pts[:, idx_array[fut] + 1 : idx_array[fut + 1]] .= fetch(new_points[fut])
        end
    end

    #f_node_dict, points_dict = prefetch_points_level_transition(tree_darray, factor_level_below[PART_ONE], coordinates)
    f_node_dict, points_dict = fetch(fetched_messages[PART_ONE])


    # build up indices on the new level by combining children indices
    # below_local = localpart(factor_level_below[PART_ONE])
    # below_offset = -1
    # if !isempty(below_local)
    #     below_offset = local_idx[1] - 1
    # end
    count = ske_offset + 1
    for i = 1 : length(factor_nodes)
        t_node = tree_nodes[i]
        f_node = factor_nodes[i]
        children = t_node.children;

        # build temp using on machine or off machine procedure
        temp = Array{factor_info_1, 1}(undef, length(children))
        for c in eachindex(children)
            temp_node = get(f_node_dict, children[c], -1)
            if temp_node == -1
                @assert false
                #@assert below_offset != -1
                #temp[c] = below_local[children[c] - below_offset]
            else
                temp[c] = temp_node
            end
        end

        if ske_offset == -1
            for k = 1 : length(children)
                # REQUIRES COMMUNICATION BETWEEN TOP LEVEL AND ONE LEVEL BELOW
                append!(f_node.ske_index, temp[k].ske_index)
            end
        else
            # set up idx range for next level
            f_node.idx_range = [ske_offset + 1, ske_offset + size(next_level_pts, 2)]
            for k = 1 : length(children)
                # REQUIRES COMMUNICATION BETWEEN TOP LEVEL AND ONE LEVEL BELOW
                temp_len = length(temp[k].ske_index)
                append!(f_node.ske_index, collect(count : count + temp_len - 1))
                count += temp_len
            end
        end
    end
    #end


    total_com = communication_node[]
    factor_nodes = localpart(factor_darray[PART_TWO])
    # prefetch points
    # t1 = time()
    # prefetch_list = [@spawnat probe_list[p] prefetch_points(tree_below, factor_level_below[PART_ONE], coordinates, 1) for p = 1 : length(probe_list)]
    # for fut in prefetch_list
    #     wait(fut)
    # end
    # println("level transition prefetch time: ", time() - t1)
    prefetch_list = []
    if changed_or_not
        prefetch_list = fetched_messages[PART_TWO]
    else 
        prefetch_list = [fetched_messages[PART_ONE]]
    end
    
    @assert typeof(prefetch_list) <: Array{Future, 1}
    # spawn in reverse order in case probelist[1] is the current machine
    #to_fetch = [@spawnat probe_list[p] optimize_level_transition!(factor_level_below, prefetch_list[p], length(factor_nodes), p, probe_list, current_layer, idx_range, coordinates, kfun, intgrl) for p = length(probe_list) : -1 : 1]
    to_fetch = Array{Future, 1}(undef, length(probe_list))
    @sync for p = length(probe_list) : -1 : 1
        @async to_fetch[length(probe_list) - p + 1] = remotecall_wait(optimize_level_transition!, probe_list[p], factor_level_below, prefetch_list[p], length(factor_nodes), p, probe_list, current_layer, idx_range, coordinates, kfun, intgrl)  
    end
    #optimize_level_transition!(factor_level_below, tree_below, length(factor_nodes), 1, probe_list, current_layer, idx_range, coordinates, intgrl)
    #optimize_level_transition!(factor_level_below, prefetch_list[1], length(factor_nodes), 1, probe_list, current_layer, idx_range, coordinates, kfun, intgrl)
    # reverse again so everything is back to regular order
    #for p = 1 : length(probe_list)
    #    fut = @spawnat probe_list[p] optimize_level_transition!(factor_level_below, prefetch_list[p], length(factor_nodes), p, probe_list, current_layer, idx_range, coordinates, kfun, intgrl) 
    #    append!(total_com, fetch(fut))
    #end
    t1 = time()
    for fut in reverse(to_fetch)
        append!(total_com, fetch(fut))
    end
    #println("level transition get update from lower time: ", time() - t1)

    # update data structure
    for i in eachindex(factor_nodes)
        f_node = factor_nodes[i]
        f_node.modified = total_com[i].sidx_array
        f_node.update_close = total_com[i].smaller_idx
    end



    return next_level_pts
end

@everywhere function optimize_level_transition!(factor_level_below::Array, prefetch::Future, total_length::Int64, match_p_idx::Int64,
    probe_list::Array{Int64, 1}, current_layer::Int64, idx_range::Array{Int64, 1}, coordinates::DArray{Float64, 2}, kfun::Function, intgrl::Float64)

    local_idx = [-1, -1]
    box_per = BOXES_PER_NODE
    offset = idx_range[1] - 1

    #f_node_dict, points_dict = prefetch_points(tree_below, factor_level_below[PART_ONE], coordinates, 1)
    f_node_dict, points_dict = fetch(prefetch)
    fetched_map = (f_node_dict, points_dict)
    ret_com = communication_node[]
    s1 = 0
    # accumulate updates from lower level
    for i = 1 : total_length
        p_idx = Int64(ceil(i / (total_length / length(probe_list))))
        chi_offset = offset * box_per + Int64((p_idx - 1) * total_length * box_per / length(probe_list))

        if length(probe_list) == 1
            p_idx = 1
        end

        # if p_idx doesn't match, then skip
        if p_idx != match_p_idx
            continue;
        end

        cur_idx = i + offset
        near_field = get_neighbor(Int64(cur_idx + (box_per ^ current_layer - 1) / (box_per - 1)), current_layer, DIMENSION)
        nlist = [cur_idx; near_field]

        count = 0
        for k in nlist
            if k >= cur_idx || !in_range(k, idx_range)
                count += 1
            end
        end


        actuallist = []
        children = zeros(Int64, count, box_per)
        count = 1
        for (idx_k, k) in enumerate(nlist)
            if k >= cur_idx || !in_range(k, idx_range)
                children[count, :] = (k * box_per - (box_per - 1)) : k * box_per
                push!(actuallist, k)
                count += 1
            end
        end

        klist = children[1, :]
        combined_mat = Array{Array{PRECISION, 2}, 1}(undef, 0)
        t1 = @timed begin
        for c = 1 : size(children, 1)
            clist = children[c, :]
            result_mat = []
            if box_per == 4
                result_mat = [build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[1] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                 build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[2] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                 build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[3] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                 build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[4] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                ]
            else
                result_mat = [build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[1] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                 build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[2] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                 build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[3] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                 build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[4] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                 build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[5] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                 build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[6] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                 build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[7] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                 build_matrix(factor_level_below, Array{tree_node, 2}(undef, 0, 0), Array{communication_node, 2}(undef, 0, 0),
                    zeros(0, 0), clist, klist[8] - chi_offset, kfun, chi_offset, [local_idx[1]; local_idx[end]], coordinates, intgrl, [], fetched_map)[1];
                ]
            end
            push!(combined_mat, result_mat)
        end
        end
        push!(ret_com, communication_node{Array{Int64, 1}, Array{Array, 1}}(actuallist, [], combined_mat, []))
        s1 += t1.time
    end

    #println("level transition build matrix time: ", s1)
    return ret_com
end




# clear out communication and modified
@everywhere function clear_auxilliary!(communication_darray::DArray, factor_darray::Array)
    communication_parts = localpart(communication_darray)
    factor_parts = localpart(factor_darray[PART_TWO])
    for i = 1 : length(factor_parts)
        f_node = factor_parts[i]
        c_node = communication_parts[i]
        # will later be collected by GC(hopefully)
        for j in eachindex(f_node.modified)
            f_node.modified[j] = []
        end
        f_node.modified = []
        f_node.update_close = []
        c_node.smaller_idx = []
        c_node.bigger_idx = []
        c_node.sidx_array = []
        c_node.bidx_array = []
    end
end

@everywhere function clear_communication!(communication_darray::DArray)
    communication_parts = localpart(communication_darray)
    for i in eachindex(communication_parts)
        c_node = communication_parts[i]
        # will later be collected by GC(hopefully)
        c_node.smaller_idx = []
        c_node.bigger_idx = []
        c_node.sidx_array = []
        c_node.bidx_array = []
    end
end


# build matrix if the form A(idx, nlist)
# LENLIST DOESN'T INCLUDE LENGTH OF PROXY
@everywhere function build_matrix(factor_darray::Array, tree_nodes::Array{tree_node, 2}, communication_nodes::Array{communication_node, 2},
    proxy_mat::AbstractArray, nlist::Array{Int64, 1}, row_box::Int64,
        kfun::Function, offset::Int64, idx_range::Array{Int64, 1}, coordinates::DArray{Float64, 2}, intgrl::Float64,
            optional_mat=Array{PRECISION, 2}[], optional_map_tuple=(Dict(), Dict()))

    factor_nodes = localpart(factor_darray[PART_ONE])
    factor_nodes_second = localpart(factor_darray[PART_TWO])
    idx_range = localindices(factor_darray[PART_ONE])[2]
    idx_range = [idx_range[1]; idx_range[end]]
    f_node_dict = optional_map_tuple[1]
    points_dict = optional_map_tuple[2]
    # current node
    row_box += offset # add offset to get index with respect to layer
    my_f_node = factor_nodes[row_box - offset]
    my_t_node = nothing
    if !isempty(tree_nodes)
        my_t_node = tree_nodes[row_box - offset]
    end
    lenlist = zeros(Int64, length(nlist));


    # fetch all the nodes
    f_node_list = factor_info_1[]
    colcount = size(proxy_mat, 2);
    for c = 1 : length(nlist)
        current_idx = nlist[c]
        if in_range(current_idx, idx_range)
            push!(f_node_list, factor_nodes[current_idx - offset])
        else
            #@assert false
            mapped_f_node = get(f_node_dict, current_idx, -1)
            if mapped_f_node == -1
                @assert false
                push!(f_node_list, factor_darray[PART_ONE][current_idx])
            else
                push!(f_node_list, mapped_f_node)
            end
        end

        lenlist[c] = length(f_node_list[end].ske_index)
        colcount += lenlist[c]
    end

    # preallocate space for matrix
    given_mat = []
    if !isempty(optional_mat)
        # if size is adequate, just use the one passed in
        if size(optional_mat, 1) >= length(my_f_node.ske_index) && size(optional_mat, 2) >= colcount
            given_mat = optional_mat
        else
            given_mat = zeros(Int64(ceil(length(my_f_node.ske_index) * 1.1)), Int64(ceil(colcount * 1.5)))
        end
    else
        given_mat = zeros(length(my_f_node.ske_index), colcount)
    end
    if size(proxy_mat, 2) != 0
        given_mat[1 : length(my_f_node.ske_index), 1 : size(proxy_mat, 2)] .= proxy_mat
    end

    prev_colcount = size(proxy_mat, 2) + 1
    colcount = size(proxy_mat, 2) + 1


    for c = 1 : length(nlist)
        current_idx = nlist[c]
        smallerbox = min(current_idx, row_box)
        biggerbox = max(current_idx, row_box)


        # initialize
        other_f_node = f_node_list[c]
        f_node = factor_nodes_second[row_box - offset]
        if current_idx < row_box
            if in_range(current_idx, idx_range)
                f_node = factor_nodes_second[current_idx - offset]
            # else
            #     # if node is interior, then by algorithmic design, its interact with nodes on other machines are not modified
            #     if my_t_node.boundary[1] == 1
            #         f_node = factor_darray[PART_TWO][current_idx]
            #     end

            end
        end

        update = []
        #if !(current_idx < row_box) || in_range(current_idx, idx_range) || (my_t_node.boundary[1] == 1)
        update = f_node.update_close
        #end
        # if the smaller box is on a different machine, then due to double storage for boundary,
        # we can look for the smallerbox in update instead
        do_flip = true
        if current_idx < row_box && !in_range(current_idx, idx_range)
            idx = findall(x -> x == smallerbox, update)
            do_flip = false
        else
            idx = findall(x -> x == biggerbox, update)
        end



        # if column empty, go to next column
        if isempty(other_f_node.ske_index)
            continue;
        end

        if !isempty(idx)
            # set up col and row
            idx = idx[1]
            colrange = 1 : size(f_node.modified[idx], 2);
            rowrange = 1 : size(f_node.modified[idx], 1);
            if smallerbox != row_box && do_flip
                colrange = 1 : size(f_node.modified[idx], 1);
                rowrange = 1 : size(f_node.modified[idx], 2);
            end
            # change to ske if size mismatch
            if length(my_f_node.ske_index) != length(rowrange)
                rowrange = rowrange[my_f_node.relative_ske_index];
            end
            if length(other_f_node.ske_index) != length(colrange)
                colrange = colrange[other_f_node.relative_ske_index];
            end

            if smallerbox == row_box || !do_flip
                @views given_mat[1 : length(my_f_node.ske_index), colcount : colcount + length(colrange) - 1] .= f_node.modified[idx][rowrange, colrange];
            else
                @views given_mat[1 : length(my_f_node.ske_index), colcount : colcount + length(colrange) - 1] .= f_node.modified[idx][colrange, rowrange]';
            end
            colcount = colcount + length(colrange);
        else
            # otherwise, it's at default location, use kfun
            mapped_points = get(points_dict, current_idx, -1)

            @views temp_pointer = given_mat[1 : length(my_f_node.ske_index), colcount : colcount + length(other_f_node.ske_index) - 1]
            if mapped_points == -1
                if !in_range(current_idx, idx_range)
                    @assert false
                end
                kfun(coordinates, my_f_node.ske_index, other_f_node.ske_index, intgrl, temp_pointer);
            else
                kfun(coordinates, my_f_node.ske_index, mapped_points, intgrl, temp_pointer);
            end
            colcount = colcount + length(other_f_node.ske_index);
        end
        #=
        # if smallerbox on another machine, accumulate all the updates in communication node that are potentially not updated yet
        @views temp_pointer = given_mat[1 : length(my_f_node.ske_index), prev_colcount : colcount - 1]

        if !in_range(smallerbox, idx_range)
            # this also implies that the biggerbox is ON CURRENT MACHINE SINCE BOTH CAN'T BE ON ANOTHER AT THE SAME TIME

            @assert !isempty(tree_nodes)
            close_list = filter(x -> in_range(x, idx_range), tree_nodes[biggerbox - offset].close)

            for neigh in close_list
                c_node = communication_nodes[neigh - offset]

                smaller_idx = c_node.smaller_idx
                bigger_idx = c_node.bigger_idx
                idx = intersect(findall(x -> x == smallerbox, smaller_idx), findall(x -> x == biggerbox, bigger_idx))

                if !isempty(idx)
                    @assert length(idx) == 1
                    idx = idx[1]
                    # since current on machine node is bidx, want to flip the two multiplied matrix
                    @views BLAS.gemm!('N', 'T', -1.0, c_node.bidx_array[idx], c_node.sidx_array[idx], 1.0, temp_pointer)
                end

            end
        end
        prev_colcount = colcount
        =#

    end


    return given_mat, lenlist
end

@everywhere function create_communication_struct(layer::Int64, probes_length::Int64)

    box_per = BOXES_PER_NODE
    total_boxes = Int64((box_per ^ layer) / probes_length) # num of boxes this process gets

    # declare communication list
    communication_list = Array{communication_node, 2}(undef, 1, total_boxes)
    for i = 1 : total_boxes
        communication_list[i] = communication_node{Array{Int64, 1}, Array{Array, 1}}(zeros(Int64, 0), zeros(Int64, 0), Array{Array, 1}(undef, 0), Array{Array, 1}(undef, 0))
    end
    return communication_list
end


@everywhere function generate_rightside(refer::DArray, cur_layer::Int64, max_layer::Int64, right_darray::AbstractArray)
    factor_local = localpart(refer)
    generated_right = localpart(right_darray)
    loop_len = length(factor_local)
    rightside = Array{rightside_block{Array{PRECISION, 1}}, 2}(undef, 1, loop_len)
    if cur_layer == max_layer
        for i = 1 : loop_len
            f_node = factor_local[i]
            #rightside[i] = rightside_block{Array{PRECISION, 1}}(ones(length(factor_local[i].ske_index)))
            rightside[i] = rightside_block{Array{PRECISION, 1}}(generated_right[f_node.ske_index  .- f_node.idx_range[PART_ONE] .+ 1])
        end
    else
        for i = 1 : loop_len
            rightside[i] = rightside_block{Array{PRECISION, 1}}(zeros(0))
        end
    end
    return rightside
end

# must be called after generate_rightside
@everywhere function fill_rightside!(to_fill::DArray, factor_darray::DArray, right_darray::AbstractArray)
    factor_local = localpart(factor_darray)
    generated_right = localpart(right_darray)
    to_fill_local = localpart(to_fill)
    loop_len = length(factor_local)
    for i = 1 : loop_len
        f_node = factor_local[i]
        r_node = to_fill_local[i]
        r_node.right_side[f_node.relative_ske_index] .= generated_right[f_node.ske_index  .- f_node.idx_range[PART_ONE] .+ 1]
        r_node.right_side[f_node.relative_red_index] .= generated_right[f_node.red_index  .- f_node.idx_range[PART_ONE] .+ 1]
    end
end

@everywhere function accumulate_update_solve!(tree_darray::DArray, factor_darray::Array, v::DArray, communication_darray::DArray, fetched_data::Future)
    tree_nodes = localpart(tree_darray)
    factor_nodes = localpart(factor_darray[PART_ONE])
    rightside_nodes = localpart(v)
    communication_nodes = localpart(communication_darray)
    local_idx = localindices(tree_darray)[2]
    # this offset is used to index into the local parts, IMPORTANT TO REMEMBER
    offset = local_idx[1] - 1
    idx_range = [local_idx[1]; local_idx[end]]
    check_repeat = Dict()
    c_dict = fetch(fetched_data)
    for i in eachindex(tree_nodes)
        if tree_nodes[i].boundary[1] == 1
            accumulate_update_solve!(tree_nodes[i], factor_nodes, rightside_nodes, communication_darray, [local_idx[1]; local_idx[end]], check_repeat, c_dict)
        end
    end
end

# # function for transmitting updates between boundaries of processes for solves
@everywhere function accumulate_update_solve!(t_node::tree_node, factor_nodes::Array, rightside_nodes::Array, communication_darray::DArray, idx_range::Array{Int64, 1}, check_repeat::Dict, c_dict::Dict)
    offset = idx_range[1] - 1
    close_list = t_node.close
    # find the ones not on current machine, which will be the ones that need to resort to the communication method in the first place
    candidate_list = filter(x->(!in_range(x, idx_range) && !haskey(check_repeat, x)), close_list)

    for candidate in candidate_list
        get!(check_repeat, candidate, 1) # add it to map
        #c_node = communication_darray[candidate]
        c_node = get(c_dict, candidate, false)
        @assert c_node != false
        relevant_idx = findall(x->in_range(x, idx_range), c_node.smaller_idx)
        smaller_list = c_node.smaller_idx[relevant_idx]

        for i = 1 : length(smaller_list)
            f_node = factor_nodes[smaller_list[i] - offset]
            v_node = rightside_nodes[smaller_list[i] - offset]
            update_v = c_node.sidx_array[relevant_idx[i]]
            if length(update_v) == length(v_node.right_side)
                v_node.right_side .-= update_v

            else
                @views v_node.right_side[f_node.relative_ske_index] .-= update_v
            end
        end

    end


end

@everywhere function level_transition_solve!(tree_darray::DArray, v::DArray, v_below::DArray, factor_level_below::DArray, indicate::String, fetched_data::Future)
    box_per = BOXES_PER_NODE

    if indicate == "forward"
        tree_nodes = localpart(tree_darray)
        rightside_nodes = localpart(v)
        f_dict, v_dict = fetch(fetched_data) 
        # build up indices on the new level by combining children indices
        for i = 1 : length(rightside_nodes)
            t_node = tree_nodes[i]
            v_node = rightside_nodes[i]
            children = t_node.children;
            #f_temp = factor_level_below[children]
            #v_temp = v_below[children]
            f_temp = factor_info_1[]
            v_temp = rightside_block[]
            for ch in children
                f_map = get(f_dict, ch, false)
                v_map = get(v_dict, ch, false)
                @assert f_map != false
                @assert v_map != false
                push!(f_temp, f_map)
                push!(v_temp, v_map)
            end
            resize!(v_node.right_side, 0)
            for k = 1 : length(children)

                # REQUIRES COMMUNICATION BETWEEN TOP LEVEL AND ONE LEVEL BELOW
                append!(v_node.right_side, v_temp[k].right_side[f_temp[k].relative_ske_index])
            end

        end

    else

        tree_nodes = localpart(tree_darray)
        rightside_nodes = localpart(v_below)
        factor_nodes = localpart(factor_level_below)
        v_dict = fetch(fetched_data)
        # load values into children from parent
        count = 1
        init_parent = tree_nodes[1].parent
        #parent_v = v[init_parent] # access children's parent
        parent_v = get(v_dict, init_parent, false)
        @assert parent_v != false
        for i = 1 : length(rightside_nodes)
            v_node = rightside_nodes[i]
            f_node = factor_nodes[i]
            v_node.right_side[f_node.relative_ske_index] = parent_v.right_side[count : count + length(f_node.relative_ske_index) - 1]

            count += length(f_node.relative_ske_index)
            if mod(i, box_per) == 0 && i != length(rightside_nodes)
                count = 1
                init_parent += 1
                #parent_v = v[init_parent]
                parent_v = get(v_dict, init_parent, false)
                @assert parent_v != false
            end
        end
    end
end




@everywhere function forward_solve!(tree_darray::DArray, factor_darray::Array, communication_darray::DArray, v::DArray, current_layer::Int64, on_boundary::Int64; fetched_data=nothing)
    s1 = 0
    t1 = time()

    tree_nodes = localpart(tree_darray)
    factor_nodes = localpart(factor_darray[PART_ONE])
    factor_nodes_third = localpart(factor_darray[PART_THREE])
    rightside_nodes = localpart(v)
    communication_nodes = localpart(communication_darray)
    local_idx = localindices(tree_darray)[2]
    # this offset is used to index into the local parts, IMPORTANT TO REMEMBER
    offset = local_idx[1] - 1
    idx_range = [local_idx[1]; local_idx[end]]
    node_mapping = Dict{}()
    
    if fetched_data !== nothing
        fetch_time = time()
        node_mapping = fetch(fetched_data)
        println("fetch time for forward_solve: ", time() - fetch_time)
    end
    
    for i = 1 : length(factor_nodes)
        f_node = factor_nodes[i]
        f_node_third = factor_nodes_third[i]
        t_node = tree_nodes[i]
        c_node = communication_nodes[i]
        v_node = rightside_nodes[i]
        s_idx = f_node.relative_ske_index
        r_idx = f_node.relative_red_index

        # if it's not part of the current boundary or no points, skip

        if isempty(s_idx) || isempty(r_idx) || t_node.boundary[1] != on_boundary
            continue
        end

        # if factoring boundary, accumulate updates first
        if on_boundary == 1
            #accumulate_update_solve!(t_node, factor_nodes, rightside_nodes, communication_darray, idx_range, check_repeat)
        end
        #------------------- Apply V^-1
        # apply U_t
        temp_r_idx = f_node_third.X_RR.U' \ (v_node.right_side[r_idx] - f_node_third.T' * v_node.right_side[s_idx])

        # 2. apply L
        # subtract vec from Sidx and closefield idx
        @views v_node.right_side[s_idx] .-= f_node_third.X_SR * temp_r_idx
        closefield = t_node.close
        neighbor_idx = f_node.neighbor_idx # not including self since closefield doesn't include self
        
        computed_vec = f_node_third.X_NR * temp_r_idx
       
        v_node.right_side[r_idx] = temp_r_idx

        if current_layer > 1
            for j = 1 : length(closefield)

                box_idx = closefield[j]
                # if no points in skeleton index, then skip
                if neighbor_idx[j + 1] - neighbor_idx[j] == 0
                    continue;
                end

                if in_range(box_idx, idx_range)
                    other_v_node = rightside_nodes[box_idx - offset]
                    other_f_node = factor_nodes[box_idx - offset]
                    # if no points in redundant, skip
                    if isempty(other_f_node.red_index)
                        continue;
                    end

                    # if processed, used compressed ske, otherwise use all points in node


                    if neighbor_idx[j + 1] - neighbor_idx[j] == length(other_f_node.ske_index)
                        other_ske_idx = other_f_node.relative_ske_index
                        @views other_v_node.right_side[other_ske_idx] .-=  computed_vec[neighbor_idx[j] + 1 : neighbor_idx[j + 1]]
                    else
                        @views other_v_node.right_side .-= computed_vec[neighbor_idx[j] + 1 : neighbor_idx[j + 1]]
                    end

                else
                    #@assert false
                    other_f_node = get(node_mapping, box_idx, false)
                    @assert other_f_node != false
                    #other_f_node = factor_darray[PART_ONE][box_idx]
                    # if no points in redundant, skip
                    if isempty(other_f_node.red_index)
                        continue;
                    end
                    # if processed, used compressed ske, otherwise use all points in node
                    if (neighbor_idx[j + 1] - neighbor_idx[j]) == length(other_f_node.ske_index)
                        other_ske_idx = other_f_node.relative_ske_index
                        @views push!(c_node.sidx_array, computed_vec[neighbor_idx[j] + 1 : neighbor_idx[j + 1]])
                    else
                        @views push!(c_node.sidx_array, computed_vec[neighbor_idx[j] + 1 : neighbor_idx[j + 1]])
                    end
                    push!(c_node.smaller_idx, box_idx)
                end

            end
        end

    end
    s1 += (time() - t1)
    println("s1: ", s1)
end


@everywhere function backward_solve!(tree_darray::DArray, factor_darray::Array, v::DArray, current_layer::Int64, on_boundary::Int64; fetched_data=nothing)

    s2 = 0
    t1 = time()
    tree_nodes = localpart(tree_darray)
    factor_nodes = localpart(factor_darray[PART_ONE])
    factor_nodes_third = localpart(factor_darray[PART_THREE])
    rightside_nodes = localpart(v)
    local_idx = localindices(tree_darray)[2]
    # this offset is used to index into the local parts, IMPORTANT TO REMEMBER
    offset = local_idx[1] - 1
    idx_range = [local_idx[1]; local_idx[end]]
    tuple_of_maps = (nothing, nothing)


    if fetched_data !== nothing
        fetch_time = time()
        tuple_of_maps = fetch(fetched_data)
        println("fetch time for backward_solve: ", time() - fetch_time)
    end
    node_mapping = tuple_of_maps[1]
    v_mapping = tuple_of_maps[2]
    
    for i = length(factor_nodes) : -1 : 1

        f_node = factor_nodes[i]
        f_node_third = factor_nodes_third[i]
        t_node = tree_nodes[i]
        v_node = rightside_nodes[i]
        s_idx = f_node.relative_ske_index
        r_idx = f_node.relative_red_index

        # skip
        if isempty(s_idx) || isempty(r_idx) || t_node.boundary[1] != on_boundary
            continue
        end

        closefield = t_node.close
        neighbor_idx = f_node.neighbor_idx # not including self since closefield doesn't include self
        
        @views vec_ridx_1 = f_node_third.X_SR' * v_node.right_side[s_idx]
        # s2 += (time() - t1)
        vec_ridx_2 = zeros(neighbor_idx[end]) # accumulate the vector, later used for X_RN multiplication

        if current_layer > 1
            for j = 1 : length(closefield)
                box_idx = closefield[j]
                # if no points in skeleton index, then skip
                if neighbor_idx[j + 1] - neighbor_idx[j] == 0
                    continue;
                end
                other_f_node = []
                other_v_node = []
                if in_range(box_idx, idx_range)
                    other_f_node = factor_nodes[box_idx - offset]
                    other_v_node = rightside_nodes[box_idx - offset]
                else
                    #@assert false
                    other_f_node = get(node_mapping, box_idx, false)
                    other_v_node = get(v_mapping, box_idx, false)
                    @assert other_v_node != false
                    @assert other_f_node != false
                    #other_f_node = factor_darray[PART_ONE][box_idx]
                    #other_v_node = v[box_idx]
                end


                if neighbor_idx[j + 1] - neighbor_idx[j] == length(other_f_node.relative_ske_index)
                    #@views vec_ridx_2 .+= f_node_third.X_NR[neighbor_idx[j] + 1 : neighbor_idx[j + 1], :]' * other_v_node.right_side[other_f_node.relative_ske_index]
                    vec_ridx_2[neighbor_idx[j] + 1 : neighbor_idx[j + 1]] = other_v_node.right_side[other_f_node.relative_ske_index]
                else
                    #@views vec_ridx_2 .+= f_node_third.X_NR[neighbor_idx[j] + 1 : neighbor_idx[j + 1], :]' * other_v_node.right_side
                    vec_ridx_2[neighbor_idx[j] + 1 : neighbor_idx[j + 1]] = other_v_node.right_side
                end


            end
        end
        # t1 = time()
        @views v_node.right_side[r_idx] .= f_node_third.X_RR.U \ (v_node.right_side[r_idx] .- vec_ridx_1 .- f_node_third.X_NR' * vec_ridx_2)
        @views v_node.right_side[s_idx] .-= f_node_third.T * v_node.right_side[r_idx]
        

    end
    s2 += (time() - t1)
    println("s2: ", s2)
end


@everywhere function solver!(probes::Array, probe_num_progression::Array, tree_parts::Array, factor_nodes::Array, communication_nodes::Array, modified_nodes::Array, 
    x_matrix_nodes::Array, rightside_nodes::Array, recovered_point_permutation::Vector, example_vec::DArray, given_vec::Union{DArray, Vector})
    layer = length(tree_parts) - 1
    num_probes = length(probes)
    OFF_BOUNDARY = 0
    ON_BOUNDARY = 1
    box_per = BOXES_PER_NODE
    # load the right side into rightside_nodes
    all_p = procs(factor_nodes[end])
    all_p = reshape(all_p, length(all_p))
    all_p = sort(all_p)
    fill_wait = Array{Future, 1}(undef, length(all_p))
    distributed_away = []
    if typeof(given_vec) <: DArray
        #fill_wait = [@spawnat p fill_rightside!(rightside_nodes[end], factor_nodes[end], given_vec) for p in all_p] # assumes that if given_vec is a DArray, then it's already permuted
        @sync for p_idx in eachindex(all_p)
            p = all_p[p_idx]
            @async fill_wait[p_idx] = remotecall_wait(fill_rightside!, p, rightside_nodes[end], factor_nodes[end], given_vec)
        end
    else
        # permute the vector first
        given_vec_permuted = given_vec[recovered_point_permutation]
        indices = example_vec.cuts[PART_ONE]
        t1 = time()
        each_part = Array{Future, 1}(undef, length(all_p))
        @sync for (p_idx, p) in enumerate(all_p)
            #@async each_part[p_idx] = wait(@spawnat p given_vec_permuted[indices[p_idx] : indices[p_idx + 1] - 1])
            @async each_part[p_idx] = remotecall_wait(eval, p, given_vec_permuted[indices[p_idx] : indices[p_idx + 1] - 1])
        end
        distributed_away = DArray(each_part)
        println("build distributed_away: ", time() - t1)
        #fill_wait = [@spawnat p fill_rightside!(rightside_nodes[end], factor_nodes[end], distributed_away) for p in all_p] # distributed it like the example_vec
        @sync for p_idx in eachindex(all_p)
            p = all_p[p_idx]
            @async fill_wait[p_idx] = remotecall_wait(fill_rightside!, p, rightside_nodes[end], factor_nodes[end], distributed_away)
        end
    end
    
    if !(typeof(given_vec) <: DArray)
        close(distributed_away)
    end

    #---------------------------------------------------------- MUST CLEAR COMMUNICATION_NODES FIRST!
    #
    # perform forward solve
    com1 = 0
    com2 = 0
    forward_level_com = 0
    forward_level_calc = 0
    backward_level_com = 0
    backward_level_calc = 0
    t1 = @timed for cur_level = layer : -1 : 0
        
        cur_num_probes = probe_num_progression[cur_level + 1]
        step_size = Int64(num_probes / cur_num_probes)
        probe_range = range(1, step=step_size, length=cur_num_probes)
        total_factor_nodes = [factor_nodes[cur_level + 1], modified_nodes[cur_level + 1], x_matrix_nodes[cur_level + 1]]
        # if not at bottom level, apply transition first
        if cur_level != layer
            fetched_data = Array{Future, 1}(undef, length(probe_range))
            st = time()
            @sync for (p_idx, p) in enumerate(probe_range)
                @async fetched_data[p_idx] = remotecall_wait(prefetch_points_level_transition_solve, probes[p], tree_parts[cur_level + 1], factor_nodes[cur_level + 2], rightside_nodes[cur_level + 2], rightside_nodes[cur_level + 1], "forward")
            end 
            
            @sync for (p_idx, p) in enumerate(probe_range)
                @async remotecall_wait(level_transition_solve!, probes[p], tree_parts[cur_level + 1], rightside_nodes[cur_level + 1], rightside_nodes[cur_level + 2], factor_nodes[cur_level + 2], "forward", fetched_data[p_idx])
            end     
            com1 = com1 + (time() - st)      
            forward_level_com += (time() - st)
        end



        if cur_level != 0
            @sync for p in eachindex(probes)
                @async remotecall_wait(announce_forward_solve_level, probes[p], cur_level)
            end
            #forward_solve!(tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], rightside_nodes[cur_level + 1], cur_level, OFF_BOUNDARY)
            st = time()
            @sync for p in probe_range
                @async remotecall_wait(forward_solve!, probes[p], tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], rightside_nodes[cur_level + 1], cur_level, OFF_BOUNDARY)
            end  
            forward_level_calc += time() - st
            # synchronization point


            # pick out processes according to 4 coloring or 8 coloring (jie hua fa)
            subset_probes = Int64(max(cur_num_probes / box_per, 1))

            for coloring = 1 : min(box_per, cur_num_probes)
                launch_idx = range(coloring, stop=cur_num_probes, step=box_per)
                fetched_data = Array{Future, 1}(undef, subset_probes)
                st = time()
                @sync for p = 1 : subset_probes
                    @async fetched_data[p] = remotecall_wait(prefetch_nodes_for_solve, probes[probe_range[launch_idx[p]]], tree_parts[cur_level + 1], factor_nodes[cur_level + 1], factor_nodes[cur_level + 1], use_factor=true)
                end 
                com1 = com1 + (time() - st)

                st = time()
                @sync for p = 1 : subset_probes
                    @async remotecall_wait(forward_solve!, probes[probe_range[launch_idx[p]]], tree_parts[cur_level + 1], total_factor_nodes, communication_nodes[cur_level + 1], rightside_nodes[cur_level + 1], cur_level, ON_BOUNDARY, fetched_data=fetched_data[p])
                end 
                forward_level_calc += time() - st

                # accumulate the updates immediately after each color had been processed
                fetched_data = Array{Future, 1}(undef, length(probe_range))
                st = time()
                @sync for (p_idx, p) in enumerate(probe_range)
                    @async fetched_data[p_idx] = remotecall_wait(prefetch_nodes_for_solve, probes[p], tree_parts[cur_level + 1], factor_nodes[cur_level + 1], communication_nodes[cur_level + 1], use_optional=true, element_type=communication_node)
                end 
                
                @sync for (p_idx, p) in enumerate(probe_range)
                    @async remotecall_wait(accumulate_update_solve!, probes[p], tree_parts[cur_level + 1], total_factor_nodes, rightside_nodes[cur_level + 1], communication_nodes[cur_level + 1], fetched_data[p_idx])
                end 
                

                # clear communication to avoid double update
                @sync for p = 1 : subset_probes
                    @async remotecall_wait(clear_communication!, probes[probe_range[launch_idx[p]]], communication_nodes[cur_level + 1])
                end
                com1 = com1 + (time() - st)
                forward_level_com += (time() - st)
            end
            
        end
        println("forward communication time for level ", cur_level, " is: ", forward_level_com)
        forward_level_com = 0
        println("forward computation time for level ", cur_level, " is: ", forward_level_calc)
        forward_level_calc = 0

    end



    # solve with first level block
    # ASSUMES THAT rightside_nodes[1] and modified_node[1] IS ON FIRST PROBE
    
    t2 = @timed remotecall_wait(middle_solve, probes[1], rightside_nodes, x_matrix_nodes)
    #fetch(@spawnat probes[1] rightside_nodes[1][1].right_side .= x_matrix_nodes[1][1].X_RR.U \ (x_matrix_nodes[1][1].X_RR.L \ rightside_nodes[1][1].right_side))
    #@spawnat probes[1] rightside_nodes[1][1].right_side .= modified_nodes[1][1].modified[1] \ rightside_nodes[1][1].right_side

    # perform backward solve, DO OFF_BOUNDARY THEN ON_BOUNDARY BECAUSE THE ORDER NEEDS TO BE SWAPPED
    t3 = @timed for cur_level = 0 : layer
        

        cur_num_probes = probe_num_progression[cur_level + 1]
        step_size = Int64(num_probes / cur_num_probes)
        probe_range = range(1, step=step_size, length=cur_num_probes)
        total_factor_nodes = [factor_nodes[cur_level + 1], modified_nodes[cur_level + 1], x_matrix_nodes[cur_level + 1]]
        


        if cur_level != 0
            
            # pick out processes according to 4 coloring or 8 coloring
            subset_probes = Int64(max(cur_num_probes / box_per, 1))
            for coloring = min(box_per, cur_num_probes) : -1 : 1
                launch_idx = range(coloring, stop=cur_num_probes, step=box_per)
                fetched_data = Array{Future, 1}(undef, subset_probes)
                st = time()
                @sync for p = subset_probes : -1 : 1
                    @async fetched_data[p] = remotecall_wait(prefetch_nodes_for_solve, probes[probe_range[launch_idx[p]]], tree_parts[cur_level + 1], factor_nodes[cur_level + 1], rightside_nodes[cur_level + 1], use_factor=true, use_optional=true, element_type=rightside_block)
                end 
                com2 = com2 + (time() - st)
                backward_level_com += (time() - st)

                st = time()
                @sync for p = subset_probes : -1 : 1
                    @async remotecall_wait(backward_solve!, probes[probe_range[launch_idx[p]]], tree_parts[cur_level + 1], total_factor_nodes, rightside_nodes[cur_level + 1], cur_level, ON_BOUNDARY, fetched_data=fetched_data[p])
                end 
                backward_level_calc += (time() - st)
            end

            st = time()
            @sync for p in reverse(probe_range)
                @async remotecall_wait(backward_solve!, probes[p], tree_parts[cur_level + 1], total_factor_nodes, rightside_nodes[cur_level + 1], cur_level, OFF_BOUNDARY)
            end 
            backward_level_calc += (time() - st)
            #backward_solve!(tree_parts[cur_level + 1], total_factor_nodes, rightside_nodes[cur_level + 1], cur_level, ON_BOUNDARY)
        end

        # if not at bottom level, apply transition first
        # different from the forward case, transition is applied first before the first backward solve
        if cur_level != layer
            @sync for p in eachindex(probes)
                @async remotecall_wait(announce_backward_solve_level, probes[p], cur_level)
            end
            cur_num_probes = probe_num_progression[cur_level + 2]
            step_size = Int64(num_probes / cur_num_probes)
            probe_range = range(1, step=step_size, length=cur_num_probes)
            fetched_data = Array{Future, 1}(undef, length(probe_range))
            st = time()
            @sync for (p_idx, p) in enumerate(probe_range)
                @async fetched_data[p_idx] = remotecall_wait(prefetch_points_level_transition_solve, probes[p], tree_parts[cur_level + 2], factor_nodes[cur_level + 2], rightside_nodes[cur_level + 2], rightside_nodes[cur_level + 1], "backward")
            end 
            
            @sync for (p_idx, p) in enumerate(probe_range)
                @async remotecall_wait(level_transition_solve!, probes[p], tree_parts[cur_level + 2], rightside_nodes[cur_level + 1], rightside_nodes[cur_level + 2], factor_nodes[cur_level + 2], "backward", fetched_data[p_idx])
            end 
            com2 = com2 + (time() - st)
            backward_level_com += (time() - st)
            # check_vec=[]
            # for i = 1 : 4
            #     append!(check_vec, rightside_nodes[2][i].right_side)
            # end
            # println(check_vec)
            # file = matopen("vec.mat", "w")
            # write(file, "comp_vec", check_vec)
            # close(file)
        end
        
        println("backward communication time for level ", cur_level, " is: ", backward_level_com)
        backward_level_com = 0
        println("backward computation time for level ", cur_level, " is: ", backward_level_calc)
        backward_level_calc = 0
    end


    println("total forward time: ", t1.time)
    println("total middle time: ", t2.time)
    println("total backward time: ", t3.time)
    println("forward communication time: ", com1)
    println("backward communication time: ", com2)
    println("total gc time: ", t1.gctime + t2.gctime + t3.gctime)
    println("total communication time: ", com1 + com2)
    println("total computation time: ", t1.time + t2.time + t3.time - com1 - com2)
    println("total overall time: ", t1.time + t2.time + t3.time)

    # On each machine, do the internal permutation that changes from block organization back to natural ordering
    temp_future = Array{Future, 1}(undef, length(all_p))
    if typeof(given_vec) <: Vector
        #temp_future = [@spawnat p permute_solution(factor_nodes[layer + 1], rightside_nodes[layer + 1], example_vec) for p in all_p]
        @sync for p_idx in eachindex(all_p)
            p = all_p[p_idx]
            @async temp_future[p_idx] = remotecall_wait(permute_solution, p, factor_nodes[layer + 1], rightside_nodes[layer + 1], example_vec)
        end
        leftside = DArray(temp_future)
        sol = convert(Vector, leftside)
        recovered_point_permutation_t = zeros(Int64, length(recovered_point_permutation))
        recovered_point_permutation_t[recovered_point_permutation] = 1 : length(recovered_point_permutation)
        return sol[recovered_point_permutation_t]
    end

    return []

end

@everywhere function middle_solve(rightside_nodes::Array, x_matrix_nodes::Array)
    rightside_nodes[1][1].right_side .= x_matrix_nodes[1][1].X_RR.U \ (x_matrix_nodes[1][1].X_RR.L \ rightside_nodes[1][1].right_side)
end


function test_function1()
    index = []
    i = 1
    if i == 1
        index = zeros(10000)
    end
    return index
end





function test_function()
    dis = dzeros((100000, 1), workers()[1:4], [4, 1])
    @time a = dis[1 : 10000, :]
    println(size(a))
    @time for i = 1 : 10000
        a[i, :]
    end
    # @time for i = 1 : 10000
    #     dis[1:100]
    # end
    # println(typeof(dis[1]))
    # println(typeof(dis[1:2]))
    b = zeros(10000, 1)
    @time b .= a

    @time for i = 1 : 10000
        b[i, 1]
    end

    m = rand(10000, 10000)

    @time @views m * b[:,1]

    ras = [@spawnat p rand(2500, 1) for p in workers()[1:4]]
    ras = reshape(ras, (4,1))
    hey = DArray(ras)
    @time hey[1:10000, :]
    @time hey[1]
    @time hey[1,:]
    println(typeof(hey[1]))
    println(typeof(hey[1,:]))
    #@time b .= hey

    #@assert isequal(b, hey)
end

#precompile(perform_factorization!, (DArray, DArray, DArray, Int64, Function, Function, Int64))
end




# this function prefetches kernel points and ske_index
@everywhere function prefetch_points(tree_darray::DArray, factor_darray::DArray, coordinates::DArray, on_boundary::Int64)
    tree_nodes = localpart(tree_darray)
    factor_nodes = localpart(factor_darray)
    local_idx = localindices(tree_darray)[2]
    # this offset is used to index into the local parts, IMPORTANT TO REMEMBER
    offset = local_idx[1] - 1
    idx_range = [local_idx[1]; local_idx[end]]

    # add the indices not on current machine to a set
    non_local_indices_set = Set{Int64}()
    for i in eachindex(factor_nodes)
        f_node = factor_nodes[i]
        t_node = tree_nodes[i]
        if t_node.boundary[1] != on_boundary
            continue;
        end

        far_field = t_node.interact
        near_field = t_node.close

        non_local_indices = filter(x -> !in_range(x, idx_range), [far_field; near_field])
        for non_local_index in non_local_indices
            push!(non_local_indices_set, non_local_index)
        end
    end

    return get_nodes_and_points(non_local_indices_set, factor_darray, coordinates)

    #=
    # sorting may help with access
    non_local_indices_array = sort(collect(non_local_indices_set))
    #println("non local indices array: ", non_local_indices_array)
    # grab non local f_nodes
    non_local_f_nodes = Array{factor_info_1, 2}(undef, 1, length(non_local_indices_array))
    #println("num f_node: ", length(non_local_indices_array))
    s1 = time()
    # goobypls = Array{factor_info_1, 1}(undef, length(factor_darray))
    # goobypls .= factor_darray[:]
    non_local_f_nodes .= factor_darray[:, non_local_indices_array]
    non_local_f_nodes = reshape(non_local_f_nodes, (length(non_local_f_nodes), ))
    #println("f_node fetch time: ", time() - s1)


    f_node_dict = Dict{Int64, factor_info_1}()
    for i in eachindex(non_local_indices_array)
        get!(f_node_dict, non_local_indices_array[i], non_local_f_nodes[i])
    end

    # grab non local kernel points
    all_indices = Int64[]
    each_box_len = Int64[]
    for i in eachindex(non_local_f_nodes)
        f_node = non_local_f_nodes[i]
        append!(all_indices, f_node.ske_index)
        append!(each_box_len, length(f_node.ske_index))
    end
    cumulative_box_len = [0; cumsum(each_box_len)]
    non_local_points = Array{Float64, 2}(undef, DIMENSION, length(all_indices))
    s1 = time()
    # goobypls = Array{Float64, 2}(undef, DIMENSION, size(coordinates, 2))
    # goobypls .= coordinates[:, :]
    non_local_points .= coordinates[:, all_indices]
    #println("points fetch time: ", time() - s1)


    points_dict = Dict{Int64, Array}()
    for i in eachindex(non_local_indices_array)
        get!(points_dict, non_local_indices_array[i], non_local_points[:, cumulative_box_len[i] + 1 : cumulative_box_len[i + 1]])
    end

    return f_node_dict, points_dict
    =#
end

# check if the box is on the machine associated with a specific color
@everywhere function verify_machine(x::Int64, determine_machine::Array)
    machine_num = Int64(ceil(x / determine_machine[PART_ONE]))
    color = mod(machine_num, BOXES_PER_NODE)
    if color == 0
        color = BOXES_PER_NODE
    end
    return color == determine_machine[PART_TWO]
end

# used for accumulate_update
@everywhere function prefetch_communication_nodes(tree_darray::DArray, communication_darray::DArray, determine_machine::Array)
    check_repeat = Dict()
    tree_nodes = localpart(tree_darray)
    local_idx = localindices(tree_darray)[2]
    # this offset is used to index into the local parts, IMPORTANT TO REMEMBER
    offset = local_idx[1] - 1
    non_local_indices_set = Set{Int64}()
    idx_range = [local_idx[1]; local_idx[end]]
    on_boundary = 1

    for i in eachindex(tree_nodes)
        t_node = tree_nodes[i]
        near_field = t_node.close
        if t_node.boundary[1] != on_boundary
            continue;
        end
        non_local_indices = filter(x -> !in_range(x, idx_range) && verify_machine(x, determine_machine), near_field)
        for non_local_index in non_local_indices
            push!(non_local_indices_set, non_local_index)
        end
    end
    println("set length: ", length(non_local_indices_set))
    ret = fetch_from_array!(sort(collect(non_local_indices_set)), communication_darray, communication_node)[1]
    return ret

end

# used for accumulate update
@everywhere function prefetch_bigger(candidate_list_future::Future, factor_darray::DArray, coordinates::DArray)
    
    candidate_list = fetch(candidate_list_future)
    if isempty(candidate_list)
        return (Dict(), Dict())
    end
    factor_nodes = localpart(factor_darray)
    local_idx = localindices(factor_darray)[2]
    idx_range = [local_idx[1]; local_idx[end]]

    # prefetch the bigger ones that aren't on current machine
    non_local_indices_set = Set{Int64}()
    for candidate in eachindex(candidate_list)
        c_node = candidate_list[candidate]
        relevant_idx = findall(x->in_range(x, idx_range), c_node.smaller_idx)
        bigger_list = c_node.bigger_idx[relevant_idx]
        # filter out the in range ones to get the ones that aren't on current machine
        non_local_indices = filter(x->!in_range(x, idx_range), bigger_list)
        for non_local_index in non_local_indices
            push!(non_local_indices_set, non_local_index)
        end
    end

    return get_nodes_and_points(non_local_indices_set, factor_darray, coordinates)


end

# for level transition. CURRENTLY FETCH ALL CHILDREN REGARDLESS OF WHETHER ON MACHINE OR NOT
@everywhere function prefetch_points_level_transition(tree_darray::DArray, factor_level_below::DArray, coordinates::DArray)
    tree_nodes = localpart(tree_darray)
    total_children = Int64[]
    local_idx = localindices(tree_darray)[2]
    idx_range = [local_idx[1]; local_idx[end]]

    # local_idx = localindices(factor_level_below)[2]
    # idx_range = -1
    # if !isempty(local_idx)
    #     idx_range = [local_idx[1]; local_idx[end]]
    # end

    all_indices_set = Set{Int64}()
    for i in eachindex(tree_nodes)
        t_node = tree_nodes[i]
        near_field = t_node.close
        append!(total_children, t_node.children)
        box_per = length(t_node.children)

        # add children of near field that are not on this machine and haven't been added
        for ele in near_field
            if !(ele in all_indices_set) && !in_range(ele, idx_range)
                append!(total_children, collect((ele - 1) * box_per + 1 : ele * box_per))
                push!(all_indices_set, ele)
            end
        end


    end

    return get_nodes_and_points(total_children, factor_level_below, coordinates)

end



@everywhere function get_nodes_and_points(non_local_indices_set::Union{Set{Int64}, Array{Int64, 1}}, factor_darray::DArray, coordinates::DArray)
    non_local_indices_array = []
    if typeof(non_local_indices_set) <: Set{Int64}
        non_local_indices_array = sort(collect(non_local_indices_set))
    else
        non_local_indices_array = non_local_indices_set
    end


    non_local_f_nodes, non_local_indices_array = fetch_from_array!(non_local_indices_array, factor_darray, factor_info_1)

    f_node_dict = Dict{Int64, factor_info_1}()
    for i in eachindex(non_local_indices_array)
        get!(f_node_dict, non_local_indices_array[i], non_local_f_nodes[i])
    end
    

    # grab non local kernel points
    all_indices = Int64[]
    each_box_len = Int64[]
    for i in eachindex(non_local_f_nodes)
        f_node = non_local_f_nodes[i]
        append!(all_indices, f_node.ske_index)
        append!(each_box_len, length(f_node.ske_index))
    end
    cumulative_box_len = [0; cumsum(each_box_len)]

    non_local_points = fetch_from_array!(all_indices, coordinates, Float64, true)[1]


    points_dict = Dict{Int64, Array}()
    for i in eachindex(non_local_indices_array)
        get!(points_dict, non_local_indices_array[i], non_local_points[:, cumulative_box_len[i] + 1 : cumulative_box_len[i + 1]])
    end

    return f_node_dict, points_dict
end



@everywhere function fetch_from_array!(non_local_indices_array::Array{Int64, 1}, access_array::DArray, element_type, is_coordinate::Bool=false)
    if isempty(non_local_indices_array)
        return ([], [])
    end
    
    machine_array_mapping = Dict{Int64, Array}()
    machine_channel_mapping = Dict{Int64, RemoteChannel}()
    machine_index_mapping = Dict{Int64, Int64}()
    original_index_mapping = Dict{Int64, Array}()
    box_per = BOXES_PER_NODE
    result_array = Array{element_type, 2}(undef, size(access_array, 1), length(non_local_indices_array))

   
    # first, figure out which machine each individual index belongs to
    involved_processes = procs(access_array)
    offset_list = deepcopy(access_array.cuts[PART_TWO])

    for (i_idx, i) in enumerate(non_local_indices_array)
        machine_idx = DistributedArrays.locate(access_array, 1, i)[PART_TWO]

        # if !is_coordinate
        #     machine_idx = Int64(ceil(i / current_layer_num_box))
        # else 
        #     machine_idx = DistributedArrays.locate(access_array, 1, i)[PART_TWO]
        # end
        
        
        process_id = involved_processes[machine_idx]
        get!(machine_index_mapping, process_id, machine_idx) # inverse mapping from process id to relative machine number
        val = get(machine_array_mapping, process_id, "no_key")
        
        if val == "no_key"
            # if no key, create new key value pair
            get!(machine_array_mapping, process_id, [i])
            get!(machine_channel_mapping, process_id, RemoteChannel(process_id))
            if is_coordinate
                get!(original_index_mapping, process_id, [i_idx])
            end
        else
            # otherwise, add the new element to the existing array
            push!(val, i)
            if is_coordinate
                val_other = get(original_index_mapping, process_id, "no_key")
                push!(val_other, i_idx)
            end
        end
    end
    
    all_keys = collect(keys(machine_channel_mapping))
    
    
    # go ahead and ask the relevant neighboring processes to put their data in the appropriate channel
    
    list_size = zeros(Int64, length(all_keys))
    
    @sync for (p_idx, p) in enumerate(all_keys)
        corresponding_index = machine_array_mapping[p]
        corresponding_channel = machine_channel_mapping[p]
        offset = offset_list[machine_index_mapping[p]] - 1
        adjusted_index = corresponding_index .- offset
        list_size[p_idx] = length(corresponding_index)
        
        if p != myid()
            #@spawnat p wait(@tspawnat PART_THREE put!(corresponding_channel, localpart(access_array)[:, adjusted_index]))
	    #@async wait(@spawnat p put!(corresponding_channel, localpart(access_array)[:, adjusted_index]))
        else
            #put!(corresponding_channel, localpart(access_array)[:, adjusted_index])
        end
    end

   
    
    # at this point, the fetched results should be in the appropriate channels
    
    order_perm = sortperm(list_size, rev=true)
    all_keys = all_keys[order_perm]
    list_size = list_size[order_perm]
    cumulative_size = [0; cumsum(list_size)]

    
    permute_back = zeros(Int64, length(non_local_indices_array))
    # loop through and fetch the data from channel, could use multithreading here
    t1 = time()
    @sync for p_idx = 1 : length(all_keys)
        p = all_keys[p_idx]
        
        if is_coordinate
            permute_back[cumulative_size[p_idx] + 1 : cumulative_size[p_idx + 1]] .= original_index_mapping[p]
        end
	    
        corresponding_index = machine_array_mapping[p]
        offset = offset_list[machine_index_mapping[p]] - 1
        adjusted_index = corresponding_index .- offset
        
        #@async thread_fetch_job!(result_array, cumulative_size[p_idx] + 1 : cumulative_size[p_idx + 1], chnl)
        thread_id = mod(p_idx, box_per)
	    if thread_id == 0 || thread_id == 1
            thread_id = box_per
        end
        #@async result_array[:, cumulative_size[p_idx] + 1 : cumulative_size[p_idx + 1]] .= remotecall_fetch(thread_fetch_job, p, access_array, adjusted_index)
        
        @async result_array[:, cumulative_size[p_idx] + 1 : cumulative_size[p_idx + 1]] .= remotecall_fetch(thread_fetch_job, p, access_array, adjusted_index)
        
        #@tspawnat 4 thread_fetch_job!(result_array, cumulative_size[p_idx] + 1 : cumulative_size[p_idx + 1], chnl)
    end

    
    println("communication time: ", time() - t1)
    
    
    if size(result_array, 1) == 1
        result_array = reshape(result_array, length(result_array))
    end

    # set up non_local_indices_array to make it match the index of result_array
    if !is_coordinate
        for (p_idx, p) in enumerate(all_keys)
            corresponding_index = machine_array_mapping[p]
            non_local_indices_array[cumulative_size[p_idx] + 1 : cumulative_size[p_idx + 1]] .= corresponding_index
        end
    else
        # if input is coordinate, permute it back 
        permute_back_t = zeros(Int64, length(permute_back))
        permute_back_t[permute_back] .= 1 : length(permute_back_t)
        result_array = result_array[:, permute_back_t]
    end

      
    comm_message_size = 0
    for i in eachindex(result_array)
        comm_message_size += sizeof(result_array[i])
    end
    if !comm_solve
        println("communication count during factorization: ", comm_counter, ", size of factorization message: ", comm_message_size)
    else
        println("communication count during solve: ", comm_counter, ", size of solve message: ", comm_message_size)
    end        
    global comm_counter += 1
    
    return result_array, non_local_indices_array
end

@everywhere function thread_fetch_job(access_array::DArray, adjusted_index::Array{Int64, 1}) where T
    return localpart(access_array)[:, adjusted_index]
end




# communication method used as part of the solve
@everywhere function prefetch_nodes_for_solve(tree_darray::DArray, factor_darray::DArray, optional_darray::DArray; use_factor::Bool=false, use_optional=false, element_type=Nothing)
    tree_nodes = localpart(tree_darray)
    factor_nodes = localpart(factor_darray)
    local_idx = localindices(tree_darray)[2]
    # this offset is used to index into the local parts, IMPORTANT TO REMEMBER
    offset = local_idx[1] - 1
    idx_range = [local_idx[1]; local_idx[end]]

    # add the indices not on current machine to a set
    non_local_indices_set = Set{Int64}()
    for i in eachindex(factor_nodes)
        f_node = factor_nodes[i]
        t_node = tree_nodes[i]
        if t_node.boundary[1] != PART_ONE
            continue;
        end

        non_local_indices = filter(x -> !in_range(x, idx_range), t_node.close)
        for non_local_index in non_local_indices
            push!(non_local_indices_set, non_local_index)
        end
    end

    return get_nodes_solve(non_local_indices_set, factor_darray, optional_darray, use_factor, use_optional, element_type)

end

@everywhere function get_nodes_solve(non_local_indices_set::Union{Set{Int64}, Array{Int64, 1}}, factor_darray::DArray, optional_darray::DArray, use_factor::Bool, use_optional::Bool, element_type)
    if !use_factor && !use_optional
        return []
    end

    non_local_indices_array = []
    if typeof(non_local_indices_set) <: Set{Int64}
        non_local_indices_array = sort(collect(non_local_indices_set))
    else
        non_local_indices_array = non_local_indices_set
    end
    


    if use_factor && !use_optional

        non_local_f_nodes, non_local_indices_array = fetch_from_array!(non_local_indices_array, factor_darray, factor_info_1)
        f_node_dict = Dict{Int64, factor_info_1}()
        for i in eachindex(non_local_indices_array)
            get!(f_node_dict, non_local_indices_array[i], non_local_f_nodes[i])
        end
        return f_node_dict

    elseif !use_factor && use_optional

        non_local_optional_nodes, non_local_indices_array = fetch_from_array!(non_local_indices_array, optional_darray, element_type)
        optional_node_dict = Dict{Int64, element_type}()
        for i in eachindex(non_local_indices_array)
            get!(optional_node_dict, non_local_indices_array[i], non_local_optional_nodes[i])
        end
        return optional_node_dict
    
    else
        non_local_f_nodes, non_local_indices_array = fetch_from_array!(non_local_indices_array, factor_darray, factor_info_1)
        f_node_dict = Dict{Int64, factor_info_1}()
        for i in eachindex(non_local_indices_array)
            get!(f_node_dict, non_local_indices_array[i], non_local_f_nodes[i])
        end

        non_local_indices_array = convert(Array{Int64, 1}, non_local_indices_array)

        non_local_optional_nodes, non_local_indices_array = fetch_from_array!(non_local_indices_array, optional_darray, element_type)
        optional_node_dict = Dict{Int64, element_type}()
        for i in eachindex(non_local_indices_array)
            get!(optional_node_dict, non_local_indices_array[i], non_local_optional_nodes[i])
        end
        return f_node_dict, optional_node_dict
    end



end

@everywhere function prefetch_points_level_transition_solve(tree_darray::DArray, factor_level_below::DArray, v_below::DArray, v::DArray, direction::String)
    tree_nodes = localpart(tree_darray)
    total_children = Int64[]
    total_parent = Int64[]
    local_idx = localindices(tree_darray)[2]
    idx_range = [local_idx[1]; local_idx[end]]

    if direction == "forward"
        for i in eachindex(tree_nodes)
            t_node = tree_nodes[i]
            append!(total_children, t_node.children)
        end
        return get_nodes_solve(total_children, factor_level_below, v_below, true, true, rightside_block)
    else 
        for i in eachindex(tree_nodes)
            t_node = tree_nodes[i]
            append!(total_parent, t_node.parent)
        end
        return get_nodes_solve(total_parent, factor_level_below, v, false, true, rightside_block)
    end

end

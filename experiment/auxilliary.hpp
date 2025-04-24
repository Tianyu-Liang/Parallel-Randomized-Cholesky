
// #include <cuda_runtime.h>
// #include <cuda/atomic>
// #include <curand_kernel.h>
// #include <cooperative_groups.h>
#include "pre_process.hpp"




template <typename type_int, typename type_data>
struct Node
{
    /* data */
    type_int count;
    type_int start;
    type_int true_col_id; // may be useful if non default permutation
    int64_t prev;
    type_data sum;

    Node(){prev = -1;}

    Node(type_int _count, type_int _start, type_int _true_col_id) : count(_count), start(_start), true_col_id(_true_col_id) {prev = -1;}
};


template <typename type_int, typename type_data>
struct Output
{
    /* data */
    type_int row;
    type_data value;
 //   type_data cumulative_value;
    type_data forward_cumulative_value;
    int multiplicity;
    

    Output(){
        multiplicity = 0;
    }

    Output(type_int _row, type_data _value, type_int _multiplicity) : row(_row), value(_value), multiplicity(_multiplicity) 
    {
    }
};



template <typename type_int, typename type_data>
struct Edge
{
    /* data */
    type_int row;
    type_data value;
    type_data forward_cumulative_value;
    int multiplicity;
    type_int sampled_row;
    type_data sampled_value;
    int64_t prev;

    Edge(){
        multiplicity = 0;
        prev = -1;
    }

    Edge(type_int _row, type_data _value, type_int _multiplicity) : row(_row), value(_value), multiplicity(_multiplicity){
        prev = -1;
    }
};


// sparse matrix on gpu
template <typename type_int, typename type_data>
struct sparse_matrix_device
{
    /* data */
    type_int num_rows;                   // Number of rows in the matrix
    type_int num_cols;                   // Number of columns in the matrix
    type_data *values;          // Non-zero values
    type_int *row_indices; // Row indices for each non-zero value
    type_int *col_ptrs;    // Column pointers
    bool *merge; // flag to determine whether the entry is merged or not
};





// a function to search for edge updates
template <typename type_int, typename type_data>
type_int search_for_updates(int job_id, type_int map_size, type_int num_of_original_nnz, Node<type_int, type_data> *device_node_list, 
        type_int *device_output_position_idx, type_int *output_start_array, 
            Edge<type_int, type_data> *device_edge_map, std::vector<Edge<type_int, type_data>> &local_work_space)
{
    

    // check if the number of updates is greater than 0, synchronization ensure that the original value doesn't get updated while reading
    std::atomic_ref<type_int> original_update_edge_count(device_node_list[job_id].count);
    std::atomic_ref<type_int> position_idx_ref(*device_output_position_idx);
    // 1 thread fetch some position index and broadcast it through shared memory
    // make two copies here, first entry will be original, second entry is the location decider
    
    //cuda::atomic_ref<type_int, cuda::thread_scope_device> atomic_position_idx(*device_output_position_idx);
    // output_start_array[squad_id * 4] = atomic_position_idx.fetch_add((original_update_edge_count + 
    //     num_of_original_nnz) * 2);
    // make sure there is at least 1 entry in order to store the diagonal entry
    //type_int size_needed = max((original_update_edge_count.load() + num_of_original_nnz) * 2, 1);
    type_int size_needed = original_update_edge_count.load() + num_of_original_nnz + 1;
    output_start_array[0] = position_idx_ref.fetch_add(size_needed);
    output_start_array[1] = 0;
    output_start_array[2] = 0;
    output_start_array[3] = 0;
    
    if(output_start_array[0] + size_needed >= map_size)
    {
        printf("need more edge map capacity, program terminates\n");
        assert(false);
    }
    if(sizeof(type_int) == 4)
    {
        if(output_start_array[0] > INT_MAX - 10000000)
        {
            printf("WARNING, MIGHT INT OVERFLOW");
            assert(false);
        }
    }
    
    // allocate more local space if necessary
    if(size_needed > local_work_space.size())
    {
        local_work_space.resize(size_needed * 2);
    }   


    // update device_node_list with the necessary locations to find the edges
    
    device_node_list[job_id].start = output_start_array[0];
    
    


    if(original_update_edge_count.load() > 0)
    {
        // this processes the first element
     
        type_int next_index = device_node_list[job_id].prev;
        type_int fill_index_start = 0;
        // if(next_index < 0 || next_index >= 80000000)
        // {
        //     printf("next index: %d, job_id: %d, num: %d, original: %d\n", next_index, job_id, original_update_edge_count.load(), num_of_original_nnz);
        // }
        local_work_space[fill_index_start] = Edge<type_int, type_data>(device_edge_map[next_index].sampled_row, 
           device_edge_map[next_index].sampled_value, 1);
        
        fill_index_start++;
        
        // rest of elements
        while(device_edge_map[next_index].prev != -1)
        {
            next_index = device_edge_map[next_index].prev;
            local_work_space[fill_index_start] = Edge<type_int, type_data>(device_edge_map[next_index].sampled_row, 
                device_edge_map[next_index].sampled_value, 1);
            
            fill_index_start++;
        }
        output_start_array[1] = fill_index_start;
    
    }


    // return number of items found
    return output_start_array[1];
}




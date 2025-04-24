

#include <chrono>
#include <cub/cub.cuh>
#include <curand_kernel.h>
#include "solver.hpp"


typedef int custom_idx;









template <typename type_int>
void writeVectorToFile(const std::vector<type_int>& vec, const std::string& filename) 
{
    std::ofstream outFile(filename);
    
    if (!outFile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }
    
    for (const type_int& element : vec) {
        outFile << element << "\n";
    }
    
    outFile.close();
}


__global__ void dummy()
{

}

// Kernel to construct the struct using placement new
template <typename type_int, typename type_data>
__global__ void perform_factorization_device(const sparse_matrix_device<type_int, type_data> spmat_device, type_int map_size,
    Edge<type_int, type_data> *device_edge_map, type_int output_size, Output<type_int, type_data> *global_device_factorization_output,
       type_int *device_min_dependency_count, Node<type_int, type_data> *device_node_list, type_int *device_output_position_idx, type_int *queue_device, type_int *rand_vec_device) 
{
    // fixed sized sorting optimization

    //using MergeSort_Mini = cub::BlockMergeSort<Output<type_int, type_data>, 16, 1>;
    using MergeSort_Mini = cub::WarpMergeSort<Output<type_int, type_data>, 1, 16>;
    using BlockMergeSort_One = cub::BlockMergeSort<Output<type_int, type_data>, THREADS_PER_BLOCK, 1>;
    using BlockMergeSort_Two = cub::BlockMergeSort<Output<type_int, type_data>, THREADS_PER_BLOCK, 2>;
    using BlockMergeSort_Four = cub::BlockMergeSort<Output<type_int, type_data>, THREADS_PER_BLOCK, 4>;

   

    // Allocate shared memory for BlockMergeSort

    __shared__ typename MergeSort_Mini::TempStorage sort_mini_st;
    __shared__ typename BlockMergeSort_One::TempStorage sort_one_st;
    __shared__ typename BlockMergeSort_Two::TempStorage sort_two_st;
    __shared__ typename BlockMergeSort_Four::TempStorage sort_four_st;

    constexpr int squad_size = WARPS_PER_SQUAD * THREADS_PER_WARP;
    using BlockScan = cub::BlockScan<type_data, squad_size>;
    using BlockScanInt = cub::BlockScan<type_int, squad_size>;
    __shared__ typename BlockScan::TempStorage temp_storage_scan[1];
    __shared__ typename BlockScanInt::TempStorage temp_storage_scan_int[1];
    __shared__ type_int output_start_array[THREADS_PER_BLOCK / WARPS_PER_SQUAD / THREADS_PER_WARP * 4];
    int absolute_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_squad_id = absolute_id / WARPS_PER_SQUAD / THREADS_PER_WARP;
    // each squad is responsible for eliminating 1 column
    int squad_id = threadIdx.x / WARPS_PER_SQUAD / THREADS_PER_WARP;
    // warp id
    int warp_id = threadIdx.x / THREADS_PER_WARP;
    // lane id is the index within the squad
    int lane_id = threadIdx.x % (WARPS_PER_SQUAD * THREADS_PER_WARP);
    // warp lane id is the index within the warp
    int warp_lane_id = threadIdx.x % THREADS_PER_WARP;
    // gap is the total number of squads
    int gap = gridDim.x * blockDim.x / WARPS_PER_SQUAD / THREADS_PER_WARP;
    int run_count = 0;
    


    type_int num_cols = spmat_device.num_cols;
    
    // assign the job id, will consider using a scheduler in the future
    // if id is within valid range of queue'
    type_int queue_access_index = global_squad_id;
    int32_t job_id = -1;
    type_int queue_size = atomicAdd(&queue_device[num_cols], 0);
 
    if(queue_access_index < queue_size)
    {
        job_id = atomicAdd(&queue_device[queue_access_index], 0);
        // this means that even though the size was modified, the actual job haven't been inserted
        if(job_id == 0 && queue_access_index != 0)
        {
            job_id = -1;
        }
    }

    // set up state and seed
    curandState state;
    curand_init(absolute_id * 10000, absolute_id, 0, &state);

    // if(threadIdx.x % 32 == 0)
    // {
    //     printf("absolute_id: %d, global_squad_id: %d, squad_id: %d, warp_id: %d, lane_id: %d, gap: %d\n", absolute_id, global_squad_id, 
    //         squad_id, warp_id, lane_id, gap);
    // }

    // skip num_cols - 1 since the last column is empty
    //int static_id = 0;


    // shared memory for column elimination
    // __shared__ Output<type_int, type_data> local_output[LOCAL_CAPACITY]; 
    __shared__ unsigned char raw_shared_buffer[LOCAL_CAPACITY * sizeof(Output<type_int, type_data>)];
    Output<type_int, type_data>* local_output =
        reinterpret_cast<Output<type_int, type_data>*>(raw_shared_buffer);
    // reset local output's multiplicity, IMPORTANT
    for(type_int i = lane_id; i < LOCAL_CAPACITY; i += squad_size)
    {
        local_output[i].multiplicity = 0; 
        local_output[i].row = 0; 
        local_output[i].value = 0.0; 
    }
    if(squad_size == 32)
    {
        //__threadfence();
        __syncwarp();
    }
    else
    {
        __syncthreads();
    }
    Output<type_int, type_data> *device_factorization_output = global_device_factorization_output;

    while(queue_access_index < num_cols)
    {
        
        unsigned long long total_start = clock64();
        if(job_id == -1)
        {
         
            queue_size = atomicAdd(&queue_device[num_cols], 0);
            
            
            if(queue_access_index < queue_size)
            {
                
                job_id = atomicAdd(&queue_device[queue_access_index], 0);
                // this means that even though the size was modified, the actual job haven't been inserted
                if(job_id == 0 && queue_access_index != 0)
                {
                    job_id = -1;
                }

                
               
            }

            
            
            
            continue;
        }

      // job_id = static_id;
      //  static_id++;
        //curand_init(0, job_id, 0, &state);
     
        // if(lane_id == 0)
        //     {
        //         printf("squad id: %d, job id stuckccccccccc: %d, queue size: %d, access index: %d\n", squad_id, job_id, queue_size, queue_access_index);
        //     }

        // job successfully queued up, move to next search location
        queue_access_index += gap;
        // job id is the last column, skip
        if(job_id == num_cols - 1)
        {
            // SET LAST COLUMN
            if(lane_id == 0)
            {
                type_int last_col_loc = atomicAdd(device_output_position_idx, 1);
                device_node_list[num_cols - 1].start = last_col_loc;
                device_node_list[num_cols - 1].count = 0;
                device_node_list[num_cols - 1].sum = 0.0;
            }
            __syncthreads();
            job_id = -1;
            continue;
        }
        
        // check if column is ready for factorization
        //cuda::atomic_ref<type_int, cuda::thread_scope_device> atomic_dependency_count(device_min_dependency_count[job_id]);
        
        
        type_int atomic_dependency_count = atomicAdd(&device_min_dependency_count[job_id], 0);
        //cuda::atomic_ref<type_int, cuda::thread_scope_device> atomic_dependency_count(device_min_dependency_count[job_id]);

        // if(lane_id == 0 && atomic_dependency_count == 1)
        // {
        //     printf("squad id: %d, job id stuckccccccccc: %lld, atomic_dependency_count: %lld\n", squad_id, job_id, atomic_dependency_count);
        // }
       
       
        // if(lane_id == 0 && job_id == 1836)
        // {
        //     printf("queue_access_index: %d, queue size: %d\n", queue_access_index, queue_size);
        // }
       
        // if(job_id % 2 == 1)
        // {
            
        //     printf("job id stuckccccccccc: %lld, atomic_dependency_count: %lld\n", job_id, atomic_dependency_count);
        // }
        // if(job_id % 1000 == 0)
        // {
        //     printf("check point reached at: %d\n", job_id);
        // }

       
        
        while(atomic_dependency_count > 0)
        {
            
           
            //continue;
        }
       

        /* search for updates, get location information */ 
        unsigned long long insert_start = clock64();
            
        type_int left_bound_idx = spmat_device.lower_tri_cptr_start[job_id];
        type_int num_of_original_nnz = spmat_device.lower_tri_cptr_end[job_id] - left_bound_idx;
        type_int update_neighbor_count = search_for_updates<type_int, type_data>(job_id, squad_id, squad_size, lane_id, map_size, num_of_original_nnz,
            device_node_list, device_output_position_idx, output_start_array, device_edge_map, global_device_factorization_output, num_cols, local_output, rand_vec_device);
        
        type_int total_neighbor_count = num_of_original_nnz + update_neighbor_count;
        unsigned long long insert_end = clock64();

        // if no entry in this column
        if(total_neighbor_count == 0)
        {
            
            if(lane_id == 0)
            {
                device_node_list[job_id].count = 0;
                device_node_list[job_id].sum = 0.0;
            }
            __syncthreads();
            job_id = -1;
            continue;
        }

       

       
        {

            type_int edge_start = output_start_array[squad_id * 4];

            // make sure to swap to local space based on output_start_array flag
            if(output_start_array[squad_id * 4 + 3])
            {
                device_factorization_output = local_output;
                edge_start = 0;
            }
            else
            {
                device_factorization_output = global_device_factorization_output;
            }
           
            /* perform merging */

            // 1. read in the nonzeros in the original input and sort entire input first based on row value
            // if(lane_id == 0)
            // {
            //     printf("before total: %d, local ad: %lx, use ad: %lx, equal local: %d\n", total_neighbor_count, &local_output[0], device_factorization_output, device_factorization_output == local_output);
            // }
            for(type_int i = edge_start + update_neighbor_count + lane_id; i < edge_start + total_neighbor_count; i += squad_size)
            {
                device_factorization_output[i] = Output<type_int, type_data>(spmat_device.row_indices[left_bound_idx + i - edge_start - update_neighbor_count],
                    spmat_device.values[left_bound_idx + i - edge_start - update_neighbor_count] * -1, 1);
            }

            if(squad_size == 32)
            {
                __syncwarp();
            }
            else
            {
                __syncthreads();
            }
            // if(lane_id == 0)
            // {
            //     printf("after total: %d\n", total_neighbor_count);
            // }
            // if(lane_id == 0)
            //     printf("id: %d, num original neighbor: %d, update neighbor: %d, total nei: %d\n", job_id + 1, num_of_original_nnz, update_neighbor_count, total_neighbor_count);
            
            
            // if(lane_id == 0 && job_id == 6)
            // {
            //     printf("update row: %d\n", device_factorization_output[edge_start].row);
            //     printf("update row next: %d\n", device_factorizationa_output[edge_start + 1].row);
            //     printf("update mul: %d\n", device_factorization_output[edge_start].multiplicity);
            //     printf("update mul next: %d\n", device_factorization_output[edge_start + 1].multiplicity);
            //     printf("total_neighbor_count here: %d\n", total_neighbor_count);
            // }
            unsigned long long sort_start = clock64();
            if(total_neighbor_count <= 8)
            {
                if(lane_id == 0)
                {
                    run_count += 1;
                }
                odd_even_sort(device_factorization_output, edge_start, edge_start + total_neighbor_count, lane_id, compare_row);
          
            }
            else if(total_neighbor_count <= 16)
            {
                             
                // Obtain a segment of consecutive items that are blocked across threads
              
                Output<type_int, type_data> local_sort_array[1];
            
                int val_per = 1;
        

            

                for(int seg = 0; seg < val_per; seg++)
                {
                    if(lane_id * val_per + seg < total_neighbor_count)
                    {
                        local_sort_array[seg] = device_factorization_output[edge_start + lane_id * val_per + seg];
                    }
                    else
                    {
                        if(sizeof(type_int) == 4)
                        {
                            local_sort_array[seg].row = INT_MAX;
                        }
                        else
                        {
                            local_sort_array[seg].row = LLONG_MAX;
                        }
                    }
                }
                // if(lane_id == 0)
                // {
                //     printf("job id: %d, neighbor count: %d\n", job_id, total_neighbor_count);
                // }
                
                // printf("blockidx: %d, id: %d, ele1: %d\n", blockIdx.x, lane_id, local_sort_array[0].row);
                //printf("blockidx: %d, id: %d, ele2: %d\n", blockIdx.x, lane_id, local_sort_array[1].row);
                
                
                if(warp_id == 0 && lane_id < 16)
                {
                    MergeSort_Mini(sort_mini_st).Sort(local_sort_array, compare_row_reference);
                }
                    
                
                
                // min(total_neighbor_count - lane_id * val_per, val_per), Output<type_int, type_data>(INT_MAX, 0.0, 0));
                //odd_even_sort(device_factorization_output, edge_start, edge_start + total_neighbor_count, lane_id, compare_row);

                for(int seg = 0; seg < val_per; seg++)
                {
                    if(lane_id * val_per + seg < total_neighbor_count)
                    {
                        device_factorization_output[edge_start + lane_id * val_per + seg] = local_sort_array[seg];
                    }
                }
            
               
            

                if(squad_size == 32)
                {
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }
                
            }
            else if(total_neighbor_count <= THREADS_PER_BLOCK)
            {
                type_int neighbor_power_two = next_power_of_two(total_neighbor_count);
                for(type_int i = edge_start + total_neighbor_count + lane_id; i < edge_start + neighbor_power_two; i += squad_size)
                {
                    if(sizeof(type_int) == 4)
                    {
                        device_factorization_output[i].row = INT_MAX;
                    }
                    else
                    {
                        device_factorization_output[i].row = LLONG_MAX;
                    }
                    
                }
                // wait for the copy to finish, sort on size that is power of two, garbage values will be naturally at the end
                if(squad_size == 32)
                {
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }
            
                if(neighbor_power_two <= THREADS_PER_BLOCK)
                {
                    // Obtain a segment of consecutive items that are blocked across threads
                    Output<type_int, type_data> local_sort_array[1];
                    if(lane_id < neighbor_power_two)
                    {
                        local_sort_array[0] = device_factorization_output[edge_start + lane_id];
                    }
                    else
                    {
                        if(sizeof(type_int) == 4)
                        {
                            local_sort_array[0].row = INT_MAX;
                        }
                        else
                        {
                            local_sort_array[0].row = LLONG_MAX;
                        }
                    }
                    
                    BlockMergeSort_One(sort_one_st).Sort(local_sort_array, compare_row_reference);
                    if(lane_id < neighbor_power_two)
                    {
                        device_factorization_output[edge_start + lane_id] = local_sort_array[0];
                    }
                   
                }

                if(squad_size == 32)
                {
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }
                
              
    
            }
            else
            {
                // pad in max ints to the next power of two in order to use bitonic sort
                
                type_int neighbor_power_two = next_power_of_two(total_neighbor_count);
                
                for(type_int i = edge_start + total_neighbor_count + lane_id; i < edge_start + neighbor_power_two; i += squad_size)
                {
                    if(sizeof(type_int) == 4)
                    {
                        device_factorization_output[i].row = INT_MAX;
                    }
                    else
                    {
                        device_factorization_output[i].row = LLONG_MAX;
                    }
                    
                }
                // wait for the copy to finish, sort on size that is power of two, garbage values will be naturally at the end
                if(squad_size == 32)
                {
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }
                bitonic_sort(device_factorization_output, edge_start, edge_start + neighbor_power_two, lane_id, compare_row);
               
                
            }
            unsigned long long sort_end = clock64();

         
            // check if row value of left entry is same as the current one
            for(type_int i = edge_start + lane_id; i < edge_start + total_neighbor_count; i += squad_size)
            {
                if(i != edge_start)
                {
                    // assuming default is 1, if equal, set to 0, meaning that it doesn't take up extra slot
                    if(device_factorization_output[i - 1].row == device_factorization_output[i].row)
                    {
                        device_factorization_output[i].multiplicity = 0;
                    }
                }
                // first element automatically sets to 1 by default
                // if(i == edge_start)
                // {
                //     printf("device_factorization_output[i].multiplicity: %d\n", device_factorization_output[i].multiplicity);
                // }
                
            }
            if(squad_size == 32)
            {
                __syncwarp();
            }
            else
            {
                __syncthreads();
            }
            
          
           

            // batched_binary_search<type_int, type_data>(spmat_device.row_indices + left_bound_idx, spmat_device.values + left_bound_idx, num_of_original_nnz, 
            //     device_factorization_output + edge_start, update_neighbor_count, spmat_device.merge + left_bound_idx, lane_id, squad_size);
            // move the existing edge into device_factorization output, ignore the merged ones
            // for(type_int i = edge_start + update_neighbor_count + lane_id; i < edge_start + total_neighbor_count; i += squad_size)
            // {
            //     if(!spmat_device.merge[left_bound_idx + i - edge_start - update_neighbor_count])
            //     {
            //         type_int old_slot = atomicAdd(&output_start_array[squad_id * 4 + 3], 1);
            //         device_factorization_output[old_slot + edge_start + update_neighbor_count] = Output<type_int, type_data>(spmat_device.row_indices[left_bound_idx + i - edge_start - update_neighbor_count],
            //             spmat_device.values[left_bound_idx + i - edge_start - update_neighbor_count], 1);
            //     }

                
            // }
            
            
            // 2. compute prefix sum to determine location, currently, this is ONE BASED INDEXING
            unsigned long long merge_start = clock64();
            type_int bound = edge_start + ((total_neighbor_count + squad_size - 1) / squad_size) * squad_size;
            type_int cumulative_index_block = 0;
            type_int total_index_add = 0;
            for(type_int i = edge_start + lane_id; i < bound; i += squad_size)
            {
                
                type_int output_val = 0;
                if(i < edge_start + total_neighbor_count)
                {
                    output_val = device_factorization_output[i].multiplicity;
                }
          
               
                BlockScanInt(temp_storage_scan_int[squad_id]).InclusiveSum(output_val,
                                                 output_val,
                                                 cumulative_index_block);
             
                if(i < edge_start + total_neighbor_count)
                {
                    device_factorization_output[i].multiplicity = output_val + total_index_add;
                }
                total_index_add += cumulative_index_block;
                if(squad_size == 32)
                {
                    //__threadfence();
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }
            }

           
        
            // if(true)
            // {
            //     if(lane_id == 0)
            //     {
            //         for(type_int i = edge_start; i < edge_start + total_neighbor_count; i += 1)
            //         {
            //             printf("job id: %d, edge %d, val: %f, multiplicity: %d, i: %d\n", job_id, device_factorization_output[i].row, device_factorization_output[i].value, device_factorization_output[i].multiplicity, i);
            //         }
            //     }
              
            //     __syncthreads();
            // }
            
            // read the last element's location to tell how many actual distinct elements there are
            type_int actual_neighbor_count = device_factorization_output[edge_start + total_neighbor_count - 1].multiplicity;
            // if(lane_id == 0 && total_neighbor_count > 100)
            // {
            //     printf("job_id: %d, total_neighbor_count: %d, update_neighbor_count: %d, actual_neighbor_count: %d\n", job_id, total_neighbor_count, update_neighbor_count, actual_neighbor_count);
            // }
            if(squad_size == 32)
            {
                __syncwarp();
            }
            else
            {
                __syncthreads();
            }
            // 3. merge entries with the same row value

            
            for(type_int i = edge_start + lane_id; i < edge_start + total_neighbor_count; i += squad_size)
            {
                // NEED TO SUBTRACT 1 TO ACCOUNT FOR ONE BASED INDEX
                type_int my_location = edge_start + device_factorization_output[i].multiplicity + total_neighbor_count - 1;
                atomicExch(&device_factorization_output[my_location].row, device_factorization_output[i].row);
                atomicAdd(&device_factorization_output[my_location].value, device_factorization_output[i].value);
                atomicAdd(&device_factorization_output[my_location].multiplicity, 1); // assume new location has a initial value of 0
                
            }

           
            
            // update the neighbor count after merging duplicates, need synchronization (wait for both add and device_factorization_output to finish updating)
            if(squad_size == 32)
            {
                //__threadfence();
                __syncwarp();
            }
            else
            {
                __syncthreads();
            }

           

            // 4. shift the merged elements to the beginning
            for(type_int i = edge_start + lane_id; i < edge_start + actual_neighbor_count; i += squad_size)
            {
                device_factorization_output[i] = device_factorization_output[i + total_neighbor_count];
                //printf("multiplicity here: %d\n", device_factorization_output[i].multiplicity);
                
            }
            if(squad_size == 32)
            {
                __syncwarp();
            }
            else
            {
                __syncthreads();
            }

            
        
          

        
            // if(true)
            // {
            //     if(lane_id == 0)
            //     {
            //         for(type_int i = edge_start; i < edge_start + actual_neighbor_count; i += 1)
            //         {
            //             printf("new job id: %d, edge %d, val: %f, multiplicity: %d, i: %d\n", job_id, device_factorization_output[i].row, device_factorization_output[i].value, device_factorization_output[i].multiplicity, i);
            //         }
            //     }
              
            //     __syncthreads();
            // }
          
            
            // update device_node_list with new locations after merging and update values after merging
            if(lane_id == 0)
            {
                device_node_list[job_id].count = actual_neighbor_count;
            }
            type_int old_neigh_count = total_neighbor_count;
            total_neighbor_count = actual_neighbor_count;
            unsigned long long merge_end = clock64();


            

          
            unsigned long long sort_val_start;
            unsigned long long sort_val_end;
            // 5. sort input based on value
         

            if(total_neighbor_count <= 8)
            {
                sort_val_start = clock64();
                odd_even_sort(device_factorization_output, edge_start, edge_start + total_neighbor_count, lane_id, compare_value);
                sort_val_end = clock64();
            }
            else if(total_neighbor_count <= 16)
            {
                // Obtain a segment of consecutive items that are blocked across threads
                

            
                Output<type_int, type_data> local_sort_array[1];

            
                int val_per = 1;    

                for(int seg = 0; seg < val_per; seg++)
                {
                    if(lane_id * val_per + seg < total_neighbor_count)
                    {
                        local_sort_array[seg] = device_factorization_output[edge_start + lane_id * val_per + seg];
                    }
                    else
                    {
                        local_sort_array[seg].value = INFINITY;
                    }
                }
                
            
                if(warp_id == 0 && lane_id < 16)
                {
                    MergeSort_Mini(sort_mini_st).Sort(local_sort_array, compare_value_reference);
                }
                    
                
                
                //odd_even_sort(device_factorization_output, edge_start, edge_start + total_neighbor_count, lane_id, compare_value);

                for(int seg = 0; seg < val_per; seg++)
                {
                    if(lane_id * val_per + seg < total_neighbor_count)
                    {
                        device_factorization_output[edge_start + lane_id * val_per + seg] = local_sort_array[seg];
                    }
                }
            
               

            
            
                
                

                if(squad_size == 32)
                {
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }
            
              
            }
            else if(total_neighbor_count <= THREADS_PER_BLOCK)
            {
                type_int neighbor_power_two = next_power_of_two(total_neighbor_count);
                for(type_int i = edge_start + total_neighbor_count + lane_id; i < edge_start + neighbor_power_two; i += squad_size)
                {
                    device_factorization_output[i].value = INFINITY;
                    
                }
                // wait for the copy to finish, sort on size that is power of two, garbage values will be naturally at the end
                if(squad_size == 32)
                {
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }
            
                if(neighbor_power_two <= THREADS_PER_BLOCK)
                {
                    // Obtain a segment of consecutive items that are blocked across threads
                    Output<type_int, type_data> local_sort_array[1];
                    if(lane_id < neighbor_power_two)
                    {
                        local_sort_array[0] = device_factorization_output[edge_start + lane_id];
                    }
                    else
                    {
                        local_sort_array[0].value = INFINITY;
                    }
                    
                    BlockMergeSort_One(sort_one_st).Sort(local_sort_array, compare_value_reference);
                    if(lane_id < neighbor_power_two)
                    {
                        device_factorization_output[edge_start + lane_id] = local_sort_array[0];
                    }
                    
                }

                if(squad_size == 32)
                {
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }

             
    
            }
            else
            {

                // pad in max ints to the next power of two in order to use bitonic sort
               
                
                
                type_int neighbor_power_two = next_power_of_two(total_neighbor_count);
                
                for(type_int i = edge_start + total_neighbor_count + lane_id; i < edge_start + neighbor_power_two; i += squad_size)
                {
                    
                    device_factorization_output[i].value = INFINITY;

                }
                // wait for the copy to finish, sort on size that is power of two, garbage values will be naturally at the end
                if(squad_size == 32)
                {
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }

                // if(job_id == 1)
                // {
                
                //     //printf("thread idx: %d\n", threadIdx.x);
                //     for(type_int i = edge_start + lane_id; i < edge_start + neighbor_power_two; i += squad_size)
                //     {
                        
                //         printf("beforeeeeeeeeeeeeeeeeee row: %d, value: %f, multiplicity here: %d, lane id: %d\n", device_factorization_output[i].row, device_factorization_output[i].value, device_factorization_output[i].multiplicity, lane_id);
                //     }
                //     __syncthreads();

                // }

                bitonic_sort(device_factorization_output, edge_start, edge_start + neighbor_power_two, lane_id, compare_value);


             
            }


            // if((job_id == 553 || job_id == 554 || job_id == 556) && lane_id == 0)
            // {
            //     for(type_int i = edge_start; i < edge_start + actual_neighbor_count; i += 1)
            //     {
            //         printf("job id: %d, edge %d, val: %f, i: %d\n", job_id, device_factorization_output[i].row, device_factorization_output[i].value, i);
            //     }

            // }
            __syncthreads();

           
        
            
            // OLD (no longer true): count for multiplicity and decide location using prefix sum (if we accumulate multiplicity while factorizing, then only neighboring edges can be the same after sorting)
            // compute prefix sum to figure out the cumulative value for sampling purposes
            // TODO, extend this to handle the case of multiple warps per squad
             
            type_data total_sum = 0.0;
            type_data block_aggregate = 0.0;
            
            // if(job_id == 7 && threadIdx.x == 0)
            // {
            //     printf("bound: %lld, i: %lld, edge_start: %lld\n", bound, bound - 1 - warp_lane_id, edge_start);
            // }
            // make sure warpscan is called by an entire wrap, that means we need to round up to multiple of 32
            // starting from end, calculate backward cumulative sum instead, forward direction seem to have stability issue when values are close to 0
            // for(int64_t i = bound - 1 - warp_lane_id; i >= edge_start && i >= 0; i -= THREADS_PER_WARP)
            // {
            //     //type_int reverse_index = i - (THREADS_PER_WARP - warp_lane_id) + warp_lane_id + 1;
               
            //     //if(lane_id == 0)
            //     //{
            //         //printf("warp_lane_id: %d, job id: %lld, bound: %lld, i: %lld, reverse: %lld, edge_start: %lld\n", warp_lane_id, job_id, bound, i, reverse_index, edge_start);
            //     //}

            //     type_data output_val = 0.0;
            //     if(i < edge_start + total_neighbor_count)
            //     {
            //         output_val = device_factorization_output[i].value;
            //     }
          
               
            //     WarpScan(temp_storage_scan[warp_id]).InclusiveSum(output_val,
            //                                      output_val,
            //                                      warp_aggregate);
             
            //     if(i < edge_start + total_neighbor_count)
            //     {
            //         device_factorization_output[i].cumulative_value = output_val + total_sum;
            //         // if(job_id == 15835)
            //         // {
            //         //     printf("cumulative_value: %.25lf\n", device_factorization_output[i].cumulative_value);
                        
            //         // }  
            //     }
            //     total_sum += warp_aggregate;
            //     if(squad_size == 32)
            //     {
            //         //__threadfence();
            //         __syncwarp();
            //     }
            // }

            // // forward sum
            // total_sum = 0.0;
            // warp_aggregate = 0.0;
            // recompute bound since list might be smaller after merge
            
            bound = edge_start + ((total_neighbor_count + squad_size - 1) / squad_size) * squad_size;
            for(type_int i = edge_start + lane_id; i < bound; i += squad_size)
            {
                
                type_data output_val = 0.0;
                if(i < edge_start + total_neighbor_count)
                {
                    output_val = device_factorization_output[i].value;
                    // if(output_val == 0.0)
                    // {
                    //     printf("0.0 at job id: %d\n", job_id);
                    // }
                }
          
               
                BlockScan(temp_storage_scan[squad_id]).InclusiveSum(output_val,
                                                 output_val,
                                                 block_aggregate);
             
                if(i < edge_start + total_neighbor_count)
                {
                    device_factorization_output[i].forward_cumulative_value = output_val + total_sum;
                }
                total_sum += block_aggregate;
                if(squad_size == 32)
                {
                    //__threadfence();
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }
            }

            
            
            // // End timing with clock64()
          
         

            /* Generate Samples and add them to map */
        
     
         
            
            //printf("job id: %lld, total neighbor: %lld, edge start: %lld, bound: %lld\n", job_id, total_neighbor_count, edge_start, bound);

            // Initialize the cuRAND state with seed, sequence number, and subsequence
            // generate a sample and return its index, calculate hash and add to map
            
            // since the next loop doesn't check the last index, explicitly set the value of the last index here
            // also store sum to node information
            if(lane_id == 0)
            {
               
                device_node_list[job_id].sum = total_sum;
                device_factorization_output[edge_start + total_neighbor_count - 1].value /= total_sum;
                
            }
            
            int count_iter = 0;
            for(type_int i = edge_start + lane_id; i < edge_start + total_neighbor_count - 1; i += squad_size)
            {
                
                //printf("thread id: %d, job id: %lld, total neighbor: %lld, count: %lld\n", threadIdx.x, job_id, total_neighbor_count, edge_start + total_neighbor_count - i - 1);
                type_int sample_index = 0;
                if(i < edge_start + total_neighbor_count - 2)
                {
                    // sample_index = sample_search(state, device_factorization_output + i + 1, edge_start + total_neighbor_count - i - 1, 
                    //     device_factorization_output[i + 1].cumulative_value, device_factorization_output[i].forward_cumulative_value);
                    sample_index = sample_search(state, device_factorization_output + i + 1, edge_start + total_neighbor_count - i - 1, 
                        total_sum, device_factorization_output[i].forward_cumulative_value);
                    sample_index = min(i + 1 + sample_index, edge_start + total_neighbor_count - 1) - i - 1;
                    // if(i + 1 + sample_index > edge_start + total_neighbor_count - 1)
                    // {
                    //     printf("uh oh\n");
                    // }
                }
                else
                {
                    
                    // sample_index = edge_start + total_neighbor_count - 1 - i - 1;
                    
                    // printf("i: %d, sample index: %d\n", i, sample_index);
                }
             

                   
                // if(job_id == 19052)
                // {
                //     printf("edge_start here: %lld, actual sample index: %lld, i + 1: %lld, cumu: %.25lf, forward cumu: %.25lf\n", edge_start, sample_index, i + 1, device_factorization_output[i + 1].cumulative_value, device_factorization_output[i].forward_cumulative_value);
                //     if(i + 1 == 67164)
                //     {
                //         printf("next forward value: %.25lf\n", device_factorization_output[i + 1].forward_cumulative_value);
                //     }
                    
                // }  
                sample_index += (i + 1); // account for offset
                type_int row_of_sample = max(device_factorization_output[i].row, device_factorization_output[sample_index].row);
                type_int col_of_sample = min(device_factorization_output[i].row, device_factorization_output[sample_index].row);

                // if(row_of_sample == col_of_sample && row_of_sample == 10)
                // {
                //     printf("i: %d, sample_index: %d, edge_start + total_neighbor_count - 1: %d, job id: %d\n", i, sample_index, edge_start + total_neighbor_count - 1, job_id);
                // }
                type_data value_of_sample = device_factorization_output[i].value * (total_sum - device_factorization_output[i].forward_cumulative_value) / total_sum;
                // if(job_id == 553 || job_id == 554 || job_id == 556)
                // {
                //     printf("row of sample: %d, col of sample: %d, sample val: %f, self value: %f\n", row_of_sample, col_of_sample, value_of_sample, device_factorization_output[i].value);
                // }
                

                
                // set value to value / total_sum
                device_factorization_output[i].value = device_factorization_output[i].value / total_sum;

               // type_data value_of_sample = device_factorization_output[i].value * device_factorization_output[i + 1].cumulative_value / total_sum;
                  
                
                
                // if value is too small, don't add for numerical stability
                // if(value_of_sample < 1e-10)
                // {
                //     //  printf("job id: %d, thread id: %d, so far: %f, row: %d, col: %d, i: %d, sample index: %d, device_factorization_output[i].row: %d, device_factorization_output[sample_index].row: %d, val: %f\n", job_id, threadIdx.x, 
                //     //      device_factorization_output[i + 1].forward_cumulative_value, row_of_sample, col_of_sample, i, sample_index, device_factorization_output[i].row, device_factorization_output[sample_index].row, value_of_sample);
                //     value_of_sample = 1e-10;
                    
                // }

                // update dependency, increment the column corresponding to max (i.e. row of sample)
            //    if(col_of_sample == 1)
            //    {
                   
            //         printf("total neighbor: %d, edge start: %d, bound: %d\n", total_neighbor_count, edge_start, bound);
            //         printf("job id: %d, thread id: %d, so far: %f, row: %d, col: %d, i: %d, sample index: %d, device_factorization_output[i].row: %d, device_factorization_output[sample_index].row: %d, val: %f\n", job_id, threadIdx.x, 
            //             device_factorization_output[i + 1].forward_cumulative_value, row_of_sample, col_of_sample, i, sample_index, device_factorization_output[i].row, device_factorization_output[sample_index].row, value_of_sample);
            //         printf("self value: %.32f, total sum: %.32f, so far value: %.32f\n", device_factorization_output[i].value, total_sum, device_factorization_output[i].forward_cumulative_value);
            //    }
                atomicAdd(&device_min_dependency_count[row_of_sample], 1);
                // if(col_of_sample == 30)
                // {
                    
                //     printf("dependency on 30: %d\n", atomicAdd(&device_min_dependency_count[30], 0));
                    
                //     //printf("current count: %d, add 1 to device node list at col_of_sample: %d from: %d\n", atomicAdd(&device_node_list[col_of_sample].count, 0), col_of_sample, job_id);
                // }
                
                // also add to dependency of col of sample for synchronization purposes
                //atomicAdd(&device_min_dependency_count[col_of_sample], 1);

                // increment count
                type_int old_count = atomicAdd(&device_node_list[col_of_sample].count, 1);

                type_int slot = (permute_static_hash(col_of_sample, map_size, num_cols, rand_vec_device) + old_count) % map_size;
                //type_int slot = (balanced_static_hash(col_of_sample, map_size, num_cols) + old_count) % map_size;
                //uint64_t slot = (murmurhash3_x64_64((char *)(&col_of_sample), sizeof(type_int), col_of_sample) + old_count) % map_size;
                //uint64_t slot = (djb2_hash((char *)(&col_of_sample), sizeof(type_int)) + old_count) % map_size;
               
                
                
                //slot = (slot + (djb2_hash((char *)(&row_of_sample), sizeof(type_int)) % 100)) % map_size;
                // slot = (slot + (djb2_hash((char *)(&row_of_sample), sizeof(type_int)) % 20)) % map_size;
                //slot = (slot + (row_of_sample % 50)) % map_size;
                bool finished_insert = false;

                
                
                while(!finished_insert)
                {
                    // check availability
                   
                
                    
                    int available_status = atomicExch(&device_edge_map[slot].availability, 1);
                    // cuda::atomic_ref<int, cuda::thread_scope_device> at_avail_acq(device_edge_map[slot].availability);
                    // int available_status = at_avail_acq.exchange(1, cuda::memory_order_acquire);
              
                    if(available_status == 0) // insert if free slot
                    {
                        // atomicExch(&device_edge_map[slot].row, row_of_sample);
                        // atomicExch(&device_edge_map[slot].col, col_of_sample);
                        // atomicAdd(&device_edge_map[slot].value, 0.0);
                        // atomicExch(&device_edge_map[slot].multiplicity, 1);
                        device_edge_map[slot].row = row_of_sample;
                        device_edge_map[slot].col = col_of_sample;
                        device_edge_map[slot].value = value_of_sample;
                        device_edge_map[slot].multiplicity = 1;
                        __threadfence();

                        // subtract the count here
                        //atomicAdd(&device_min_dependency_count[col_of_sample], -1);

                        
                        
                        // cuda::atomic_ref<int, cuda::thread_scope_device> at_avail_rel(device_edge_map[slot].availability);
                        // at_avail_rel.exchange(2, cuda::memory_order_release);
                        atomicExch(&device_edge_map[slot].availability, 2);

                        
                        
                        // cuda::atomic_ref<type_int, cuda::thread_scope_device> dependency_sync(device_min_dependency_count[col_of_sample]);
                        // dependency_sync.fetch_sub(1, cuda::memory_order_release);
                        
                       
                        finished_insert = true;

                        // if(threadIdx.x == 2){

                        //     printf("thread id: %d, slot: %lld, row sample: %lld, col sample: %lld\n", threadIdx.x, slot, row_of_sample, col_of_sample);
                        //     printf("availability: %d\n", atomicAdd(&device_edge_map[slot].availability, 0));
                        //     printf("address: %x\n", &device_edge_map[slot].availability);
                        //     return;
                        // }
                        
                         
                    }
                    else
                    {
                        
                        atomicExch(&device_edge_map[slot].availability, 2);
                        // cuda::atomic_ref<int, cuda::thread_scope_device> at_avail_rel(device_edge_map[slot].availability);
                        // at_avail_rel.exchange(2, cuda::memory_order_release);
                        slot = (slot + 1) % map_size;
                        

                        // Calculate elapsed clock cycles
                        count_iter++;
                        
                        
                    }
                

                    
                }
         
                
              
                    
                
            }
            
          
            

      

            // wait for everything to finish before potentially unblocking other nodes
            if(squad_size == 32)
            {
                //__threadfence();
                __syncwarp();
            }
            else
            {
                
                __syncthreads();
            }
           

            // update dependency by subtracting away from ones impacted by current node/column
       
          
            for(type_int i = edge_start + lane_id; i < edge_start + total_neighbor_count; i += squad_size)
            {
                
    
                
                
                type_int old_dependency = atomicAdd(&device_min_dependency_count[device_factorization_output[i].row], -(type_int)(device_factorization_output[i].multiplicity));
                // if(old_dependency < device_factorization_output[i].multiplicity)
                // {
                //     printf("job id: %d, row: %d, old_dependency: %d, multiplicity: %d\n", job_id + 1, device_factorization_output[i].row + 1, old_dependency, device_factorization_output[i].multiplicity);
                // }
                
                //cuda::atomic_ref<type_int, cuda::thread_scope_device> dependency_sync(device_min_dependency_count[device_factorization_output[i].row]);
                //type_int old_dependency = dependency_sync.fetch_sub(device_factorization_output[i].multiplicity, cuda::memory_order_release);
                if(old_dependency == device_factorization_output[i].multiplicity)
                {
                    queue_size = atomicAdd(&queue_device[num_cols], 1);
                    atomicExch(&queue_device[queue_size], device_factorization_output[i].row);
                    
                }

         
                
            }

            if(squad_size == 32)
            {
                //__threadfence();
                __syncwarp();
            }
            else
            {
                __syncthreads();
            }
           
            

            // copy local storage to global if necessary
            if(output_start_array[squad_id * 4 + 3])
            {
                type_int global_start = output_start_array[squad_id * 4];
                for(type_int i = edge_start + lane_id; i < edge_start + total_neighbor_count; i += squad_size)
                {
                    global_device_factorization_output[global_start + i] = local_output[i]; 
                }

                if(squad_size == 32)
                {
                    //__threadfence();
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }

                // reset local output's multiplicity, IMPORTANT
                for(type_int i = lane_id; i < LOCAL_CAPACITY; i += squad_size)
                {
                    local_output[i].multiplicity = 0; 
                    local_output[i].value = 0.0; 
                    local_output[i].row = 0; 
                }

                if(squad_size == 32)
                {
                    //__threadfence();
                    __syncwarp();
                }
                else
                {
                    __syncthreads();
                }
           
                
            }

            

          
            
            unsigned long long total_end = clock64();
            // Print the elapsed time for this thread (optional, careful with printf in kernels)
            // if (lane_id == 0 && total_end - total_start > 1000000) {  // Just printing from one thread to avoid clutter
            //     printf("lane_id: %d, total cycle: %llu, sort row time in clock cycles: %llu, sort val cycle: %llu, merge time in clock cycles: %llu, insert cycle: %llu, job id: %d, count iter: %d, total neighbor: %d, old neighbor count: %d\n", lane_id, total_end - total_start, sort_end - sort_start, sort_val_end - sort_val_start, merge_end - merge_start, insert_end - insert_start, job_id, count_iter, total_neighbor_count, old_neigh_count);
            //     //printf("job id: %d, count iter: %d\n", job_id, count_iter);
            // } 
            

        }

        //__threadfence();
     


        // get new job
        //printf("job id: %lld, atomic_dependency_count: %lld, update_neighbor_count: %lld\n", job_id, atomic_dependency_count, update_neighbor_count);
        //job_id += gap;
        job_id = -1;
        // if(job_id == 15795 && job_id % 2 == 1)
        // {
        //     printf("is job id %lld being runnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn\n", job_id);
        // }
        //printf("job id: %lld, atomic_dependency_count: %lld, update_neighbor_count: %lld\n", job_id, atomic_dependency_count, update_neighbor_count);
    }

   
    if(lane_id == 0)
    {
        //printf("run count: %d\n", run_count);
    }
    if(lane_id == 0 && blockIdx.x == 0)
    {
        printf("count num: %d\n", device_node_list[0].count);
    }
    
    // if(threadIdx.x == 0 && blockIdx.x == 1)
    // {
    //     printf("squad id: %d\n", squad_id);
    //     printf("lane id: %d\n", lane_id);
    //     printf("count[0]: %ld\n", device_min_dependency_count[0]);
    //     printf("count[1]: %ld\n", device_min_dependency_count[1]);
    // }


    
}

bool compare(const Output<int, double> &a, const Output<int, double> &b)
{
    return a.row < b.row;  
};

template <typename type_int, typename type_data>
__global__ void construct_rowptr(Node<type_int, type_data> *node_list_host, type_int *csr_row_ptr, type_int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int gap = gridDim.x * block_size;

    for(type_int index = bid * block_size + tid; index < n - 1; index += gap)
    {
        csr_row_ptr[index + 1] = node_list_host[index].count + 1;
    }


    if(tid == 0 && bid == 0)
    {
        csr_row_ptr[0] = 0;
    }
    

}


template <typename type_int, typename type_data>
__global__ void remove_last_csr(Output<type_int, type_data> *device_factorization_output, Node<type_int, type_data> *node_list_host, 
    type_int *csr_row_ptr, type_int n)
{
    
    int bid = blockIdx.x;
    int gap = gridDim.x;
    int block_size = blockDim.x;
    int lane_id = threadIdx.x % block_size;

    // determine row start
    for(type_int index = bid; index < n; index += gap)
    {
        type_int size_local = node_list_host[index].count;
        type_int start_local = node_list_host[index].start;
        // determine start location within each row
        for(type_int inner = lane_id; inner < size_local; inner += block_size)
        {
            // can only happen to 1 thread
            if (device_factorization_output[start_local + inner].row == n)
            {
                csr_row_ptr[index + 1]--;
            }

        }
        
    }

 

}

template <typename type_int, typename type_data>
__global__ void shift_element(Output<type_int, type_data> *device_factorization_output, Node<type_int, type_data> *node_list_host, type_int n)
{
    int bid = blockIdx.x;
    int gap = gridDim.x;
    int block_size = blockDim.x;
    int lane_id = threadIdx.x % block_size;

    __shared__ type_int shift_location;
    __shared__ bool changed;

    if(lane_id == 0)
    {
        shift_location = -1;
        changed = false;
    }

    // Ensure all threads see the updated shared variable
    __syncthreads();


    /* 1. if remove, remove the last row, and make sure to shift it */
   

    for(type_int index = bid; index < n; index += gap)
    {
        type_int size_local = node_list_host[index].count;
        type_int start_local = node_list_host[index].start;
        // determine start location within each row

        for(type_int inner = lane_id; inner < size_local; inner += block_size)
        {
 
            if(device_factorization_output[start_local + inner].row == n)
            {
                // only 1 will assign this
                shift_location = inner;
                changed = true;
            }
        }

        __syncthreads();



        // shift based on shift location
        if(changed)
        {
            // round to the nearest block size, so the all threads will enter the loop and call syncthreads to prevent hang
            type_int rounded_to_blocksize = ((size_local + block_size - 1) / block_size) * block_size;
            rounded_to_blocksize += (shift_location + 1);

            for(type_int inner = lane_id + shift_location + 1; inner < rounded_to_blocksize; inner += block_size)
            {
            
                // store this, sync to make sure it doesn't get overwritten before being read
                Output<type_int, type_data> temp;
                if(inner < size_local)
                {
                    temp = device_factorization_output[start_local + inner];
                }
                

                __syncthreads();

                // now shift
                if(inner < size_local)
                {
                    // if(index == 0)
                    // {
                    //     printf("start_local + inner - 1: %d, temp row: %d\n", start_local + inner - 1, temp.row);
                    // }
                    device_factorization_output[start_local + inner - 1] = temp;
                }

            }

            // make sure to reset this
            if(lane_id == 0)
            {
                changed = false;
            }
            __syncthreads();

        }
        
    }


}

template <typename type_int, typename type_data>
__global__ void construct_csr(Output<type_int, type_data> *device_factorization_output, Node<type_int, type_data> *node_list_host, 
    type_int *csr_row_ptr, type_int *csr_col_ind, type_data *csr_val, type_int n)
{
    
    int bid = blockIdx.x;
    int gap = gridDim.x;
    int block_size = blockDim.x;
    int lane_id = threadIdx.x % block_size;


    


    /* 2. copy the result into indices and val */
    // determine row start
    for(type_int index = bid; index < n; index += gap)
    {
        type_int size_local = csr_row_ptr[index + 1] - csr_row_ptr[index] - 1;
        type_int start_local = node_list_host[index].start;
        // determine start location within each row
        for(type_int inner = lane_id; inner < size_local; inner += block_size)
        {
            // if(index == 0)
            // {
            //     printf("row: %d, ptr index: %d, size local: %d\n", device_factorization_output[start_local + inner].row, csr_row_ptr[index] + inner, size_local);
            //     // printf("col here: %d\n", csr_col_ind[1]);
            // }

            csr_col_ind[csr_row_ptr[index] + inner] = device_factorization_output[start_local + inner].row;
            csr_val[csr_row_ptr[index] + inner] = -device_factorization_output[start_local + inner].value;
            
            
        }
        // fill in the diagonal element, explicitly 1
        if(lane_id == 0)
        {
            csr_col_ind[csr_row_ptr[index + 1] - 1] = index;
            csr_val[csr_row_ptr[index + 1] - 1] = 1.0;
        }
        
    }
    // if(csr_col_ind[2] == 0)
    // {
    //     printf("col end: %d, \n", csr_col_ind[1]);
    // }

}

template <typename type_int, typename type_data>
__global__ void copy_diagonal(Node<type_int, type_data> *node_list_host, type_data *diagonal, type_int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_size = blockDim.x;
    int gap = gridDim.x * block_size;

    for(type_int index = bid * block_size + tid; index < n; index += gap)
    {
        diagonal[index] = node_list_host[index].sum;
    }

    
}



template <typename type_int, typename type_data>
void segment_sort_indices(type_int *csr_col_ind, type_int *csr_rowptr, type_data *csr_val, type_int n, type_int total_nnz)
{
    
   // Declare, allocate, and initialize device-accessible
    // pointers for sorting data
    type_int num_items = total_nnz;  // e.g., 7
    type_int num_segments = n;       // e.g., 3


    // Determine temporary device storage requirements
    void    *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes, csr_col_ind, csr_col_ind, csr_val, csr_val,
        num_items, num_segments, csr_rowptr, csr_rowptr + 1);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes, csr_col_ind, csr_col_ind, csr_val, csr_val,
        num_items, num_segments, csr_rowptr, csr_rowptr + 1);
    
    cudaFree(d_temp_storage);
}


template <typename type_int, typename type_data>
void factorization_driver(sparse_matrix_processor<type_int, type_data> &processor, int num_blocks, bool check_solve, double tolerance)
{
    // this is a warmp up run
    int *dummy_pt;
    cudaMalloc((void**)&dummy_pt, 4);
    cudaFree(dummy_pt);

    int edge_pool_size = processor.mat.nonZeros() * 4;
    printf("Edge size: %ld\n", sizeof(Edge<type_int, type_data>{}));

    // Allocate memory on the host (CPU)
    Edge<type_int, type_data> *host_edge_map = (Edge<type_int, type_data> *)malloc(edge_pool_size * sizeof(Edge<type_int, type_data>{}));
    for(size_t i = 0; i < edge_pool_size; i++)
    {
        
        host_edge_map[i] = Edge<type_int, type_data>(0, 0, 0.0);
    }
    
    // Allocate device memory (GPU)
    Edge<type_int, type_data> *device_edge_map;
    cudaMalloc((void**)&device_edge_map, edge_pool_size * sizeof(Edge<type_int, type_data>{}));

    // Copy data from the host (CPU) to the device (GPU)
    cudaMemcpy(device_edge_map, host_edge_map, edge_pool_size * sizeof(Edge<type_int, type_data>{}), cudaMemcpyHostToDevice);

    // Allocate storage for output, also allocate space for position index
    Output<type_int, type_data> *device_factorization_output;
    cudaMalloc((void**)&device_factorization_output, edge_pool_size * sizeof(Output<type_int, type_data>{}));
    type_int host_output_position_idx = 0;
    type_int *device_output_position_idx;
    cudaMalloc((void**)&device_output_position_idx, sizeof(type_int));
    cudaMemcpy(device_output_position_idx, &host_output_position_idx, sizeof(type_int), cudaMemcpyHostToDevice);


    
    // copy sparse matrix from cpu to gpu
    processor.insert_lower_triangular_info(processor.mat);
    sparse_matrix<type_int, type_data> &spmat = processor.mat;
    sparse_matrix_device<type_int, type_data> spmat_device;
    spmat_device.num_rows = spmat.num_rows;
    spmat_device.num_cols = spmat.num_cols;

    cudaMalloc((void**)&spmat_device.values, spmat.values.size() * sizeof(type_data));
    cudaMemcpy(spmat_device.values, spmat.values.data(), spmat.values.size() * sizeof(type_data), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&spmat_device.row_indices, spmat.row_indices.size() * sizeof(type_int));
    cudaMemcpy(spmat_device.row_indices, spmat.row_indices.data(), spmat.row_indices.size() * sizeof(type_int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&spmat_device.col_ptrs, spmat.col_ptrs.size() * sizeof(type_int));
    cudaMemcpy(spmat_device.col_ptrs, spmat.col_ptrs.data(), spmat.col_ptrs.size() * sizeof(type_int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&spmat_device.merge, spmat.row_indices.size() * sizeof(bool));
    cudaMemset(spmat_device.merge, 0, spmat.row_indices.size());

    cudaMalloc((void**)&spmat_device.lower_tri_cptr_start, spmat.lower_tri_cptr_start.size() * sizeof(type_int));
    cudaMemcpy(spmat_device.lower_tri_cptr_start, spmat.lower_tri_cptr_start.data(), spmat.lower_tri_cptr_start.size() * sizeof(type_int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&spmat_device.lower_tri_cptr_end, spmat.lower_tri_cptr_end.size() * sizeof(type_int));
    cudaMemcpy(spmat_device.lower_tri_cptr_end, spmat.lower_tri_cptr_end.data(), spmat.lower_tri_cptr_end.size() * sizeof(type_int), cudaMemcpyHostToDevice);

    // copy min_dependency_array
    if(spmat.num_cols != processor.min_dependency_count.size())
    {
        printf("processor.mat.num_cols: %d, spmat.num_cols: %d, min_dependency_count.size(): %ld\n", processor.mat.num_cols, spmat.num_cols, processor.min_dependency_count.size());
        assert(spmat.num_cols == processor.min_dependency_count.size());
    }
    
    type_int *device_min_dependency_count;
    //printf("dependency at 44481: %d\n", processor.min_dependency_count[44481]);
    cudaMalloc((void**)&device_min_dependency_count, processor.min_dependency_count.size() * sizeof(type_int));
    cudaMemcpy(device_min_dependency_count, processor.min_dependency_count.data(), 
        processor.min_dependency_count.size() * sizeof(type_int), cudaMemcpyHostToDevice);
    std::vector<type_int> queue_cpu(spmat.num_cols + 2, 0); // last two element represent counters
    for(type_int i = 0; i < processor.min_dependency_count.size(); i++)
    {
        if(processor.min_dependency_count[i] == 0)
        {
            queue_cpu[queue_cpu[queue_cpu.size() - 2]] = i;
            queue_cpu[queue_cpu.size() - 2]++;
            //printf("scheduled: %d\n", i);
        }
    }
    printf("initial queue size: %d\n", queue_cpu[queue_cpu.size() - 2]);
    assert(queue_cpu[0] == 0);
    type_int *queue_device;
    cudaMalloc((void**)&queue_device, (spmat.num_cols + 2) * sizeof(type_int));
    cudaMemcpy(queue_device, queue_cpu.data(), (spmat.num_cols + 2) * sizeof(type_int), cudaMemcpyHostToDevice);

    
    
    // create node array
    std::vector<Node<type_int, type_data>> node_list_host(spmat.num_cols);
    for(size_t i = 0; i < spmat.num_cols; i++)
    {
        node_list_host[i] = Node<type_int, type_data>(0, 0);
    }
    Node<type_int, type_data> *device_node_list;
    cudaMalloc((void**)&device_node_list, spmat.num_cols * sizeof(Node<type_int, type_data>{}));
    cudaMemcpy(device_node_list, node_list_host.data(), spmat.num_cols * sizeof(Node<type_int, type_data>{}), cudaMemcpyHostToDevice);


    // random permutation
    // Shuffle the array using std::shuffle.
    // Optionally, print the array (for verification).
    type_int rand_vec_size = spmat.num_cols;
    std::vector<type_int> rand_vec(rand_vec_size);
    for (type_int i = 0; i < rand_vec_size; i++) {
       rand_vec[i] = i;
    }
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(rand_vec.data(), rand_vec.data() + rand_vec_size, g);

    // Allocate raw device memory.
    type_int* rand_vec_device = nullptr;
    cudaError_t err = cudaMalloc((void**)&rand_vec_device, rand_vec_size * sizeof(type_int));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
        exit(0);
    }

    // Copy the shuffled permutation from the host to the device.
    err = cudaMemcpy(rand_vec_device, rand_vec.data(), rand_vec_size * sizeof(type_int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
        exit(0);
    }


    
    printf("CURRENTLY, MULTIPLICITY STORAGE USES INT32, DON'T NEED 64 BIT UNLESS IN EXTREME CIRCUMSTANCES \n");
    printf("CURRENTLY, BINARY SEARCH USES INT32, DON'T NEED 64 BIT UNLESS IN EXTREME CIRCUMSTANCES \n");
    printf("defaulting row and col to 0 in map may cause problem for column 0 \n");

    // launch kernel
    if(HOST_WARPS_PER_SQUAD * HOST_THREADS_PER_WARP > HOST_THREADS_PER_BLOCK)
    {
        printf("FOR NOW, DOESN'T SUPPORT CASE WHERE THREADS PER SQUAD IS GREATER THAN THREADS PER BLOCK, \
            WOULD NEED TO CHANGE ATOMIC SCOPE (I.E. THREAD_SCOPE_BLOCK)\n");
        assert(false);
        exit(0);
    }
    if(HOST_WARPS_PER_SQUAD > 32)
    {
        printf("FOR NOW, DOESN'T SUPPORT MORE THAN 32 THREADS PER SQUAD\n");
        assert(false);
        exit(0);
    }
    
    if(HOST_WARPS_PER_SQUAD > 32 && (HOST_THREADS_PER_BLOCK / (HOST_WARPS_PER_SQUAD * HOST_THREADS_PER_WARP) > 16))
    {
        printf("HAVE MORE THAN 16 SQUADS, CAN BE PROBLEMATIC\n");
        assert(false);
        exit(0);
    }
    printf("check point 1\n");
    dummy<<<1, 1>>>();
    printf("check point 2\n");
    cudaDeviceSynchronize();
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Record start event
    cudaEventRecord(startEvent, 0);
    perform_factorization_device<type_int, type_data><<<num_blocks, HOST_THREADS_PER_BLOCK>>>(spmat_device, edge_pool_size, device_edge_map, edge_pool_size, 
        device_factorization_output, device_min_dependency_count, device_node_list, device_output_position_idx, queue_device, rand_vec_device);
         
    // Record stop event
    cudaEventRecord(stopEvent, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stopEvent);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);

    // Print the time taken
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;



    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Kernel launch successful!" << std::endl;
    }

    if(check_solve)
    {
        
        auto start = std::chrono::high_resolution_clock::now();
        bool removal = false;
        // 1. initialize csr ptr and trim if necessary
        type_int num_cols = spmat.num_cols;
        if(removal)
        {
            num_cols--;
        }
        type_int *csr_rowptr_device;
        cudaMalloc((void**)&csr_rowptr_device, (num_cols + 1) * sizeof(type_int));


        construct_rowptr<type_int, type_data><<<256, 128>>>(device_node_list, csr_rowptr_device, num_cols + 1);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "construct rowptr launch failed: " << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "construct rowptr launch successful!" << std::endl;
        }
        


        // 2. count total elements without trim last row (note that at this point, last col is implicitly trimmed, so won't be accounted for here)
        // count last col explicitly add back the count
        type_int *total_nnz_with_last_row_device;
        cudaMalloc(&total_nnz_with_last_row_device, sizeof(type_int));
        void *d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, csr_rowptr_device + 1, total_nnz_with_last_row_device, num_cols);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run sum-reduction
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, csr_rowptr_device + 1, total_nnz_with_last_row_device, num_cols);
        cudaFree(d_temp_storage);
        
        type_int total_nnz_with_last_row;
        cudaMemcpy(&total_nnz_with_last_row, total_nnz_with_last_row_device, sizeof(type_int), cudaMemcpyDeviceToHost);
        type_int count_last_col;
        if(removal)
        {
            cudaMemcpy(&count_last_col, &device_node_list[num_cols].count, sizeof(type_int), cudaMemcpyDeviceToHost);
            count_last_col++;
        }
        else
        {
            count_last_col = 0;
        }
        

       


        // 3. remove last row, then shift element, then calculate rowptr and calculate cumulative sum
        if(removal)
        {
            // must be in this order, if shift first, then csr won't see the last element and therefore won't update
            remove_last_csr<type_int, type_data><<<2048, 32>>>(device_factorization_output, device_node_list, 
                csr_rowptr_device, num_cols);
            cudaDeviceSynchronize();
            
            shift_element<type_int, type_data><<<2048, 32>>>(device_factorization_output, device_node_list, num_cols);
            cudaDeviceSynchronize();
        }
      
        // Determine temporary device storage requirements for inclusive
        // prefix sum
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, csr_rowptr_device, csr_rowptr_device, num_cols + 1);

        // Allocate temporary storage for inclusive prefix sum
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        // Run inclusive prefix sum
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, csr_rowptr_device, csr_rowptr_device, num_cols + 1);
        cudaFree(d_temp_storage);


        // 4. set up col indices and values
        type_int total_nnz;
        cudaMemcpy(&total_nnz, csr_rowptr_device + num_cols, sizeof(type_int), cudaMemcpyDeviceToHost);
        printf("total nnz: %d\n", total_nnz);
        if(!removal)
        {
            spmat_device.nnz = processor.mat.nonZeros();
        }

        type_int *csr_col_ind_device;
        cudaMalloc((void**)&csr_col_ind_device, (total_nnz) * sizeof(type_int));
        type_data *csr_val_device;
        cudaMalloc((void**)&csr_val_device, (total_nnz) * sizeof(type_data));
        
        construct_csr<type_int, type_data><<<2048, 32>>>(device_factorization_output, device_node_list, 
            csr_rowptr_device, csr_col_ind_device, csr_val_device, num_cols);
              
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "construct_csr launch failed: " << cudaGetErrorString(err) << std::endl;
        } else {
            std::cout << "construct_csr launch successful!" << std::endl;
        }

     
        //printf("entry at %d: %d\n", num_cols, check[num_cols]);

        // 5. sort the column indices and values (pair sort)
        
        segment_sort_indices<type_int, type_data>(csr_col_ind_device, csr_rowptr_device, csr_val_device, num_cols, total_nnz);

        // std::vector<type_int> check(total_nnz, 1);
        // std::vector<type_data> check_val(total_nnz, 1);
        // cudaMemcpy(check.data(), csr_col_ind_device, (total_nnz) * sizeof(type_int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(check_val.data(), csr_val_device, (total_nnz) * sizeof(type_data), cudaMemcpyDeviceToHost);
        
        // //writeVectorToFile(check, "triangular_fac");
        // for(int i = 0; i < 20; i++)
        // {
        //     printf("entry at %d: %d, val: %f\n", i, check[i], check_val[i]);
        // }

        // 6. create diagonal entries
        type_data *diagonal_device;
        cudaMalloc((void**)&diagonal_device, (num_cols) * sizeof(type_data));
        copy_diagonal<type_int, type_data><<<256, 128>>>(device_node_list, diagonal_device, num_cols);
        cudaDeviceSynchronize();


     
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
      
        


        // TODO: REMEMBER TO FREE STUFF AT THE END
        cudaFree(device_node_list);
        cudaFree(device_factorization_output);
        cudaFree(device_edge_map);
        
       
        printf("pre trimming, num cols: %ld, laplacian nnz: %ld, factorization nnz: %ld\n", processor.mat.num_cols, processor.mat.nonZeros(), total_nnz_with_last_row + count_last_col);
      

        
        
        std::cout << "total conversion time: " << duration.count() << " milliseconds" << std::endl;

        prepare_and_solve(spmat_device, csr_rowptr_device, csr_val_device, csr_col_ind_device, diagonal_device, tolerance, 1, total_nnz, removal);

   
        
    }
    else
    {
        // TODO: REMEMBER TO FREE STUFF AT THE END
        cudaFree(device_node_list);
        cudaFree(device_factorization_output);
        cudaFree(device_edge_map);

    }
   
    
    printf("\n\n\n\n\n\n\n\n\n\n\n");
    
    // if(check_solve)
    // {
    //     // copy the answer back to cpu
    //     cudaMemcpy(node_list_host.data(), device_node_list, spmat.num_cols * sizeof(Node<type_int, type_data>{}), cudaMemcpyDeviceToHost);
    //     std::vector<Output<type_int, type_data>> device_factorization_output_host(edge_pool_size);
    //     cudaMemcpy(device_factorization_output_host.data(), device_factorization_output, edge_pool_size * sizeof(Output<type_int, type_data>{}), cudaMemcpyDeviceToHost);
        

    //     // TODO: REMEMBER TO FREE STUFF AT THE END
    //     cudaFree(device_node_list);
    //     cudaFree(device_factorization_output);
    //     cudaFree(device_edge_map);
        
    //     // make sure to add 1 to the size of each column since factorization did not include diagonal
    //     std::vector<type_int> csr_rowptr_host(spmat.num_cols + 1);
    //     std::vector<type_data> diagonal_entries(spmat.num_cols);
    //     csr_rowptr_host[0] = 0;
    //     size_t total_needed_size = 0;
    //     for(size_t i = 0; i < node_list_host.size(); i++)
    //     {
            
    //         total_needed_size = total_needed_size + node_list_host[i].count + 1;
    //         csr_rowptr_host[i + 1] = total_needed_size;

    //         // sort list and compute sum, also negate entries since the factorization was done with positive entries
    //         type_int col_start = node_list_host[i].start;
    //         //printf("col_start: %d, col_size: %d\n", col_start, node_list_host[i].count + 1);
    //         // if(i > 0)
    //         // {
    //         //     if(col_start < node_list_host[i - 1].start + node_list_host[i - 1].count)
    //         //     {
    //         //         printf("i - 1: %d, prev col start: %d, prev count: %d, cur start: %d\n", i - 1, node_list_host[i - 1].start, node_list_host[i - 1].count, col_start);
    //         //         assert(col_start > node_list_host[i - 1].start + node_list_host[i - 1].count);
    //         //     }
                
    //         // }
       
    //         // compute column sum and append diagonal, negate entries
    //         for(size_t j = col_start; j < col_start + node_list_host[i].count; j++)
    //         {
    //             device_factorization_output_host[j].value = -device_factorization_output_host[j].value;
    //         }
    //         diagonal_entries[i] = node_list_host[i].sum;

    //         // UPDATE COUNT
    //         device_factorization_output_host[col_start + node_list_host[i].count] = Output<type_int, type_data>(i, 1.0, 1);
    //         // if(i == 40089)
    //         // {
    //         //     printf("insert for %ld at: %d\n", i, col_start);
    //         //     // printf("output row: %d\n", device_factorization_output_host[col_start + node_list_host[i].count].row);
    //         //     for(int j = col_start; j < col_start + node_list_host[i].count + 1; j++)
    //         //     {
    //         //         printf("output row: %d, val: %f\n", device_factorization_output_host[j].row, device_factorization_output_host[j].value);
    //         //     }
    //         // }
    //         node_list_host[i].count++;
    //         std::sort(device_factorization_output_host.data() + col_start, device_factorization_output_host.data() + col_start + node_list_host[i].count, compare);
    //         // if(true)
    //         // {
    //         //     printf("insert for 0 at: %d\n", node_list_host[i].count);
    //         //     // printf("output row: %d\n", device_factorization_output_host[col_start + node_list_host[i].count].row);
    //         //     for(int j = col_start; j < col_start + node_list_host[i].count; j++)
    //         //     {
    //         //         printf("after output row: %d\n", device_factorization_output_host[j].row);
    //         //     }
    //         // }
    //     }
        
    //     std::vector<type_data> csr_val_host(total_needed_size);
    //     std::vector<type_int> csr_col_ind_host(total_needed_size);

    //     // start writing the result into a csr, preparing for cusparse operations
    //     for(size_t i = 0; i < node_list_host.size(); i++)
    //     {
          
    //         for(size_t j = csr_rowptr_host[i]; j < csr_rowptr_host[i + 1]; j++)
    //         {
    //             csr_col_ind_host[j] = device_factorization_output_host[node_list_host[i].start + j - csr_rowptr_host[i]].row;
    //             csr_val_host[j] = device_factorization_output_host[node_list_host[i].start + j - csr_rowptr_host[i]].value;

    //             // assert that diagonal elements are 1
    //             if(device_factorization_output_host[node_list_host[i].start + j - csr_rowptr_host[i]].row == i)
    //             {
    //                 assert(device_factorization_output_host[node_list_host[i].start + j - csr_rowptr_host[i]].value == 1.0);
    //             }
    //             assert(device_factorization_output_host[node_list_host[i].start + j - csr_rowptr_host[i]].row >= i);
    //         }
    //     }
    //     // int max_row = 0;
    //     // double max_val = 0.0;
    //     // double min_val = 1.0;
    //     // for(int i = 0; i < csr_rowptr_host.size() - 1; i++)
    //     // //for(int i = 0; i < 10; i++)
    //     // {
    //     //     for(size_t j = csr_rowptr_host[i]; j < csr_rowptr_host[i + 1]; j++)
    //     //     {
    //     //         //if(std::isnan(csr_val_host[j]))
    //     //         if(true)
    //     //         {
    //     //             printf("col: %d, row: %d, factor value: %f\n", i, csr_col_ind_host[j], csr_val_host[j]);
                    
    //     //         }
    //     //         max_row = max(max_row, csr_col_ind_host[j]);
    //     //         max_val = max(max_val, abs(csr_val_host[j]));
    //     //         min_val = min(min_val, abs(csr_val_host[j]));
    //     //     }
           
            
    //     // }
    //     // printf("max row: %d\n", max_row);
    //     // printf("max val: %f\n", max_val);
    //     // printf("min val: %f\n", min_val);
        
    //     // double max_val = 0.0;
    //     // for(int i = 0; i < spmat.num_cols; i++)
    //     // //for(int i = 0; i < 10; i++)
    //     // {
    //     //     if(true)
    //     //     {
    //     //         printf("diagonal entries at %d: %f\n", i, diagonal_entries[i]);
                
    //     //     }
    //     //     max_val = max(max_val, diagonal_entries[i]);
            
    //     // }
    //     // printf("max val: %f\n", max_val);
      
    //     // for(int i = 0; i < processor.mat.col_ptrs.size() - 1; i++)
    //     // {
    //     //     for(size_t j = processor.mat.col_ptrs[i]; j < processor.mat.col_ptrs[i + 1]; j++)
    //     //     {
    //     //         if(i < 50)
    //     //         {
    //     //             printf("col: %d, row: %d, factor value: %f\n", i, processor.mat.row_indices[j], processor.mat.values[j]);
                 
    //     //         }
                
    //     //     }
    //     // }

    //     /* 
    //     // write solution to file
    //     std::string wpath = "c_sol.mtx";
    //     std::ofstream output_stream(wpath);
    //     if (!output_stream.is_open()) {
    //         std::cerr << "Failed to open file for writing." << std::endl;
    //         exit(1);
    //     }
    //     // write_csr_to_matrix_market(csr_rowptr_host, csr_col_ind_host, csr_val_host, spmat.num_cols, spmat.num_cols, "c_sol.mtx");
    //     fast_matrix_market::matrix_market_header header(diagonal_entries.size(), diagonal_entries.size());
    //     header.object = fast_matrix_market::matrix;
    //     header.symmetry = fast_matrix_market::general;
    //     fast_matrix_market::write_options opts;
    //     opts.precision = 16;
    //     fast_matrix_market::write_matrix_market_csc(output_stream,
    //                              header, 
    //                              csr_rowptr_host,
    //                              csr_col_ind_host,
    //                              csr_val_host,
    //                              false,
    //                              opts);
    //     output_stream.flush();  // Ensure any buffered output is written to the file
    //     output_stream.close();  // Close the file stream when done
    //     */

     
        
    //     prepare_and_solve(processor.mat, csr_rowptr_host, csr_val_host, csr_col_ind_host, diagonal_entries, tolerance, 0);

    //     // read in preconditioner for verification
    //     // Load the matrix from file
    //     // std::vector<type_data> diag_vec;

    //     // // Open the file
    //     // std::ifstream file("diag");
        
    //     // // Check if the file was successfully opened
    //     // if (!file.is_open()) {
    //     //     std::cerr << "Failed to open the file!" << std::endl;
    //     //     exit(1);  // Exit with error code
    //     // }

    //     // // Read each double from the file
    //     // type_data number;
    //     // while (file >> number) {
    //     //     diag_vec.push_back(number);  // Store each number into the vector
    //     // }

    //     // // Close the file
    //     // file.close();
    //     // std::string path = "verify";
    //     // std::ifstream input_stream(path);
    //     // triplet_matrix<type_int, type_data> input_triplet;
    //     // input_triplet.nrows = 28924;
    //     // input_triplet.ncols = 28924;
    //     // // read input into triplet
    //     // fast_matrix_market::read_matrix_market_triplet(
    //     //     input_stream, input_triplet.nrows, input_triplet.ncols, input_triplet.rows, input_triplet.cols, input_triplet.vals);

    //     // sparse_matrix<type_int, type_data> cscMatrix = processor.triplet_to_csc_with_diagonal(input_triplet.nrows, input_triplet.ncols,
    //     //                     input_triplet.rows,
    //     //                     input_triplet.cols,
    //     //                     input_triplet.vals);
        
       

    
    //     //prepare_and_solve(processor.mat, cscMatrix.col_ptrs, cscMatrix.values, cscMatrix.row_indices, diag_vec);
        
    // }
    // else
    // {
    //     // TODO: REMEMBER TO FREE STUFF AT THE END
    //     cudaFree(device_node_list);
    //     cudaFree(device_factorization_output);
    //     cudaFree(device_edge_map);
    // }
    
    // printf("\n\n\n\n\n\n\n\n\n\n\n");

    









  
    
  
}



int main(int argc, char* argv[]) {

   
    
  
    //compute_parmetis_ordering(argv[1]);
    printf("problem: %s\n", argv[1]);
    sparse_matrix_processor<custom_idx, double> processor(argv[1]);
    
    factorization_driver<custom_idx, double>(processor, std::stoi(argv[2]), std::stoi(argv[3]), std::atof(argv[4]));


    return 0;
    
   

     


    
}
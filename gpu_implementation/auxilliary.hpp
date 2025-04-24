
#include <cuda_runtime.h>
#include <cuda/atomic>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include "pre_process.hpp"

#define LOCAL_CAPACITY 64

__device__ const int THREADS_PER_WARP = 32;
__device__ const int WARPS_PER_SQUAD = 1;



const int HOST_WARPS_PER_SQUAD = 1;
const int HOST_THREADS_PER_WARP = 32;

__device__ const int THREADS_PER_BLOCK = THREADS_PER_WARP * WARPS_PER_SQUAD;
const int HOST_THREADS_PER_BLOCK = HOST_THREADS_PER_WARP * HOST_WARPS_PER_SQUAD;


template <typename type_int, typename type_data>
struct Node
{
    /* data */
    type_int count;
    type_int start;
    type_data sum;

    __device__ __host__ Node(){}

    __device__ __host__ Node(type_int _count, type_int _start) : count(_count), start(_start) {}
};


template <typename type_int, typename type_data>
struct Output
{
    /* data */
    int multiplicity;
    type_int row;
    
    type_data value;
 //   type_data cumulative_value;
    type_data forward_cumulative_value;
    
    

    __device__ __host__ Output(){
        multiplicity = 0;
    }

    __device__ __host__ Output(type_int _row, type_data _value, type_int _multiplicity) : row(_row), value(_value), multiplicity(_multiplicity) 
    {
    }


};



template <typename type_int, typename type_data>
struct Edge
{
    /* data */
    type_int row;
    type_int col;
    type_data value;
    int multiplicity;
    int availability;

    __device__ __host__ Edge(){
        multiplicity = 0;
        availability = 0;
    }

    __device__ __host__ Edge(type_int _row, type_int _col, type_data _value) : row(_row), col(_col), value(_value){
        multiplicity = 0;
        availability = 0;
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
    type_int *lower_tri_cptr_start; // used for accessing only lower triangular part
    type_int *lower_tri_cptr_end; // used for accessing only lower triangular part
    type_int nnz;
    bool *merge; // flag to determine whether the entry is merged or not
};

template <typename type_int>
__forceinline__ __device__ type_int next_power_of_two(type_int n) {
    if (n == 0) {
        return 1;  // The next power of 2 greater than 0 is 1
    }
    n--;                // Step 1: Subtract 1
    n |= n >> 1;        // Step 2: Set bits to the right
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;                // Step 3: Add 1 to get the next power of 2
    return n;
}

template <typename type_int>
__forceinline__ __device__ type_int balanced_static_hash(type_int job_id, type_int map_size, type_int total_col)
{
    // ex. 8 -> 4, 9 -> 4
    type_int threshhold = total_col / 2;
    type_int length_per_col = map_size / total_col;
    // if(job_id < threshhold)
    // {
    //     return (job_id * 2 + 1) * length_per_col;
    // }
    // else
    // {
    //     return (total_col - job_id - 1) * 2 * length_per_col;
    // }

    if(job_id < threshhold)
    {
        return ((threshhold - job_id - 1) * 2 + 1) * length_per_col;
    }
    else
    {
        return (total_col - job_id - 1) * 2 * length_per_col;
    }
}


template <typename type_int>
__forceinline__ __device__ type_int permute_static_hash(type_int job_id, type_int map_size, type_int total_col, type_int* rand_vec_device)
{


    type_int length_per_col = map_size / total_col;
    return rand_vec_device[job_id] * length_per_col;

    
}


// A simple DJB2 hash function
__forceinline__ __device__ uint64_t djb2_hash(char *input, int inputLength) {


    // Initialize the hash value (5381 is a magic constant for DJB2)
    unsigned long hash = 5381;
    
    // Hash each character in the input string
    for (int i = 0; i < inputLength; ++i) {
        hash = ((hash << 5) + hash) + input[i]; // hash * 33 + c
    }

    // Return the result
    return hash;
}

__forceinline__ __device__ uint32_t rotl32(uint32_t x, int r) {
    return (x << r) | (x >> (32 - r));
}

__forceinline__ __device__ uint32_t murmurhash3_x86_32(const char *key, int len, uint32_t seed) {
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;
    uint32_t h1 = seed;
    const int nblocks = len / 4;
    
    const uint32_t *blocks = (const uint32_t *)(key + nblocks * 4);
    for (int i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;
        
        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }
    
    const uint8_t *tail = (const uint8_t *)(key + nblocks * 4);
    uint32_t k1 = 0;
    
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
                k1 *= c1; k1 = rotl32(k1, 15); k1 *= c2; h1 ^= k1;
    }
    
    h1 ^= len;
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6b;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35;
    h1 ^= h1 >> 16;
    
    return h1;
}

__forceinline__ __device__ uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

__forceinline__ __device__ uint64_t murmurhash3_x64_64(const char *key, int len, uint64_t seed) {
    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;
    uint64_t h1 = seed;
    uint64_t h2 = seed;
    const int nblocks = len / 16;

    const uint64_t *blocks = (const uint64_t *)(key);
    for (int i = 0; i < nblocks; i++) {
        uint64_t k1 = blocks[i * 2 + 0];
        uint64_t k2 = blocks[i * 2 + 1];
        
        k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
        h1 = rotl64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;
        
        k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
        h2 = rotl64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
    }

    const uint8_t *tail = (const uint8_t *)(key + nblocks * 16);
    uint64_t k1 = 0;
    uint64_t k2 = 0;

    switch (len & 15) {
        case 15: k2 ^= (uint64_t)(tail[14]) << 48;
        case 14: k2 ^= (uint64_t)(tail[13]) << 40;
        case 13: k2 ^= (uint64_t)(tail[12]) << 32;
        case 12: k2 ^= (uint64_t)(tail[11]) << 24;
        case 11: k2 ^= (uint64_t)(tail[10]) << 16;
        case 10: k2 ^= (uint64_t)(tail[9]) << 8;
        case  9: k2 ^= (uint64_t)(tail[8]) << 0;
                 k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
        
        case  8: k1 ^= (uint64_t)(tail[7]) << 56;
        case  7: k1 ^= (uint64_t)(tail[6]) << 48;
        case  6: k1 ^= (uint64_t)(tail[5]) << 40;
        case  5: k1 ^= (uint64_t)(tail[4]) << 32;
        case  4: k1 ^= (uint64_t)(tail[3]) << 24;
        case  3: k1 ^= (uint64_t)(tail[2]) << 16;
        case  2: k1 ^= (uint64_t)(tail[1]) << 8;
        case  1: k1 ^= (uint64_t)(tail[0]) << 0;
                 k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
    }

    h1 ^= len; h2 ^= len;
    
    h1 += h2;
    h2 += h1;

    h1 ^= h1 >> 33;
    h1 *= 0xff51afd7ed558ccdULL;
    h1 ^= h1 >> 33;
    h1 *= 0xc4ceb9fe1a85ec53ULL;
    h1 ^= h1 >> 33;

    h2 ^= h2 >> 33;
    h2 *= 0xff51afd7ed558ccdULL;
    h2 ^= h2 >> 33;
    h2 *= 0xc4ceb9fe1a85ec53ULL;
    h2 ^= h2 >> 33;

    h1 += h2;
    h2 += h1;

    // Return a single 64-bit hash by XOR-ing the two results
    return h1 ^ h2;
}

// a function to search for edge updates
template <typename type_int, typename type_data>
__forceinline__ __device__ type_int search_for_updates(int job_id, int squad_id, int squad_size, int lane_id, type_int map_size, 
    type_int num_of_original_nnz, Node<type_int, type_data> *device_node_list, 
        type_int *device_output_position_idx, type_int *output_start_array, 
            Edge<type_int, type_data> *device_edge_map, Output<type_int, type_data> *device_factorization_output, type_int num_cols, 
                Output<type_int, type_data> *local_output, type_int *rand_vec_device)
{
    

    // check if the number of updates is greater than 0, synchronization ensure that the original value doesn't get updated while reading
    type_int original_update_edge_count = atomicAdd(&device_node_list[job_id].count, 0);
    type_int size_needed = max((original_update_edge_count + num_of_original_nnz) * 2, 1);
    
    // 1 thread fetch some position index and broadcast it through shared memory
    // make two copies here, first entry will be original, second entry is the location decider
    if(lane_id == 0)
    {
        //cuda::atomic_ref<type_int, cuda::thread_scope_device> atomic_position_idx(*device_output_position_idx);
        // output_start_array[squad_id * 4] = atomic_position_idx.fetch_add((original_update_edge_count + 
        //     num_of_original_nnz) * 2);
        // make sure there is at least 1 entry in order to store the diagonal entry
        
        
        output_start_array[squad_id * 4 + 2] = 0;
  
        if(size_needed < LOCAL_CAPACITY)
        {
            output_start_array[squad_id * 4] = atomicAdd(device_output_position_idx, original_update_edge_count + num_of_original_nnz + 1);
            output_start_array[squad_id * 4 + 1] = 0;
            output_start_array[squad_id * 4 + 3] = 1;
        }
        else
        {
            output_start_array[squad_id * 4] = atomicAdd(device_output_position_idx, size_needed);
            output_start_array[squad_id * 4 + 1] = output_start_array[squad_id * 4];
            output_start_array[squad_id * 4 + 3] = 0;
        }
        
        
            
        // update device_node_list with the necessary locations to find the edges
        device_node_list[job_id].start = output_start_array[squad_id * 4];
    
        
    }

        
    // synchronize to make sure values are ready
    if(squad_size == 32)
    {
        //__threadfence();
        __syncwarp();
    }
    else
    {
        // TO DO
        // possibly use the named barrier available in CUTLASS
        //namedBarrierSync(squad_id, squad_size);
        __syncthreads();
    }


 
    if(original_update_edge_count > 0)
    {
        
        // swtich to local if size needed is smaller than allocation
        if(size_needed < LOCAL_CAPACITY)
        {
            device_factorization_output = local_output;
        }


        // broadcast the starting value
        //cuda::atomic_ref<type_int, cuda::thread_scope_block> atomic_squad_position_idx(output_start_array[squad_id * 4]);

        type_int search_iter = 0;

        // look for relevant updates in device_edge_map, using job_id as the hash index
        //uint64_t hash = djb2_hash((char *)(&job_id), sizeof(type_int));
        //uint64_t hash = murmurhash3_x64_64((char *)(&job_id), sizeof(type_int), job_id);
        //type_int hash = balanced_static_hash(job_id, map_size, num_cols);
        type_int hash = permute_static_hash(job_id, map_size, num_cols, rand_vec_device);
        
        type_int storage_location = hash % map_size;
        

        // if(job_id == 17748 && lane_id == 0)
        // {
        //     printf("how many to look for: %lld\n", original_update_edge_count);
        // }
        
        while(atomicAdd(&output_start_array[squad_id * 4 + 2], 0) < original_update_edge_count)
        {
            
            // need to check availability before fetching the edge, 2 means that it is both occupied and FINISHED UPDATING
            // need to account for wrap around (i.e. if index exceed bound), that's why there is a % map_size
            type_int slot = (storage_location + lane_id + squad_size * search_iter) % map_size;
            
            
            int original_availability = 2;
            type_int search_availability_val = atomicCAS(&device_edge_map[slot].availability, original_availability, 1);

            // type_int search_availability_val = 2;
            // cuda::atomic_ref<int, cuda::thread_scope_device> at_avail_acq(device_edge_map[slot].availability);
            // at_avail_acq.compare_exchange_strong(search_availability_val, 1, cuda::memory_order_acquire);
            
            
            if(search_availability_val == 2)
            {
                Edge<type_int, type_data> &candidate = device_edge_map[slot];
                if(candidate.col == job_id)
                {
                    // if(job_id == 17748)
                    // {
                    //     printf("search slot: %lld\n", slot);
                    // }
                    // it's a hit
                    type_int insert_location = atomicAdd(&output_start_array[squad_id * 4 + 1], 1);
                    atomicAdd(&output_start_array[squad_id * 4 + 2], candidate.multiplicity);
                    Output<type_int, type_data> created_output(candidate.row, 
                        candidate.value, candidate.multiplicity);
                    
                    device_factorization_output[insert_location] = created_output;
                    // free up the slot
                    //__threadfence();
                    //atomicExch(&device_edge_map[slot].availability, 0);

                    device_edge_map[slot].row = 0;
                    device_edge_map[slot].col = 0;
                    device_edge_map[slot].multiplicity = 0;
                    
                    __threadfence();
                    atomicExch(&device_edge_map[slot].availability, 0);

                    // cuda::atomic_ref<int, cuda::thread_scope_device> at_avail_rel(device_edge_map[slot].availability);
                    // at_avail_rel.exchange(0, cuda::memory_order_release);

                    //availability.store(0);
                    
                }
                else
                {
                    // indicating that it's done with the search
                    atomicExch(&device_edge_map[slot].availability, 2);
                    // cuda::atomic_ref<int, cuda::thread_scope_device> at_avail_rel(device_edge_map[slot].availability);
                    // at_avail_rel.exchange(2, cuda::memory_order_release);
                    //availability.store(2);
                }
                
            }
            else if(search_availability_val == 1)
            {
                continue;
            }

          

            search_iter++;
        }
        // if(lane_id == 0 && search_iter > 100)
        // {
        //     printf("search iter: %d, job id: %d, problematic %d\n", search_iter, job_id, problematic);
        // }
        

       
        
    
    }

    // wait for update to device factorization output to finish
    // make sure everyone sees the update
    if(squad_size == 32)
    {
        //__threadfence();
        __syncwarp();
    }
    else
    {
        __syncthreads();
    }
    // don't return original_update_edge_count since this value contains duplicates, use the difference, which is the actual number of unique value
    if(output_start_array[squad_id * 4 + 3])
    {
        // if uses local space, then starts at 0, so no need to subtract
        return output_start_array[squad_id * 4 + 1];
    }
    else
    {
        return output_start_array[squad_id * 4 + 1] - output_start_array[squad_id * 4];
    }
    
}



// Kernel to perform binary search
template <typename type_int, typename type_data>
__forceinline__ __device__ void batched_binary_search(type_int* row, type_data* data, type_int n, Output<type_int, type_data>* d_candidates, 
    type_int numKeys, bool *merge, int lane_id, int squad_size) 
{
    
    
    for(;lane_id < numKeys; lane_id += squad_size){
        
        int64_t key = d_candidates[lane_id].row;
        
        int64_t left = 0;
        int64_t right = n - 1;
        int64_t result = -1; // Default result for not found

        // Perform binary search
        while (left <= right && result == -1) {
            int64_t mid = left + (right - left) / 2;

            // if (row[mid] == key) {
            //     result = mid; // Key found at index mid
            //     break;
            // } else if (row[mid] < key) {
            //     left = mid + 1;
            // } else {
            //     right = mid - 1;
            // }
            left = (row[mid]< key) * (mid + 1) + (row[mid] >= key) * left;
            right = (row[mid]<= key) * right + (row[mid] > key) * (mid - 1);
            result = (row[mid] == key) * mid - (row[mid] != key);
        }

        // Store the result (index of the found key or -1 if not found)
        if(result != -1)
        {
           
            // set merge to 1 so that the data won't be copied twice
            d_candidates[lane_id].value += data[result];
            d_candidates[lane_id].multiplicity += 1;
            merge[result] = 1;
           
        }

      
    
    }

    // sync here to make sure that d_candidates and merge are ready
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

template <typename type_int, typename type_data>
__forceinline__ __device__ type_int sample_search(curandState &state, Output<type_int, type_data> *device_factorization_output, type_int n, 
    type_data total_sum, type_data offset)
{
    double random_num = curand_uniform_double(&state);
    // double key = random_num * total_sum;
    double key = random_num * (total_sum - offset) + offset;

    

    type_int left = 0;
    type_int right = n;
    type_int mid = left + (right - left) / 2;
    int64_t result = -1; // Default result for not found
   
    // Perform binary search
    while (left < right && result == -1) {
        mid = left + (right - left) / 2;
        //printf("thread id: %d, left: %lld, right: %lld, mid: %lld\n", threadIdx.x, left, right, mid);
        //type_data target = (device_factorization_output[mid].forward_cumulative_value - offset);
        type_data target = (device_factorization_output[mid].forward_cumulative_value);
    
        // if(pls == 67164)
        // {
        //     printf("target: %.25lf, key: %.25lf, compare: %d, equal: %d, forward val: %.25lf, \
        //         offset: %.25lf\n", target, key, target < key, target == key, device_factorization_output[mid].forward_cumulative_value, offset);
        // }
        // if(total_sum == offset)
        // {
        //     printf("target: %.25lf, key: %.25lf, compare: %d, equal: %d\n", target, key, target < key, target == key);
        // }
        
        left = (target < key) * (mid + 1) + (target >= key) * left;
        right = (target <= key) * right + (target > key) * (mid);
        result = (target == key) * mid - (target != key);
    }

    // if(job_id == 553 || job_id == 554 || job_id == 556)
    // {
    //     printf("job id: %d, random number: %f, remaining cumulative value: %f, own row: %d, target: %d\n", job_id, random_num, (total_sum - offset), own_row, device_factorization_output[left].row);
    // }
 

    // return the index of the first element that is greater than or equal to key
    if(result == -1)
        return left;
    else 
        return result;

}



__forceinline__ __device__ bool compare_row(const void* a, const void* b) {
    const Output<int, double> *o_a = (const Output<int, double>*)a;
    const Output<int, double> *o_b = (const Output<int, double>*)b;

    return o_a->row < o_b->row;
}

__forceinline__ __device__ bool compare_row_reference(Output<int, double> &a, Output<int, double> &b) {


    return a.row < b.row;
}


__forceinline__ __device__ bool compare_value(const void* a, const void* b) {
    const Output<int, double> *o_a = (const Output<int, double>*)a;
    const Output<int, double> *o_b = (const Output<int, double>*)b;

    return (o_a->value < o_b->value) || (o_a->value == o_b->value && o_a->row < o_b->row);
    //return (o_a->value < o_b->value);
}

__forceinline__ __device__ bool compare_value_reference(Output<int, double> &a, Output<int, double> &b) {


    return a.value < b.value;
}


template <typename type_int, typename type_data>
__device__ void odd_even_sort(Output<type_int, type_data> *input, type_int left, type_int right, int start_id, bool (*operation)(const void*, const void*))
{
    
    // type_int count = right - left;
    // type_int bound = count / 2 + (count & 1);
    // type_int id = start_id;
    
    // // __shared__ int check;
    // // if(threadIdx.x == 0)
    // // {
    // //     check = 0;
    // // }
    // // __syncwarp();
  
    
   
    
    // bool sorted = false;
    // for(type_int i = 0; i < bound && !sorted; i++)
    // {
    //     type_int true_id = left + id * 2;
    //     bool first_pass = true;
    //     if(true_id + 1 < right)  //even phase
    //     {
    //         // type_int temp = (input[true_id].row <= input[true_id + 1].row) * input[true_id].row + (input[true_id].row > input[true_id + 1].row) * input[true_id + 1].row;
    //         // type_int o_temp = (input[true_id].row <= input[true_id + 1].row) * input[true_id + 1].row + (input[true_id].row > input[true_id + 1].row) * input[true_id].row;
    //         // input[true_id].row = temp;
    //         // input[true_id + 1].row = o_temp;

            
    //         //input[true_id].row > input[true_id + 1].row
    //         if(operation(&input[true_id + 1], &input[true_id]))
    //         {
    //             Output<type_int, type_data> temp = input[true_id];
    //             input[true_id] = input[true_id + 1];
    //             input[true_id + 1] = temp;
    //             first_pass = false;
    //         }
            
    //     }

    //     __syncwarp();

    //     bool second_pass = true;
    //     true_id = left + id * 2 + 1;
    //     if(true_id + 1 < right)     //odd phase
    //     {
    //         if(operation(&input[true_id + 1], &input[true_id]))
    //         {
    //             Output<type_int, type_data> temp = input[true_id];
    //             input[true_id] = input[true_id + 1];
    //             input[true_id + 1] = temp;
    //             second_pass = false;
    //         }
            
    //         // type_int temp = (input[true_id].row <= input[true_id + 1].row) * input[true_id].row + (input[true_id].row > input[true_id + 1].row) * input[true_id + 1].row;
    //         // type_int o_temp = (input[true_id].row <= input[true_id + 1].row) * input[true_id + 1].row + (input[true_id].row > input[true_id + 1].row) * input[true_id].row;
    //         // input[true_id].row = temp;
    //         // input[true_id + 1].row = o_temp;
    //     }


    //     //bool result = 1;
    //     //WarpScanAnd(temp_storage_scan[warp_id]).InclusiveScan(first_pass && second_pass, result, Logical_and(), sorted);
        
    //     // atomicCAS(&check, 1, (int)(first_pass && second_pass));
    //     // sorted = check;

    //     // atomicAdd(&check, (int)(first_pass && second_pass));
    //     // sorted = (check == 32);
        
    //     // if(threadIdx.x == 0)
    //     // {
    //     //     check = 0;
    //     // }
    //     // if(sorted == true && threadIdx.x == 0){
    //     //     printf("iteration: %d\n", i);
    //     // }
    //     __syncwarp();

    //     // if(m == 0)
    //     // {
    //     //     printf("iteration: %d, id: %d, result: %d, first pass %d, second pass %d, first and second %d\n", m, threadIdx.x, result, first_pass, second_pass, first_pass && second_pass);
    //     // }

        
        
    ////for }
    




    if(right == left)
    {
        return;
    }
    type_int count = right - left;
    type_int bound = count / 2 + (count & 1);
    type_int sorted = 0;


    
    for(type_int i = 0; i < bound && !sorted; i++)
    {
        bool first_pass = true;
        for(type_int id = start_id; id * 2 < count; id += THREADS_PER_WARP * WARPS_PER_SQUAD)
        {
            // if(threadIdx.x == 0)
            // {
            //     printf("sorted: %d\n", sorted);
            //     printf("finished: %d\n", sorted != expected_count);
            // }
            type_int true_id = left + id * 2;
            if(true_id + 1 < right)  //even phase
            {
                // type_int temp = (input[true_id].row <= input[true_id + 1].row) * input[true_id].row + (input[true_id].row > input[true_id + 1].row) * input[true_id + 1].row;
                // type_int o_temp = (input[true_id].row <= input[true_id + 1].row) * input[true_id + 1].row + (input[true_id].row > input[true_id + 1].row) * input[true_id].row;
                // input[true_id].row = temp;
                // input[true_id + 1].row = o_temp;

                
                //input[true_id].row > input[true_id + 1].row
                if(operation(&input[true_id + 1], &input[true_id]))
                {
                    Output<type_int, type_data> temp = input[true_id];
                    input[true_id] = input[true_id + 1];
                    input[true_id + 1] = temp;
                    first_pass = false;
                }
                
            }


            
        }
        
        //__syncwarp(__activemask());
        //int first_vote = __syncthreads_and(first_pass);
        __syncthreads();

        bool second_pass = true;
        for(type_int id = start_id; id * 2 < count; id += THREADS_PER_WARP * WARPS_PER_SQUAD)
        {
           

          

            
            type_int true_id = left + id * 2 + 1;
            if(true_id + 1 < right)     //odd phase
            {
                
                //input[true_id].row > input[true_id + 1].row
                if(operation(&input[true_id + 1], &input[true_id]))
                {
                    Output<type_int, type_data> temp = input[true_id];
                    input[true_id] = input[true_id + 1];
                    input[true_id + 1] = temp;
                    second_pass = false;
                }
                
                // type_int temp = (input[true_id].row <= input[true_id + 1].row) * input[true_id].row + (input[true_id].row > input[true_id + 1].row) * input[true_id + 1].row;
                // type_int o_temp = (input[true_id].row <= input[true_id + 1].row) * input[true_id + 1].row + (input[true_id].row > input[true_id + 1].row) * input[true_id].row;
                // input[true_id].row = temp;
                // input[true_id + 1].row = o_temp;
            }

            
            
        }

        //__syncwarp(__activemask());
        //int second_vote = __syncthreads_and(second_pass);
        __syncthreads();
        // if list is sorted, terminate early
        //sorted = (first_vote && second_vote);
        
    }

}

template <typename type_int, typename type_data>
__device__ void bitonic_sort(Output<type_int, type_data> *input, type_int left, type_int right, int id, bool (*operation)(const void*, const void*)) 
{
    // determine id
 
    type_int total_size = right - left;
    for(type_int outer_level = 1; outer_level < total_size; outer_level = outer_level << 1)
    {
        // determine whether to make sequence increasing or decreasing using id
        bool inc = ((id / outer_level) % 2) == 0; // this should be determined by the outer most layer
        int outer_gap = outer_level << 1;
        for(type_int level = outer_level; level >= 1; level = level >> 1)
        {
            int gap = level << 1;
            int inner_lane = id % level;
            int inner_group = id / level;
            int group_size = min(level, THREADS_PER_WARP * WARPS_PER_SQUAD);
            int jump_size = max(level, THREADS_PER_WARP * WARPS_PER_SQUAD) * 2;
            type_int true_id = inner_group * gap + inner_lane + left;
            

            // if(threadIdx.x == 0)
            // {
            //     printf("outer level: %d, level: %d, \n", outer_level, level);
            // }
                

            for(type_int outer = true_id; outer < right; outer += jump_size)
            {
                // update inc if one round of loop isn't enough to cover all elements and threads need to jump to the next working set
                inc = ((id / outer_level + (outer - inner_lane - left - inner_group * gap) / outer_gap) % 2) == 0;
                // if(left == 0)
                // {
                    // __syncwarp();
                    // for(type_int i = outer; i < outer - inner_lane + gap; i += level){
                    //     printf("before outer level: %d, level: %d, row: %d, i: %d, \n", outer_level, level, input[i].row, i);
                    // }
                    // for(type_int i = outer; i < outer - inner_lane + 10; i += level){
                    //     printf("row: %d, i: %d, long: %d, smaller count: %d\n", input[i].row, i, start_of_next_seg + other_less_count + other_equal_count - (smaller_count) + inner_lane, smaller_count);
                    // }
                    
                    
                    //__syncwarp();
                // }
                
                
                for(type_int i = outer; i < outer - inner_lane + level; i += group_size)
                {
                    // if increase
                    if(inc)
                    {
                        // swap the larger elements to right side
                        
                        //input[i].row > input[i + level].row
                        if(operation(&input[i + level], &input[i]))
                        {
                            Output<type_int, type_data> swap_element = input[i];
                            input[i] = input[i + level];
                            input[i + level] = swap_element;
                        }
                    }
                    else
                    {
                        // swap smaller elements to right side
                        // input[i].row < input[i + level].row
                        if(operation(&input[i], &input[i + level]))
                        {
                            Output<type_int, type_data> swap_element = input[i];
                            input[i] = input[i + level];
                            input[i + level] = swap_element;
                        }
                    }
                }

                

                //    __syncwarp();
                //     for(type_int i = outer; i < outer - inner_lane + gap; i += level){
                //         printf("after outer level: %d, level: %d, row: %d, i: %d, outer - inner_lane - left: %d\n", outer_level, level, input[i].row, i, outer - inner_lane - left);
                //     }
                //     // for(type_int i = outer; i < outer - inner_lane + 10; i += level){
                //     //     printf("row: %d, i: %d, long: %d, smaller count: %d\n", input[i].row, i, start_of_next_seg + other_less_count + other_equal_count - (smaller_count) + inner_lane, smaller_count);
                //     // }
                    
                    
                //     __syncwarp();
            }

            // __syncwarp(__activemask());
            //__syncwarp();
            __syncthreads();

        }
         

        
    }
}




  // for(type_int i = edge_start + lane_id; i < bound; i += squad_size)
            // {
                
            //     type_data input_val = 0.0;
            //     type_data output_val = 0.0;
            //     if(i < edge_start + total_neighbor_count)
            //     {
            //         input_val = device_factorization_output[i].value;
            //         output_val = device_factorization_output[i].cumulative_value;
            //     }
          
               
            //     WarpScan(temp_storage_scan[warp_id]).InclusiveSum(input_val,
            //                                      output_val,
            //                                      warp_aggregate);
             
            //     if(i < edge_start + total_neighbor_count)
            //     {
            //         device_factorization_output[i].cumulative_value = output_val + total_sum;
            //         if(job_id == 15835)
            //         {
            //             printf("cumulative_value: %.25lf\n", device_factorization_output[i].cumulative_value);
                        
            //         }  
            //     }
            //     total_sum += warp_aggregate;
            //     if(squad_size == 32)
            //     {
            //         __threadfence();
            //         __syncwarp();
            //     }
            // }

//   type_data sum_prev = 0.0;

//                     sum_prev = device_factorization_output[i].cumulative_value;
                 
//                 type_int sample_index = sample_search(state, device_factorization_output + i + 1, edge_start + total_neighbor_count - i - 1, 
//                     total_sum, sum_prev);
//                 if(job_id == 15835)
//                 {
//                     printf("edge_start here: %lld, actual sample index: %lld, i + 1: %lld, diff: %.25lf, offset: %.25lf, equal 0: %d\n", edge_start, sample_index, i + 1, total_sum - sum_prev, sum_prev, (total_sum - sum_prev) == 0);
                    
//                 }  
//                 sample_index += (i + 1); // account for offset
//                 type_int row_of_sample = max(device_factorization_output[i].row, device_factorization_output[sample_index].row);
//                 type_int col_of_sample = min(device_factorization_output[i].row, device_factorization_output[sample_index].row);
//                 type_data value_of_sample = device_factorization_output[i].value * (total_sum - sum_prev) / total_sum;

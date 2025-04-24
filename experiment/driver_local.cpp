

//#include <mpi.h>
#include <chrono>
#include "custom_cg.hpp"
#include <atomic>


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

/*
void compute_parmetis_ordering(char *input_string)
{
    typedef MPI_Datatype CustomMPIType; 
    CustomMPIType MY_RECEIVE_MPI_TYPE = MPI_INT64_T;
    CustomMPIType MY_SEND_MPI_TYPE = MPI_INT64_T;
    if (sizeof(idx_t) == 4){
        MY_RECEIVE_MPI_TYPE = MPI_INT;  // Define custom_type as 32-bit integer
        MY_SEND_MPI_TYPE = MPI_INT;
    }

    
    
     
    int rank, size;
    int root_process = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm comm = MPI_COMM_WORLD;
    std::cout << "my rank: " << rank << "\n";
    if(rank == root_process && (sizeof(idx_t) != 8 || sizeof(size_t) != 8))
    {
        std::cout << "size of idx_t: " << sizeof(idx_t) << "\n";
        std::cout << "size of size_t: " << sizeof(size_t) << "\n";
        std::cout << "WARNING! data type int needs to be 8 byte to avoid overflow\n";
        printf("------------------------------\n\n\n");
        //exit(0);
    }

    
    // vertices set up
    size_t num_cols;
    int vtxdist[size]; // SHOULDN'T STACK OVERFLOW BECAUSE NUMBER OF PROCESSES IS USUALLY SMALL
    idx_t actual_vtxdist[size + 1];
    int send_count[size];
    int cumulative_sum[size + 1];
    size_t vertex_per_process;
    size_t remainder;
    idx_t *xadj;// CSR format

    // Adjacency list setup
    int element_count[size];
    int element_count_cumulative[size + 1];
    idx_t *adjncy; 

    // permutation that is collected by the root node at the end
    idx_t *permutation;
    if(rank == root_process)
    {
        
        sparse_matrix_processor<custom_idx, double> processor(input_string);
        num_cols = processor.mat.cols();
        permutation = (idx_t *)malloc(num_cols * sizeof(idx_t));
        // calculate number of vertices that each process is responsible for
        vertex_per_process = num_cols / size;
        remainder = num_cols % size;
        
        for(size_t i = 0; i < size; i++)
        {
            vtxdist[i] = vertex_per_process + (i < remainder);
        }
        // needs to account for the 1 extra bound
        cumulative_sum[0] = 0;
        actual_vtxdist[0] = 0;
        for(size_t i = 1; i < size + 1; i++)
        {
            cumulative_sum[i] = cumulative_sum[i - 1] + vtxdist[i - 1] + 1;
            actual_vtxdist[i] = actual_vtxdist[i - 1] + vtxdist[i - 1];
        }
        MPI_Bcast(&num_cols, 1, MPI_UINT64_T, root_process, MPI_COMM_WORLD);
        xadj = (idx_t *)malloc((vtxdist[rank] + 1) * sizeof(idx_t));
        adjncy = (idx_t *)malloc((processor.mat.col_ptrs[actual_vtxdist[rank + 1]] - processor.mat.col_ptrs[actual_vtxdist[rank]]) * sizeof(idx_t));
        std::cout << "xadj size: " << vtxdist[rank] + 1 << "\n";
        std::cout << "adjncy size: " << (processor.mat.col_ptrs[actual_vtxdist[rank + 1]] - processor.mat.col_ptrs[actual_vtxdist[rank]]) << "\n";

        // fill in xadj and adjncy, might not need since scatterv does it for root as well
        // size_t counter = 0;
        // for (int k = cumulative_sum[rank]; k < cumulative_sum[rank + 1]; ++k) 
        // {
        //     xadj[k - cumulative_sum[rank]] = processor.mat.col_ptrs[k];
        //     for (typename sparse_matrix<custom_idx, double>::InnerIterator it(mat, k); it; ++it) 
        //     {
        //         adjncy[counter] = it.row();
        //         counter++;
        //     }
        // }

        // set up send count and scatter the rest of the sparse matrix to everyone else
       
        for(int i = 0; i < size; i++)
        {
            send_count[i] = vtxdist[i] + 1;
        }  
        // need to convert data type in case the ints have size mismatch(i.e. 32 vs 64 bits)
        if(sizeof(custom_idx) == sizeof(idx_t))
        {
            MPI_Scatterv(processor.mat.col_ptrs.data(), send_count, actual_vtxdist, MY_SEND_MPI_TYPE, xadj, vtxdist[rank] + 1, MY_RECEIVE_MPI_TYPE, root_process, MPI_COMM_WORLD);
        }
        else
        {
            idx_t *temp_data_conversion = (idx_t *)malloc(processor.mat.col_ptrs.size() * sizeof(idx_t));
            for(size_t i = 0; i < processor.mat.col_ptrs.size(); i++)
            {
                temp_data_conversion[i] = processor.mat.col_ptrs.data()[i];
            }
            MPI_Scatterv(temp_data_conversion, send_count, actual_vtxdist, MY_SEND_MPI_TYPE, xadj, vtxdist[rank] + 1, MY_RECEIVE_MPI_TYPE, root_process, MPI_COMM_WORLD);
             
            free(temp_data_conversion);
        }
        
        


        // set up nonzero count and broadcast it to other processes
        for(size_t i = 0; i < size; i++)
        {
            element_count[i] = processor.mat.col_ptrs[actual_vtxdist[i + 1]] - processor.mat.col_ptrs[actual_vtxdist[i]];
        }
       
        element_count_cumulative[0] = 0;
        for(size_t i = 1; i < size + 1; i++)
        {
            element_count_cumulative[i] = element_count_cumulative[i - 1] + element_count[i - 1];
        }
       
        MPI_Bcast(element_count, size, MPI_INT, root_process, MPI_COMM_WORLD);
        
      
        if(sizeof(custom_idx) == sizeof(idx_t))
        {
            MPI_Scatterv(processor.mat.col_ptrs.data(), element_count, element_count_cumulative, MY_SEND_MPI_TYPE, adjncy, element_count[rank], MY_RECEIVE_MPI_TYPE, root_process, MPI_COMM_WORLD);
        }
        else
        {
            idx_t *temp_data_conversion = (idx_t *)malloc(processor.mat.row_indices.size() * sizeof(idx_t));
            for(size_t i = 0; i < processor.mat.row_indices.size(); i++)
            {
                temp_data_conversion[i] =processor.mat.row_indices.data()[i];
            }
            MPI_Scatterv(temp_data_conversion, element_count, element_count_cumulative, MY_SEND_MPI_TYPE, adjncy, element_count[rank], MY_RECEIVE_MPI_TYPE, root_process, MPI_COMM_WORLD);
            free(temp_data_conversion);
        }
       

        //   for(int i = 0; i < 144650; i++){
        //     if(xadj[i] != processor.mat.col_ptrs[i])
        //         std::cout << "gooby " << xadj[i] << "\n";
        // }
        // std::cout << xadj[144649] << "\n";
        // std::cout <<  "space \n";
        // for(int i = 0; i < 2148786; i++){
        //     if(adjncy[i] != processor.mat.row_indices[i]){
        //         std::cout << "gooby " << adjncy[i] << "\n";
        //     }
            
        // }
       
        
    }
    else
    {
        
        MPI_Bcast(&num_cols, 1, MPI_UINT64_T, rank, MPI_COMM_WORLD);
        // calculate number of vertices that each process is responsible for
        vertex_per_process = num_cols / size;
        remainder = num_cols % size;
       // std::cout << "num cols: " << num_cols << "\n" << std::flush;
        for(size_t i = 0; i < size; i++)
        {
            vtxdist[i] = vertex_per_process + (i < remainder);
        }

        // needs to account for the 1 extra bound
        cumulative_sum[0] = 0;
        actual_vtxdist[0] = 0;
        for(size_t i = 1; i < size + 1; i++)
        {
            cumulative_sum[i] = cumulative_sum[i - 1] + vtxdist[i - 1] + 1;
            actual_vtxdist[i] = actual_vtxdist[i - 1] + vtxdist[i - 1];
        }

        // receive xadj
      //  std::cout << "vtxdist[rank] + 1: " << vtxdist[rank] + 1 << "\n" << std::flush;
        xadj = (idx_t *)malloc((vtxdist[rank] + 1) * sizeof(idx_t));
        for(int i = 0; i < size; i++)
        {
            send_count[i] = vtxdist[i] + 1;
        }  
        MPI_Scatterv(NULL, send_count, actual_vtxdist, MY_SEND_MPI_TYPE, xadj, vtxdist[rank] + 1, MY_RECEIVE_MPI_TYPE, root_process, MPI_COMM_WORLD);
        

        // receive element count
        MPI_Bcast(element_count, size, MPI_INT, root_process, MPI_COMM_WORLD);
        element_count_cumulative[0] = 0;
        for(size_t i = 1; i < size + 1; i++)
        {
            element_count_cumulative[i] = element_count_cumulative[i - 1] + element_count[i - 1];
        }

        // receive adjncy
        adjncy = (idx_t *)malloc(element_count[rank] * sizeof(idx_t));
        MPI_Scatterv(NULL, element_count, element_count_cumulative, MY_SEND_MPI_TYPE, adjncy, element_count[rank], MY_RECEIVE_MPI_TYPE, root_process, MPI_COMM_WORLD);
    }

    // if(rank == 0){
    //     for(int i = 0; i < size + 1; i++)
    //     {
    //         std::cout << "vtxdist: " <<  actual_vtxdist[i] << "\n";
    //     }
    //     std::cout <<  "space \n";

    

    //     for(int i = 0; i < vtxdist[0] + 1; i++)
    //     {
    //         std::cout << "xadj: " << xadj[i] << "\n" << std::flush;
    //     }
    // }
    
    // subtract offset so all arrays start at 0
    idx_t offset = xadj[0];
    for(size_t i = 0; i < vtxdist[rank] + 1; i++)
    {
        xadj[i] -= offset;
    }

    

   
    idx_t numflag = 0; // C-style numbering
    idx_t options[3] = {0};
    idx_t *order = (idx_t *)malloc(vtxdist[rank] * sizeof(idx_t));
    // idx_t levels_of_partition = floor(log2(size));
    // idx_t partition_size = 2 ^ levels_of_partition + 2 ^ (levels_of_partition - 1) + 10;
    idx_t *partition_info =(idx_t *)malloc(2 * size * sizeof(idx_t));

   
    
  
    auto start = std::chrono::high_resolution_clock::now();
    ParMETIS_V3_NodeND(actual_vtxdist, xadj, adjncy, &numflag, options, order, partition_info, &comm);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to find elimination order using parmetis: " << duration.count() << " seconds" << std::endl;
    
    int actual_vtxdist_copy[size + 1];
    for(int i = 0; i < size + 1; i++)
    {
        actual_vtxdist_copy[i] = actual_vtxdist[i];
    }
    // gather the permutation to root
    MPI_Gatherv(order, vtxdist[rank], MY_SEND_MPI_TYPE,
                permutation, vtxdist, actual_vtxdist_copy,
                MY_RECEIVE_MPI_TYPE, root_process, MPI_COMM_WORLD);
    
    printf("my rank: %d\n", rank);
    if(rank == root_process)
    {

        // std::vector<idx_t> permutation_vec;
        // permutation_vec.assign(permutation, permutation + num_cols);
        // writeVectorToFile(permutation_vec, "permutation_vec.txt");
    }
}
*/


template <typename type_int, typename type_data>
void perform_factorization_device(const custom_space::sparse_matrix<type_int, type_data> &spmat_device, type_int map_size,
    Edge<type_int, type_data> *device_edge_map, type_int output_size,
       type_int *device_min_dependency_count, Node<type_int, type_data> *device_node_list, type_int *device_output_position_idx, type_int *queue_device,
            type_int thread_id, type_int total_threads, int64_t *test_vec, int *schedule_id) 
{

    // // set thread schedule affinity
    // cpu_set_t cpuset;
    // CPU_ZERO(&cpuset);
    // CPU_SET(schedule_id[thread_id], &cpuset); // Assign thread to CPU corresponding to thread_num

    // pthread_t thread = pthread_self();
    // int ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);



    //int thread_id = omp_get_thread_num();
    //int gap = omp_get_num_threads();
    int gap = total_threads;
    //printf("id: %d, cpu: %d, gap: %d\n", thread_id, sched_getcpu(), gap);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    int neg_track = 0;
    int positive_track = 0;

    type_int num_cols = spmat_device.num_cols;
    type_int output_start_array[4];
    std::vector<Edge<type_int, type_data>> local_work_space_vec(2000);

    auto start = std::chrono::steady_clock::now();

    // assign the job id, will consider using a scheduler in the future
    // if id is within valid range of queue'
    type_int queue_access_index = thread_id;
    int64_t job_id = -1;
    std::atomic_ref<type_int> queue_size(queue_device[num_cols]);
    //std::atomic_ref<int64_t> test_size(test_vec[num_cols]);
    
    if(queue_access_index < queue_size.load())
    {
        std::atomic_ref<type_int> poll_id(queue_device[queue_access_index]);

        job_id = poll_id.load();
        // this means that even though the size was modified, the actual job haven't been inserted
        if(job_id == 0 && queue_access_index != 0)
        {
            job_id = -1;
        }
    }



    // skip num_cols - 1 since the last column is empty
    //int static_id = 0;
    while(queue_access_index < num_cols)
    {
        
        
        if(job_id == -1)
        {
         
            
            // if(queue_access_index < queue_size.load())
            // {
            //     std::atomic_ref<type_int> poll_id(queue_device[queue_access_index]);
            //     job_id = poll_id.load();
            //     // this means that even though the size was modified, the actual job haven't been inserted
            //     if(job_id == 0 && queue_access_index != 0)
            //     {
                    
            //         job_id = -1;
            //     }
                
               
            // }
            std::atomic_ref<type_int> poll_id(queue_device[queue_access_index]);
            
            // this means that even though the size was modified, the actual job haven't been inserted
            while(poll_id.load(std::memory_order_acquire) == 0 && queue_access_index != 0)
            {
                
                job_id = -1;
            }
            job_id = poll_id.load(std::memory_order_acquire);
            // std::atomic_ref<int64_t> test_id(test_vec[queue_access_index]);
            // if(test_id.load() == 0 && queue_access_index != 0)
            // {
            //     job_id = -1;
            // }


            // if (queue_access_index == queue_size.load())
            // {
            //     printf("job id: %d, queue_access_index: %d, queue_size.load(): %d\n", job_id, queue_access_index, queue_size.load());
            //     exit(0);
            // }
                
            // will keep skip until gets a valid id 
            neg_track++;
            continue;
        }
        positive_track++;
      // job_id = static_id;
      //  static_id++;
        //curand_init(0, job_id, 0, &state);
     
        

        // job successfully queued up, move to next search location
        queue_access_index += gap;
        // job id is the last column, skip
        if(job_id == num_cols - 1)
        {
            
            std::atomic_ref<type_int> last_col_loc_ref(*device_output_position_idx);
            type_int last_col_loc = last_col_loc_ref.fetch_add(1);

            device_node_list[num_cols - 1].start = last_col_loc;
            device_node_list[num_cols - 1].count = 0;
            device_node_list[num_cols - 1].sum = 0.0;
        
            
            job_id = -1;
            continue;
        }
        
        // if(job_id % 1000 == 0)
        // {
        //     printf("thread id: %d, cpu id: %d\n", omp_get_thread_num(), sched_getcpu());
        // }

        // check if column is ready for factorization
        //cuda::atomic_ref<type_int, cuda::thread_scope_device> atomic_dependency_count(device_min_dependency_count[job_id]);
        

        std::atomic_ref<type_int> atomic_dependency_count(device_min_dependency_count[job_id]);
         
        //cuda::atomic_ref<type_int, cuda::thread_scope_device> atomic_dependency_count(device_min_dependency_count[job_id]);

        
        
        while(atomic_dependency_count.load() > 0)
        {
            
           
            //continue;
        }

        // if(job_id == 143800)
        // {
        //     type_int sum = 0;
        //     for(type_int gg = 143801; gg < 334863; gg++)
        //     {
        //         sum += device_node_list[gg].count;
        //     }
        //     printf("fill: %d\n", sum);
        //     //exit(0);
        // }
        

        /* search for updates, get location information */ 
        // if(sched_getcpu() != thread_id)
        // {
        //     printf("uh oh\n");
        // }
        
            
        type_int left_bound_idx = spmat_device.col_ptrs[job_id];
        type_int num_of_original_nnz = spmat_device.col_ptrs[job_id + 1] - left_bound_idx;
        type_int update_neighbor_count = search_for_updates<type_int, type_data>(job_id, map_size, num_of_original_nnz,
            device_node_list, device_output_position_idx, output_start_array, device_edge_map, local_work_space_vec);
        
        type_int total_neighbor_count = num_of_original_nnz + update_neighbor_count;
        Edge<type_int, type_data> *local_work_space = local_work_space_vec.data();

        // if no entry in this column
        if(total_neighbor_count == 0)
        {
            
            device_node_list[job_id].count = 0;
            device_node_list[job_id].sum = 0.0;
            job_id = -1;
            continue;
        }

      
        // printf("total neighbor at job id %d: %d, device node prev: %d\n", job_id, total_neighbor_count, device_node_list[job_id].prev);
       
        {
           
            /* perform merging */
            //type_int edge_start = output_start_array[0];
            

            type_int edge_start = 0;

         
           
           

            // 1. read in the nonzeros in the original input and sort entire input first based on row value
            if(edge_start + total_neighbor_count >= local_work_space_vec.size())
            {
                local_work_space_vec.resize((edge_start + total_neighbor_count + 1) * 2);
                local_work_space = local_work_space_vec.data();
            }
            
            for(type_int i = edge_start + update_neighbor_count; i < edge_start + total_neighbor_count; i++)
            {
                local_work_space[i] = Edge<type_int, type_data>(spmat_device.row_indices[left_bound_idx + i - edge_start - update_neighbor_count], 
                    spmat_device.values[left_bound_idx + i - edge_start - update_neighbor_count], 1);

            }
         

            //odd_even_sort(device_factorization_output, edge_start, edge_start + total_neighbor_count, lane_id, compare_row);
            std::sort(local_work_space + edge_start, local_work_space + edge_start + total_neighbor_count, [](const Edge<type_int, type_data>& a, const Edge<type_int, type_data>& b) {
                return a.row < b.row;
            });


            // if(job_id == 314)
            // {
            //     printf("total neighbor: %d, update_neighbor_count: %d\n", total_neighbor_count, update_neighbor_count);
            //     assert(false);
            // }
            
              
   
            // 3. merge entries with the same row value
            // count how many actual distinct elements there are after merging
            type_int actual_neighbor_count = 0;
            if(total_neighbor_count > 0)
            {
                Edge<type_int, type_data> *edge_ref = &local_work_space[edge_start];
                actual_neighbor_count++;
                
                for(type_int i = edge_start + 1; i < edge_start + total_neighbor_count; i++)
                {
                    if(local_work_space[i].row == edge_ref->row)
                    {
                        // merge the two
                        edge_ref->value += local_work_space[i].value;
                        edge_ref->multiplicity += local_work_space[i].multiplicity;
                    }
                    else
                    {
                        // update the chase pointer, shift the new distinct element to here, DON'T ACCUMULATE (i.e. no +=)
                        edge_ref++;
                        actual_neighbor_count++;
                        edge_ref->value = local_work_space[i].value;
                        edge_ref->multiplicity = local_work_space[i].multiplicity;
                        edge_ref->row = local_work_space[i].row;
                    }
                    
                }
            }
          

            
            
            
            // update device_node_list with new locations after merging and update values after merging
            device_node_list[job_id].count = actual_neighbor_count;
            total_neighbor_count = actual_neighbor_count;


            // 5. sort input based on value
            //odd_even_sort(device_factorization_output, edge_start, edge_start + total_neighbor_count, lane_id, compare_value);
            std::sort(local_work_space + edge_start, local_work_space + edge_start + total_neighbor_count, [](const Edge<type_int, type_data>& a, const Edge<type_int, type_data>& b) {
                return a.value < b.value;
            });
        
            

             
            
            // compute cumulative sum
            type_data total_sum = 0.0;
            if(total_neighbor_count > 0)
            {
                local_work_space[edge_start].forward_cumulative_value = local_work_space[edge_start].value;
                total_sum += local_work_space[edge_start].value;
                for(type_int i = edge_start + 1; i < edge_start + total_neighbor_count; i++)
                {
                    local_work_space[i].forward_cumulative_value = local_work_space[i].value + local_work_space[i - 1].forward_cumulative_value;
                    total_sum += local_work_space[i].value;
                }
            }
            device_node_list[job_id].sum = total_sum;
            
            

            /* Generate Samples and set up links */
            for(type_int i = edge_start; i < edge_start + total_neighbor_count - 1; i++)
            {
                double number_decision = dis(gen) * (total_sum - local_work_space[i].forward_cumulative_value);
                Edge<type_int, type_data> *edge_iter = std::lower_bound(local_work_space + i + 1, local_work_space + edge_start + total_neighbor_count, number_decision, 
                    [local_work_space, i](const Edge<type_int, type_data>& element, const double& value) -> bool {
                        return (element.forward_cumulative_value - local_work_space[i].forward_cumulative_value) < value;
                    });
                type_int generated_row = std::max(edge_iter->row, local_work_space[i].row);
                type_int generated_col = std::min(edge_iter->row, local_work_space[i].row);
                type_data generated_value = local_work_space[i].value * (total_sum - local_work_space[i].forward_cumulative_value) / total_sum;

                // set up the generated edge
                local_work_space[i].sampled_row = generated_row;
                local_work_space[i].sampled_value = generated_value;

                // atomically update the link, must use atomics here
                
                std::atomic_ref<int64_t> link_ref(device_node_list[generated_col].prev);
                local_work_space[i].prev = link_ref.exchange(i + output_start_array[0]);

                // add count
                std::atomic_ref<type_int> col_count_ref(device_node_list[generated_col].count);
                col_count_ref.fetch_add(1);

                // add dependency update
                std::atomic_ref<type_int> dependency_update(device_min_dependency_count[generated_row]);
                dependency_update.fetch_add(1);
                // if(job_id == 37969)
                // {
                //     printf("sampled row: %d, generated col: %d, job id: %d, value: %f, row of loop: %d\n", generated_row, generated_col, job_id, local_work_space[i].value, local_work_space[i].row);
                // }


            }


            // scale by total sum
            for(type_int i = edge_start; i < edge_start + total_neighbor_count; i++)
            {
                local_work_space[i].value = local_work_space[i].value / total_sum;

            }
                
            
            // copy to global space
            for(type_int i = edge_start; i < edge_start + total_neighbor_count; i++)
            {
                device_edge_map[i + output_start_array[0]] = local_work_space[i]; 
            }

      

            // update dependency by subtracting away from ones impacted by current node/column
       
          
            for(type_int i = edge_start + output_start_array[0]; i < edge_start + total_neighbor_count + output_start_array[0]; i++)
            {
                

                
                //type_int old_dependency = atomicAdd(&device_min_dependency_count[device_factorization_output[i].row], -(type_int)(device_factorization_output[i].multiplicity));
                std::atomic_ref<type_int> dependency_ref(device_min_dependency_count[device_edge_map[i].row]);
                type_int old_dependency = dependency_ref.fetch_sub(device_edge_map[i].multiplicity);
                //cuda::atomic_ref<type_int, cuda::thread_scope_device> dependency_sync(device_min_dependency_count[device_factorization_output[i].row]);
                //type_int old_dependency = dependency_sync.fetch_sub(device_factorization_output[i].multiplicity, cuda::memory_order_release);
                //printf("job id: %d, row: %d, old_dependency: %d, multiplicity: %d\n", job_id, device_edge_map[i].row, old_dependency, device_edge_map[i].multiplicity);
                if(old_dependency == device_edge_map[i].multiplicity)
                {

                   type_int old_queue_size = queue_size.fetch_add(1);
                    //queue_size = atomicAdd(&queue_device[num_cols], 1);
                   std::atomic_ref<type_int> job_schedule_ref(queue_device[old_queue_size]);
                   job_schedule_ref.exchange(device_edge_map[i].row, std::memory_order_release);
                    // #pragma omp capture
                    // {
                    //     type_int old_val = queue_device[old_queue_size];
                    //     queue_device[old_queue_size] = device_edge_map[i].row;
                    // }
                    
                    // int64_t old_test_size = test_size.fetch_add(1);
                    // std::atomic_ref<int64_t> test_ref(test_vec[old_test_size]);
                    // test_ref.exchange(device_edge_map[i].row);
                    //atomicExch(&queue_device[queue_size], device_factorization_output[i].row);
     
                }

    
                
                
            }
            
       
      
            

        }

        // reset job id to -1, so it will look for a new job
        job_id = -1;
      
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = end - start;

    // printf("id: %d, cpu: %d, negative track: %d, positive track: %d, time: %f\n", thread_id, sched_getcpu(), 
    //     neg_track, positive_track, duration.count());


    
}

bool compare(const Edge<int, double> &a, const Edge<int, double> &b)
{
    return a.row < b.row;  
};


template <typename type_int, typename type_data>
void factorization_driver(sparse_matrix_processor<type_int, type_data> &processor, type_int num_threads, char* path, bool is_graph)
{
    assert(INT_MAX == 2147483647);
    int space_multiply_factor = 5;
    int edge_pool_size = processor.mat.nonZeros() * space_multiply_factor;
    if(edge_pool_size < 0)
    {
        printf("WARNING: allocation requirement became negative, indicating an int overflow");
        edge_pool_size = INT_MAX - 100000000;
        //assert(false);
    }
    printf("Edge size: %ld\n", sizeof(Edge<type_int, type_data>{}));

    // Allocate memory on the host (CPU)
    Edge<type_int, type_data> *host_edge_map = (Edge<type_int, type_data> *)malloc(edge_pool_size * sizeof(Edge<type_int, type_data>{}));
    for(size_t i = 0; i < edge_pool_size; i++) 
    {
        
        host_edge_map[i] = Edge<type_int, type_data>();
    }
    

    type_int host_output_position_idx = 0;



    
    // copy sparse matrix from cpu to gpu
    custom_space::sparse_matrix<type_int, type_data> spmat = processor.make_lower_triangular(processor.mat);


    // copy min_dependency_array
    if(spmat.num_cols != processor.min_dependency_count.size())
    {
        printf("processor.mat.num_cols: %d, spmat.num_cols: %d, min_dependency_count.size(): %ld\n", processor.mat.num_cols, spmat.num_cols, processor.min_dependency_count.size());
        assert(spmat.num_cols == processor.min_dependency_count.size());
    }
    
    //printf("dependency count at 5: %d\n", processor.min_dependency_count[5]);
    std::vector<type_int> queue_cpu(spmat.num_cols + 2, 0); // last two element represent counters
    std::vector<int64_t> test_vec(spmat.num_cols + 1, 0);
    
    for(type_int i = 0; i < processor.min_dependency_count.size(); i++)
    {
        
        if(processor.min_dependency_count[i] == 0)
        {
            queue_cpu[queue_cpu[queue_cpu.size() - 2]] = i;
            queue_cpu[queue_cpu.size() - 2]++;
            //printf("scheduled: %d\n", i);

            test_vec[test_vec[test_vec.size() - 1]] = i;
            test_vec[test_vec.size() - 1]++;
       }
    
        
    }
    // std::random_device rd;
    // std::mt19937 generator(rd());

    //std::shuffle(queue_cpu.begin() + 1, queue_cpu.begin() + 10626424, generator);
    assert(queue_cpu[0] == 0);
    printf("initial queue size: %d\n", queue_cpu[queue_cpu.size() - 2]);

    std::vector<type_int> min_dependency_count = processor.min_dependency_count;

    
    
    // create node array
    std::vector<Node<type_int, type_data>> node_list_host(spmat.num_cols);
    for(size_t i = 0; i < spmat.num_cols; i++)
    {
        node_list_host[i] = Node<type_int, type_data>(0, 0, i);
    }
  

    printf("CURRENTLY, MULTIPLICITY STORAGE USES INT32, DON'T NEED 64 BIT UNLESS IN EXTREME CIRCUMSTANCES \n");
    printf("CURRENTLY, BINARY SEARCH USES INT32, DON'T NEED 64 BIT UNLESS IN EXTREME CIRCUMSTANCES \n");
    printf("defaulting row and col to 0 in map may cause problem for column 0 \n");
    

   
    //int schedule_id[] = {0, 128, 1, 129, 2, 130, 3, 131, 4, 132, 5, 133, 6, 134, 7, 135};
    int schedule_id[] = {0, 128, 1, 129, 2, 130, 3, 131, 4, 132, 5, 133, 6, 134, 7, 135, 8, 136, 9, 137, 10, 138, 11, 139, 12, 140, 13, 141, 14, 142, 15, 143};
    //int schedule_id[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    
    omp_set_num_threads(num_threads);
    //kmp_set_defaults("KMP_AFFINITY=norespect|KMP_SETTINGS=1");

    // Parallel region starts
    #pragma omp parallel
    {
        auto start = std::chrono::steady_clock::now();
        // cpu_set_t cpuset;
        // CPU_ZERO(&cpuset);
        // CPU_SET(omp_get_thread_num(), &cpuset); // Assign thread to CPU corresponding to thread_num

        // pthread_t thread = pthread_self();
        // int ret = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
        // printf("thread id: %d, cpu id: %d\n", omp_get_thread_num(), sched_getcpu());
        perform_factorization_device<type_int, type_data>(spmat, edge_pool_size, host_edge_map, edge_pool_size, min_dependency_count.data(), node_list_host.data(), 
            &host_output_position_idx, queue_cpu.data(), omp_get_thread_num(), num_threads, test_vec.data(), schedule_id);
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> duration = end - start;

        #pragma omp barrier

        if(omp_get_thread_num() == 0)
        {
            printf("Factorization execution time: %f seconds\n", duration.count());
            //std::cout << "Factorization execution time: " << duration.count() << " seconds" << std::endl;
        }
        
    }
    
    /*
    std::vector<std::thread> threads;
   
    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(perform_factorization_device<type_int, type_data>, 
            spmat, edge_pool_size, host_edge_map, edge_pool_size, min_dependency_count.data(), node_list_host.data(), 
                &host_output_position_idx, queue_cpu.data(), i, num_threads, test_vec.data(), schedule_id);
        sched_param sch_params;
        sch_params.sched_priority = 99; // Example priority
        if (pthread_setschedparam(threads[i].native_handle(), SCHED_FIFO, &sch_params)) {
            //std::cerr << "Failed to set thread scheduling parameters.\n";
        }
    }


    // Join threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    */

    /*
    std::vector<std::future<void>> futures;
    
    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        
    
        futures.emplace_back(std::async(std::launch::async, perform_factorization_device<type_int, type_data>, 
            spmat, edge_pool_size, host_edge_map, edge_pool_size, min_dependency_count.data(), node_list_host.data(), 
                &host_output_position_idx, queue_cpu.data(), i, num_threads, test_vec.data(), schedule_id));
        
        
    }

    // Join threads
    for (int i = 0; i < num_threads; ++i) {
        futures[i].get();
    }
    */     
    

    

  

    // make sure to add 1 to the size of each column since factorization did not include diagonal
    std::vector<type_int> csr_rowptr_host(spmat.num_cols + 1);
    std::vector<type_data> diagonal_entries(spmat.num_cols);
    csr_rowptr_host[0] = 0;
    size_t total_needed_size = 0;
    for(size_t i = 0; i < node_list_host.size(); i++)
    {
        
        total_needed_size = total_needed_size + node_list_host[i].count + 1;
        csr_rowptr_host[i + 1] = total_needed_size;

        // sort list and compute sum, also negate entries since the factorization was done with positive entries
        type_int col_start = node_list_host[i].start;
        //printf("col_start: %d, col_size: %d\n", col_start, node_list_host[i].count + 1);
        // if(i > 0)
        // {
        //     if(col_start < node_list_host[i - 1].start + node_list_host[i - 1].count)
        //     {
        //         printf("i - 1: %d, prev col start: %d, prev count: %d, cur start: %d\n", i - 1, node_list_host[i - 1].start, node_list_host[i - 1].count, col_start);
        //         assert(col_start > node_list_host[i - 1].start + node_list_host[i - 1].count);
        //     }
            
        // }
    
        // compute column sum and append diagonal, negate entries
        for(size_t j = col_start; j < col_start + node_list_host[i].count; j++)
        {
            host_edge_map[j].value = -host_edge_map[j].value;
        }
        diagonal_entries[i] = node_list_host[i].sum;

        // UPDATE COUNT
        host_edge_map[col_start + node_list_host[i].count] = Edge<type_int, type_data>(i, 1.0, 1);
        //Output<type_int, type_data>(i, 1.0, 1);
        // if(i == 40089)
        // {
        //     printf("insert for %ld at: %d\n", i, col_start);
        //     // printf("output row: %d\n", device_factorization_output_host[col_start + node_list_host[i].count].row);
        //     for(int j = col_start; j < col_start + node_list_host[i].count + 1; j++)
        //     {
        //         printf("output row: %d, val: %f\n", device_factorization_output_host[j].row, device_factorization_output_host[j].value);
        //     }
        // }
        node_list_host[i].count++;
        std::sort(host_edge_map + col_start, host_edge_map + col_start + node_list_host[i].count, compare);
        // if(true)
        // {
        //     printf("insert for 0 at: %d\n", node_list_host[i].count);
        //     // printf("output row: %d\n", device_factorization_output_host[col_start + node_list_host[i].count].row);
        //     for(int j = col_start; j < col_start + node_list_host[i].count; j++)
        //     {
        //         printf("after output row: %d\n", device_factorization_output_host[j].row);
        //     }
        // }
    }
    
    printf("nnz ratio: %f, factor nnz: %d, original nnz: %d\n", double(total_needed_size) / double(spmat.nonZeros()), total_needed_size, spmat.nonZeros());
    std::vector<type_data> csr_val_host(total_needed_size);
    std::vector<type_int> csr_col_ind_host(total_needed_size);

    // start writing the result into a csr, preparing for cusparse operations
    for(size_t i = 0; i < node_list_host.size(); i++)
    {
        
        for(size_t j = csr_rowptr_host[i]; j < csr_rowptr_host[i + 1]; j++)
        {
            csr_col_ind_host[j] = host_edge_map[node_list_host[i].start + j - csr_rowptr_host[i]].row;
            csr_val_host[j] = host_edge_map[node_list_host[i].start + j - csr_rowptr_host[i]].value;

            // assert that diagonal elements are 1
            if(host_edge_map[node_list_host[i].start + j - csr_rowptr_host[i]].row == i)
            {
                assert(host_edge_map[node_list_host[i].start + j - csr_rowptr_host[i]].value == 1.0);
            }
            assert(host_edge_map[node_list_host[i].start + j - csr_rowptr_host[i]].row >= i);
        }
    }

    // write solution to file
    std::string prefix(path);
    if(prefix.length() > 0)
    {
        std::string filename = "l_sol.mtx";
        std::string wpath = path + filename;
        std::ofstream output_stream(wpath);
        if (!output_stream.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
            exit(1);
        }
        printf("rowptr size: %d\n", csr_rowptr_host.size());
        printf("col indices size: %d\n", csr_col_ind_host.size());
        // write_csr_to_matrix_market(csr_rowptr_host, csr_col_ind_host, csr_val_host, spmat.num_cols, spmat.num_cols, "c_sol.mtx");
        fast_matrix_market::matrix_market_header header(diagonal_entries.size(), diagonal_entries.size());
        header.object = fast_matrix_market::matrix;
        header.symmetry = fast_matrix_market::general;
        fast_matrix_market::write_options opts;
        opts.precision = 16;
        fast_matrix_market::write_matrix_market_csc(output_stream,
                                    header, 
                                    csr_rowptr_host,
                                    csr_col_ind_host,
                                    csr_val_host,
                                    false,
                                    opts);
        output_stream.flush();  // Ensure any buffered output is written to the file
        output_stream.close();  // Close the file stream when done
        std::string diagname = "sol_diag.txt";
        writeVectorToFile(diagonal_entries, (path + diagname).c_str());
    }

    // find the real e-tree
    std::vector<type_int> etree(spmat.num_cols, 0);
    
    for(size_t i = 0; i < spmat.num_cols; i++)
    {
        if(csr_rowptr_host[i] + 1 < csr_rowptr_host[i + 1])
        {
            etree[i] = csr_col_ind_host[csr_rowptr_host[i] + 1];
        }
    }
    std::vector<std::vector<type_int>> ftree = processor.create_factorization_tree_from_etree(etree);

    std::vector<type_int> layer_info = processor.layer_information(ftree);

    size_t verify_count = 0;
    for(size_t i = 0; i < layer_info.size(); i++)
    {
        verify_count += layer_info[i];
    }
    std::cout << "actual depth of tree after factorization: " << layer_info.size() - 1 << ", total count: " << verify_count << ", number of partitions: " << layer_info[1] << "\n";

    // triangular solve longest DAG path

    std::vector<size_t> max_path_dp(spmat.num_cols, 1);
    for(size_t i = 0; i < spmat.num_cols; i++)
    {
        size_t left_end = csr_rowptr_host[i];
        size_t right_end = csr_rowptr_host[i + 1];

        for(size_t j = left_end + 1; j < right_end; j++)
        {
            max_path_dp[csr_col_ind_host[j]] = std::max(max_path_dp[i] + 1, max_path_dp[csr_col_ind_host[j]]);
        }
        
    }
    auto max_path = std::max_element(max_path_dp.begin(), max_path_dp.end());

    printf("triangular solve max path: %d at index: %d\n", *max_path, std::distance(max_path_dp.begin(), max_path));
  
    

    custom_space::sparse_matrix<type_int, type_data> precond_M(processor.mat.rows(), processor.mat.cols(), std::move(csr_val_host), std::move(csr_col_ind_host), std::move(csr_rowptr_host));

    if(num_threads == 32)
    {
       example_pcg_solver(processor.mat, precond_M, diagonal_entries.data(), is_graph);
    }
    
    for(int li = 0; li < 10; li++)
    {
        printf("\n");
    }

    /*
    bool check_solve = 0;
    if(check_solve)
    {
        // copy the answer back to cpu
        cudaMemcpy(node_list_host.data(), device_node_list, spmat.num_cols * sizeof(Node<type_int, type_data>{}), cudaMemcpyDeviceToHost);
        std::vector<Output<type_int, type_data>> device_factorization_output_host(edge_pool_size);
        cudaMemcpy(device_factorization_output_host.data(), device_factorization_output, edge_pool_size * sizeof(Output<type_int, type_data>{}), cudaMemcpyDeviceToHost);
        

        // TODO: REMEMBER TO FREE STUFF AT THE END
        cudaFree(device_node_list);
        cudaFree(device_factorization_output);
        cudaFree(device_edge_map);
        
        // make sure to add 1 to the size of each column since factorization did not include diagonal
        std::vector<type_int> csr_rowptr_host(spmat.num_cols + 1);
        std::vector<type_data> diagonal_entries(spmat.num_cols);
        csr_rowptr_host[0] = 0;
        size_t total_needed_size = 0;
        for(size_t i = 0; i < node_list_host.size(); i++)
        {
            
            total_needed_size = total_needed_size + node_list_host[i].count + 1;
            csr_rowptr_host[i + 1] = total_needed_size;

            // sort list and compute sum, also negate entries since the factorization was done with positive entries
            type_int col_start = node_list_host[i].start;
            //printf("col_start: %d, col_size: %d\n", col_start, node_list_host[i].count + 1);
            // if(i > 0)
            // {
            //     if(col_start < node_list_host[i - 1].start + node_list_host[i - 1].count)
            //     {
            //         printf("i - 1: %d, prev col start: %d, prev count: %d, cur start: %d\n", i - 1, node_list_host[i - 1].start, node_list_host[i - 1].count, col_start);
            //         assert(col_start > node_list_host[i - 1].start + node_list_host[i - 1].count);
            //     }
                
            // }
       
            // compute column sum and append diagonal, negate entries
            for(size_t j = col_start; j < col_start + node_list_host[i].count; j++)
            {
                device_factorization_output_host[j].value = -device_factorization_output_host[j].value;
            }
            diagonal_entries[i] = node_list_host[i].sum;

            // UPDATE COUNT
            device_factorization_output_host[col_start + node_list_host[i].count] = Output<type_int, type_data>(i, 1.0, 1);
            // if(i == 40089)
            // {
            //     printf("insert for %ld at: %d\n", i, col_start);
            //     // printf("output row: %d\n", device_factorization_output_host[col_start + node_list_host[i].count].row);
            //     for(int j = col_start; j < col_start + node_list_host[i].count + 1; j++)
            //     {
            //         printf("output row: %d, val: %f\n", device_factorization_output_host[j].row, device_factorization_output_host[j].value);
            //     }
            // }
            node_list_host[i].count++;
            std::sort(device_factorization_output_host.data() + col_start, device_factorization_output_host.data() + col_start + node_list_host[i].count, compare);
            // if(true)
            // {
            //     printf("insert for 0 at: %d\n", node_list_host[i].count);
            //     // printf("output row: %d\n", device_factorization_output_host[col_start + node_list_host[i].count].row);
            //     for(int j = col_start; j < col_start + node_list_host[i].count; j++)
            //     {
            //         printf("after output row: %d\n", device_factorization_output_host[j].row);
            //     }
            // }
        }
        
        std::vector<type_data> csr_val_host(total_needed_size);
        std::vector<type_int> csr_col_ind_host(total_needed_size);

        // start writing the result into a csr, preparing for cusparse operations
        for(size_t i = 0; i < node_list_host.size(); i++)
        {
          
            for(size_t j = csr_rowptr_host[i]; j < csr_rowptr_host[i + 1]; j++)
            {
                csr_col_ind_host[j] = device_factorization_output_host[node_list_host[i].start + j - csr_rowptr_host[i]].row;
                csr_val_host[j] = device_factorization_output_host[node_list_host[i].start + j - csr_rowptr_host[i]].value;

                // assert that diagonal elements are 1
                if(device_factorization_output_host[node_list_host[i].start + j - csr_rowptr_host[i]].row == i)
                {
                    assert(device_factorization_output_host[node_list_host[i].start + j - csr_rowptr_host[i]].value == 1.0);
                }
                assert(device_factorization_output_host[node_list_host[i].start + j - csr_rowptr_host[i]].row >= i);
            }
        }
        // int max_row = 0;
        // double max_val = 0.0;
        // double min_val = 1.0;
        // for(int i = 0; i < csr_rowptr_host.size() - 1; i++)
        // //for(int i = 0; i < 10; i++)
        // {
        //     for(size_t j = csr_rowptr_host[i]; j < csr_rowptr_host[i + 1]; j++)
        //     {
        //         //if(std::isnan(csr_val_host[j]))
        //         if(true)
        //         {
        //             printf("col: %d, row: %d, factor value: %f\n", i, csr_col_ind_host[j], csr_val_host[j]);
                    
        //         }
        //         max_row = max(max_row, csr_col_ind_host[j]);
        //         max_val = max(max_val, abs(csr_val_host[j]));
        //         min_val = min(min_val, abs(csr_val_host[j]));
        //     }
           
            
        // }
        // printf("max row: %d\n", max_row);
        // printf("max val: %f\n", max_val);
        // printf("min val: %f\n", min_val);
        
        // double max_val = 0.0;
        // for(int i = 0; i < spmat.num_cols; i++)
        // //for(int i = 0; i < 10; i++)
        // {
        //     if(true)
        //     {
        //         printf("diagonal entries at %d: %f\n", i, diagonal_entries[i]);
                
        //     }
        //     max_val = max(max_val, diagonal_entries[i]);
            
        // }
        // printf("max val: %f\n", max_val);
      
        // for(int i = 0; i < processor.mat.col_ptrs.size() - 1; i++)
        // {
        //     for(size_t j = processor.mat.col_ptrs[i]; j < processor.mat.col_ptrs[i + 1]; j++)
        //     {
        //         if(i < 50)
        //         {
        //             printf("col: %d, row: %d, factor value: %f\n", i, processor.mat.row_indices[j], processor.mat.values[j]);
                 
        //         }
                
        //     }
        // }

        // write solution to file
        std::string wpath = "c_sol.mtx";
        std::ofstream output_stream(wpath);
        if (!output_stream.is_open()) {
            std::cerr << "Failed to open file for writing." << std::endl;
            exit(1);
        }
        // write_csr_to_matrix_market(csr_rowptr_host, csr_col_ind_host, csr_val_host, spmat.num_cols, spmat.num_cols, "c_sol.mtx");
        fast_matrix_market::matrix_market_header header(diagonal_entries.size(), diagonal_entries.size());
        header.object = fast_matrix_market::matrix;
        header.symmetry = fast_matrix_market::general;
        fast_matrix_market::write_matrix_market_csc(output_stream,
                                 header, 
                                 csr_rowptr_host,
                                 csr_col_ind_host,
                                 csr_val_host,
                                 false);
        output_stream.flush();  // Ensure any buffered output is written to the file
        output_stream.close();  // Close the file stream when done

        prepare_and_solve(processor.mat, csr_rowptr_host, csr_val_host, csr_col_ind_host, diagonal_entries);

        // read in preconditioner for verification
        // Load the matrix from file
        // std::vector<type_data> diag_vec;

        // // Open the file
        // std::ifstream file("diag");
        
        // // Check if the file was successfully opened
        // if (!file.is_open()) {
        //     std::cerr << "Failed to open the file!" << std::endl;
        //     exit(1);  // Exit with error code
        // }

        // // Read each double from the file
        // type_data number;
        // while (file >> number) {
        //     diag_vec.push_back(number);  // Store each number into the vector
        // }

        // // Close the file
        // file.close();
        // std::string path = "verify";
        // std::ifstream input_stream(path);
        // triplet_matrix<type_int, type_data> input_triplet;
        // input_triplet.nrows = 28924;
        // input_triplet.ncols = 28924;
        // // read input into triplet
        // fast_matrix_market::read_matrix_market_triplet(
        //     input_stream, input_triplet.nrows, input_triplet.ncols, input_triplet.rows, input_triplet.cols, input_triplet.vals);

        // sparse_matrix<type_int, type_data> cscMatrix = processor.triplet_to_csc_with_diagonal(input_triplet.nrows, input_triplet.ncols,
        //                     input_triplet.rows,
        //                     input_triplet.cols,
        //                     input_triplet.vals);
        
       

    
        //prepare_and_solve(processor.mat, cscMatrix.col_ptrs, cscMatrix.values, cscMatrix.row_indices, diag_vec);
        
    }
    else
    {
        // TODO: REMEMBER TO FREE STUFF AT THE END
        cudaFree(device_node_list);
        cudaFree(device_factorization_output);
        cudaFree(device_edge_map);
    }
    


    */






    

  
    
  
}



int main(int argc, char* argv[]) {

   // MPI_Init(&argc, &argv);
    
  
    //compute_parmetis_ordering(argv[1]);
    printf("problem: %s\n", argv[1]);
    sparse_matrix_processor<custom_idx, double> processor(argv[1]);
    
    if(argc == 4)
    {
        factorization_driver<custom_idx, double>(processor, atoi(argv[2]), argv[3], 1);
    }
    else if(argc > 4)
    {
        factorization_driver<custom_idx, double>(processor, atoi(argv[2]), argv[3], 0);
    }
    else
    {
        printf("argument count not correct\n");
        assert(false);
    }
    


  //  MPI_Finalize();
    return 0;
    
   

     


    
}

// export KMP_AFFINITY=norespect
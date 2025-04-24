#include "auxilliary.hpp"
#include <random>
#include <cusparse.h>
#include <cublas_v2.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

template <typename type_data>
void generate_random_vector(type_data *h_vec, size_t n, unsigned long seed) 
{
    std::mt19937 gen(seed);                     // Mersenne Twister RNG
    std::uniform_real_distribution<type_data> dist(-1.0, 1.0); // Range [-1, 1]


    for (size_t i = 0; i < n; i++) {
        h_vec[i] = dist(gen);
    }

 
}

template <typename type_data>
void generate_zero_sum_vector(type_data *h_vec, size_t n, unsigned long seed) 
{
    std::mt19937 gen(seed);                     // Mersenne Twister RNG
    std::uniform_real_distribution<type_data> dist(0.0, 1.0); // Range [-1, 1]

    type_data sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        h_vec[i] = dist(gen);
        sum += h_vec[i];
    }

    // // Adjust the vector so the sum of all entries is zero
    type_data mean = sum / n;
    for (size_t i = 0; i < n; i++) {
        h_vec[i] -= mean;
        
    }
}

template <typename type_data> 
__global__ void diagonal_solve(type_data *diagonal, type_data *left_raw_array, size_t num_cols)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    
    for(size_t i = id; i < num_cols; i += num_threads)
    {
        if(diagonal[i] != 0.0)
        {
          
            left_raw_array[i] = left_raw_array[i] / diagonal[i];
            
        }
        else
        {
            //left_raw_array[i] = 0;
        }
        //left_raw_array[i] = left_raw_array[i] / 2;
        
    }
}

template <typename type_data> 
__global__ void diagonal_mul_32t(type_data *diagonal, type_data *left_raw_array, size_t num_cols)
{

    // for(size_t i = threadIdx.x; i < num_cols; i += 32)
    // {
    //     if(diagonal[i] != 0.0)
    //     {
    //         left_raw_array[i] = left_raw_array[i] * diagonal[i];
    //     }
    //     else
    //     {
    //         //left_raw_array[i] = 0;
    //     }

        
    // }
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    //printf("id: %d, num threads: %d\n", id, num_threads);

    for(size_t i = id; i < num_cols; i += num_threads)
    {
        if(diagonal[i] != 0.0)
        {
            left_raw_array[i] = left_raw_array[i] * diagonal[i];
            
        }
        else
        {
            //left_raw_array[i] = 0;
        }

        
    }
    // if(threadIdx.x == 0)
    // {
    //     for(size_t i = 0; i < num_cols; i++)
    //     {
    //         if(diagonal[i] != 0.0)
    //         {
    //             left_raw_array[i] = left_raw_array[i] * diagonal[i];
    //         }
    //         else
    //         {
    //             //left_raw_array[i] = 0;
    //         }
    //     }
    // }

}

template <typename type_data> 
__global__ void shift_center(type_data *input, size_t num_cols, type_data mean)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    for(size_t i = id; i < num_cols; i += num_threads)
    {
        input[i] -= mean;
    }
}

// z = M^{-1} * r = L^{-T}D^{-1}L^{-1} * r
template <typename type_data> 
int apply_preconditioner(cusparseHandle_t &cusparseHandle, cublasHandle_t &cublasHandle, cusparseSpMatDescr_t &descr_L,
    cusparseSpSVDescr_t  &spsv_descr_L, cusparseSpSVDescr_t  &spsv_descr_U, cusparseDnVecDescr_t &left, const cusparseDnVecDescr_t &right, 
    cusparseDnVecDescr_t& vec_intermediate, type_data *diagonal, type_data *intermediate_raw_array, size_t num_cols, bool use_preconditioner)
{

    if(use_preconditioner)
    {
        type_data alpha = 1.0;
        cudaMemset(intermediate_raw_array, 0, num_cols);
        // double *lefthost = (double *)malloc(num_cols * sizeof(double));
        // cudaMemcpy(lefthost, intermediate_raw_array, num_cols * sizeof(type_data), cudaMemcpyDeviceToHost);
        // for(int i = 0; i < num_cols; i++)
        // {
        //     printf("lefthost[%d] before lower solve: %f\n", i, lefthost[i]);
        // }
        // printf("\n\n\n");
        // lower triangular solve
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                        &alpha, descr_L, right, vec_intermediate, CUDA_R_64F,
                                        CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_L) );

        // cudaMemcpy(lefthost, intermediate_raw_array, num_cols * sizeof(type_data), cudaMemcpyDeviceToHost);
        // for(int i = 0; i < 1; i++)
        // {
        //     printf("lefthost[%d] after lower solve: %f\n", i, lefthost[i]);
        // }                                        
        // printf("\n\n\n");
    
        // diagonal solve

        diagonal_solve<type_data><<<96, 256>>>(diagonal, intermediate_raw_array, num_cols);
        
        cudaDeviceSynchronize();

        // cudaMemcpy(lefthost, intermediate_raw_array, num_cols * sizeof(type_data), cudaMemcpyDeviceToHost);
        // for(int i = 0; i < 1; i++)
        // {
        //     if(lefthost[i] != 0.0)
        //     {
        //         printf("lefthost[%d] after diag solve: %f\n", i, lefthost[i]);
        //     }
            
        // }    
        // printf("\n\n\n");
    

        // upper triangular solve
        CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, descr_L, vec_intermediate, left, CUDA_R_64F,
                                        CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_U) );

        // double *left_raw;
        // cusparseDnVecGetValues(left, (void**)&left_raw);
        // double mean = 0.0;
        // cublasDasum(cublasHandle, num_cols, left_raw, 1, &mean);
        // mean /= num_cols;
        // shift_center<type_data><<<96, 256>>>(left_raw, num_cols, mean);
        // cudaDeviceSynchronize();

        // double *left_raw;
        // cusparseDnVecGetValues(left, (void**)&left_raw);
        // cudaMemcpy(lefthost, left_raw, num_cols * sizeof(type_data), cudaMemcpyDeviceToHost);
        // for(int i = 0; i < num_cols; i++)
        // {
        //     printf("lefthost[%d] after upper solve: %f\n", i, lefthost[i]);
        // }    
        // printf("\n\n\n");                            
    }
    else
    {
        double *data_left;
        cusparseDnVecGetValues(left, (void**)&data_left);

        double *data_right;
        cusparseDnVecGetValues(right, (void**)&data_right);

        cudaMemcpy(data_left, data_right, num_cols * sizeof(type_data), cudaMemcpyDeviceToDevice);

    }
        

    return EXIT_SUCCESS;
}


template <typename type_int, typename type_data>
int prepare_and_solve(sparse_matrix_device<type_int, type_data> &laplacian, type_int *csr_rowptr_device, 
    type_data *csr_val_device, type_int *csr_col_ind_device, type_data *diagonal_entries_device, double tolerance, bool physics, type_int factorization_nnz, bool removal)
{
    cudaEvent_t start, stop;
    float milliseconds = 0;
    
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    

    /* 1. copy factorization to device */


    // copy original laplacian to device
    type_int *lap_rowptr_device = laplacian.col_ptrs;
    type_int *lap_col_ind_device = laplacian.row_indices;
    type_data *lap_val_device = laplacian.values;




    /* 2. solve  ----------------------------------------------------------       */ 
   

    // create random right side and left side vector, move them to gpu
    size_t num_cols = laplacian.num_cols;
    size_t nnz = laplacian.nnz;
    printf("post trimming, num cols: %ld, laplacian nnz: %ld, factorization nnz: %ld\n", num_cols, nnz, factorization_nnz);
    type_data *rightside = (type_data *)malloc(num_cols * sizeof(type_data));
    //generate_zero_sum_vector<type_data>(rightside, num_cols, 0);
    std::vector<type_data> leftside(num_cols, 0.0);
    

    type_data *rightside_device;
    cudaMalloc((void **)&rightside_device, num_cols * sizeof(type_data));


    type_data *leftside_device;
    cudaMalloc((void **)&leftside_device, num_cols * sizeof(type_data));
    cudaMemcpy(leftside_device, leftside.data(), num_cols * sizeof(type_data), cudaMemcpyHostToDevice);

    



    // initialize handles
    cusparseHandle_t cusparseHandle;
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    bool use_preconditioner = true;


    // initialize some variables
    type_data cg_alpha, cg_beta, r0, r1;
    int k = 0;

    // Allocate GPU memory for required vectors
    type_data *d_r, *d_p, *d_z, *d_Ap, *d_M, *intermediate;
    cudaMalloc((void **)&d_r, num_cols * sizeof(type_data));
    cudaMalloc((void **)&d_p, num_cols * sizeof(type_data));
    cudaMalloc((void **)&d_z, num_cols * sizeof(type_data));
    cudaMalloc((void **)&d_Ap, num_cols * sizeof(type_data));
    cudaMalloc((void **)&d_M, num_cols * sizeof(type_data));
    cudaMalloc((void **)&intermediate, num_cols * sizeof(type_data));
    cusparseDnVecDescr_t vec_dp, vec_dAp;
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_dp, num_cols, d_p, CUDA_R_64F) );
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_dAp, num_cols, d_Ap, CUDA_R_64F) );

    CHECK_CUDA(cudaEventRecord(start));  

    // set lower triangular matrix/analysis descriptor

    cusparseSpMatDescr_t descr_L;
    cusparseSpSVDescr_t  spsv_descr_L;
    cusparseDnVecDescr_t vec_dr, vec_dz, vec_intermediate;
    void* buffer_L = NULL;
    size_t buffer_size_L = 0;
    type_data solve_alpha = 1.0;
    CHECK_CUSPARSE( cusparseCreateCsr(&descr_L, num_cols, num_cols, factorization_nnz,
                                      csr_rowptr_device, csr_col_ind_device, csr_val_device,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );
    // Create dense vector x
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_dr, num_cols, d_r, CUDA_R_64F) );
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_dz, num_cols, d_z, CUDA_R_64F) );
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_intermediate, num_cols, intermediate, CUDA_R_64F) );

    // descriptor for analysis data between calls
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsv_descr_L) );
    // Specify Lower|Upper fill mode.
    cusparseFillMode_t fillmode_L = CUSPARSE_FILL_MODE_UPPER;
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(descr_L, CUSPARSE_SPMAT_FILL_MODE,
                                              &fillmode_L, sizeof(fillmode_L)) );
    // Specify Unit|Non-Unit diagonal type.
    cusparseDiagType_t diagtype_L = CUSPARSE_DIAG_TYPE_UNIT;
    CHECK_CUSPARSE( cusparseSpMatSetAttribute(descr_L, CUSPARSE_SPMAT_DIAG_TYPE,
                                              &diagtype_L, sizeof(diagtype_L)) );

     // allocate an external buffer for analysis
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                                cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                &solve_alpha, descr_L, vec_dr, vec_intermediate, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_L,
                                &buffer_size_L) );
    CHECK_CUDA( cudaMalloc(&buffer_L, buffer_size_L) );

    CHECK_CUSPARSE( cusparseSpSV_analysis(
                                cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                                &solve_alpha, descr_L, vec_dr, vec_intermediate, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_L, buffer_L) );
    
    // set upper triangular matrix/analysis descriptor


    cusparseSpSVDescr_t  spsv_descr_U;
    void* buffer_U = NULL;
    size_t buffer_size_U = 0;
    // descriptor for analysis data between calls
    CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsv_descr_U) );
    
     // allocate an external buffer for analysis
    CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &solve_alpha, descr_L, vec_intermediate, vec_dz, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_U,
                                &buffer_size_U) );
    CHECK_CUDA( cudaMalloc(&buffer_U, buffer_size_U) );

    CHECK_CUSPARSE( cusparseSpSV_analysis(
                                cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &solve_alpha, descr_L, vec_intermediate, vec_dz, CUDA_R_64F,
                                CUSPARSE_SPSV_ALG_DEFAULT, spsv_descr_U, buffer_U) );



                        
          
    
    // set up descriptor for mat-vec SPMV
    double minus_one = -1.0;
    double one = 1.0;
    double zero = 0.0;
    cusparseSpMatDescr_t laplacian_des;
    cusparseDnVecDescr_t vec_x;
    void*                buffer_lap    = NULL;
    size_t               buffer_size_lap = 0;
    // Create sparse matrix laplacian in CSR format
    // TODO: consider using preprocess
    CHECK_CUSPARSE( cusparseCreateCsr(&laplacian_des, num_cols, num_cols, nnz,
                                      lap_rowptr_device, lap_col_ind_device, lap_val_device,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );
        
    //  // Specify Lower|Upper fill mode.
    //  fillmode_L = CUSPARSE_FILL_MODE_LOWER;
    //  CHECK_CUSPARSE( cusparseSpMatSetAttribute(laplacian_des, CUSPARSE_SPMAT_FILL_MODE,
    //                                            &fillmode_L, sizeof(fillmode_L)) );


    
    // Create dense vector vec_x
    cudaMemset(leftside_device, 0, num_cols);
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_x, num_cols, leftside_device, CUDA_R_64F) );
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, laplacian_des, vec_x, &one, vec_dr, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size_lap) );
    CHECK_CUDA( cudaMalloc(&buffer_lap, buffer_size_lap) );

    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
     
    printf("solve preprocess time: %f ms\n", milliseconds);
    
   
    
    // generate random left to create random right, CAREFUL: USES SAME BUFFER
    type_data *random_init_left = (type_data *)malloc(num_cols * sizeof(type_data));
    generate_random_vector(random_init_left, num_cols, 0);
    type_data *random_init_left_device;
    cudaMalloc((void **)&random_init_left_device, num_cols * sizeof(type_data));
    cudaMemcpy(random_init_left_device, random_init_left, num_cols * sizeof(type_data), cudaMemcpyHostToDevice);
    cusparseDnVecDescr_t vec_init_left, vec_right;
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_init_left, num_cols, random_init_left_device, CUDA_R_64F) );
    CHECK_CUSPARSE( cusparseCreateDnVec(&vec_right, num_cols, rightside_device, CUDA_R_64F) );
   
    
    
    if(physics)
    {
        // use zero sum vector as rightside
        generate_zero_sum_vector<type_data>(rightside, num_cols, 0);
        cudaMemcpy(rightside_device, rightside, num_cols * sizeof(type_data), cudaMemcpyHostToDevice);
    }
    else
    {
         // use spmv to generate rightside
        cudaMemset(rightside_device, 0, num_cols);
        CHECK_CUSPARSE( cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one, laplacian_des, vec_init_left, &zero, vec_right, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, buffer_lap) );
    }


    /*
    double *righthost = (double *)malloc(num_cols * sizeof(type_data));
    cudaMemcpy(righthost, rightside_device, num_cols * sizeof(type_data), cudaMemcpyDeviceToHost);
    printf("right[0] before: %f\n", righthost[0]);
    */



  
    
    CHECK_CUDA(cudaEventRecord(start));

    // Initialize r = b - Ax
    cudaMemcpy(d_r, rightside_device, num_cols * sizeof(double), cudaMemcpyDeviceToDevice);
    CHECK_CUSPARSE( cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &minus_one, laplacian_des, vec_x, &one, vec_dr, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, buffer_lap) );
    


    // Apply preconditioner: z = M^{-1} * r
    apply_preconditioner(cusparseHandle, cublasHandle, descr_L,
        spsv_descr_L, spsv_descr_U, vec_dz, vec_dr, vec_intermediate,
        diagonal_entries_device, intermediate, num_cols, use_preconditioner);

    

    
 
    
    
    // Initialize p = z
    cudaMemcpy(d_p, d_z, num_cols * sizeof(type_data), cudaMemcpyDeviceToDevice);

    // Compute r1 = dot(r, z)
    cublasDdot(cublasHandle, num_cols, d_r, 1, d_z, 1, &r1);
    int MAX_ITERS = 300;
    type_data TOL = tolerance;
    
    
  
    double exit_tol;
    cublasDnrm2(cublasHandle, num_cols, d_r, 1, &exit_tol);


    
    bool tol_condition = r1 > TOL;
    double initial_r1 = r1;

    while (k < MAX_ITERS && tol_condition) {
        // Ap = A * p

        CHECK_CUSPARSE( cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &one, laplacian_des, vec_dp, &zero, vec_dAp, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, buffer_lap) );
        

        // alpha = r1 / dot(p, Ap)
        cublasDdot(cublasHandle, num_cols, d_p, 1, d_Ap, 1, &cg_alpha);
        cg_alpha = r1 / cg_alpha;

        // x = x + alpha * p
        cublasDaxpy(cublasHandle, num_cols, &cg_alpha, d_p, 1, leftside_device, 1);

        // r = r - alpha * Ap
        double minus_cg_alpha = -cg_alpha;
        cublasDaxpy(cublasHandle, num_cols, &minus_cg_alpha, d_Ap, 1, d_r, 1);

        // Apply preconditioner: z = M * r
        apply_preconditioner(cusparseHandle, cublasHandle, descr_L,
            spsv_descr_L, spsv_descr_U, vec_dz, vec_dr, vec_intermediate,
                diagonal_entries_device, intermediate, num_cols, use_preconditioner);

        // Compute r0 = r1, and then new r1 = dot(r, z)
        r0 = r1;
        cublasDdot(cublasHandle, num_cols, d_r, 1, d_z, 1, &r1);

        //if (r1 < TOL * TOL) break;

        // beta = r1 / r0
        cg_beta = r1 / r0;

        // p = z + beta * p
        cublasDscal(cublasHandle, num_cols, &cg_beta, d_p, 1);
        cublasDaxpy(cublasHandle, num_cols, &one, d_z, 1, d_p, 1);

        cublasDnrm2(cublasHandle, num_cols, d_r, 1, &exit_tol);
        //printf("iterations: %d, residual: %f, r1: %f\n", k, exit_tol, r1);
        k++;
        tol_condition = (sqrt(r1) / sqrt(initial_r1)) > TOL;
        
    }

    printf("final iteration: %d, residual: %f\n", k, r1);
    
    CHECK_CUDA(cudaDeviceSynchronize());  // Ensure kernel execution is finished
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
     
    printf("pcg time: %f ms\n", milliseconds);
    
    CHECK_CUSPARSE( cusparseSpMV(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &one, laplacian_des, vec_x, &zero, vec_init_left, CUDA_R_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, buffer_lap) );
    cublasDaxpy(cublasHandle, num_cols, &minus_one, rightside_device, 1, random_init_left_device, 1);
    double norm_a;
    double norm_b;
    cublasDnrm2(cublasHandle, num_cols, random_init_left_device, 1, &norm_a);
    cublasDnrm2(cublasHandle, num_cols, rightside_device, 1, &norm_b);
    
    /*
    cudaMemcpy(righthost, rightside_device, num_cols * sizeof(type_data), cudaMemcpyDeviceToHost);
    type_data *lefthost = (type_data *)malloc(num_cols * sizeof(type_data));
    cudaMemcpy(lefthost, leftside_device, num_cols * sizeof(type_data), cudaMemcpyDeviceToHost);
    cudaMemcpy(random_init_left, random_init_left_device, num_cols * sizeof(type_data), cudaMemcpyDeviceToHost);
    printf("right[0] after: %f\n", righthost[0]);
    printf("left[0] after: %f\n", lefthost[0]);
    printf("(Ax - b)[0] after: %f\n", random_init_left[0]);
    */
    printf("norm a: %f\n", norm_a);
    printf("norm b: %f\n", norm_b);
    printf("normalized diff norm: %.16f\n", norm_a / norm_b);
   


    
    // TODO: Clean up
    CHECK_CUSPARSE( cusparseDestroySpMat(descr_L) );
    CHECK_CUSPARSE( cusparseDestroyDnVec(vec_dr) );
    CHECK_CUSPARSE( cusparseDestroyDnVec(vec_dz) );
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsv_descr_L));
    CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsv_descr_U));
    CHECK_CUSPARSE( cusparseDestroy(cusparseHandle) );
    CHECK_CUSPARSE( cusparseDestroySpMat(laplacian_des) );
    CHECK_CUSPARSE( cusparseDestroyDnVec(vec_x) );
    CHECK_CUDA( cudaFree(buffer_L) );
    CHECK_CUDA( cudaFree(buffer_U) );
    CHECK_CUDA( cudaFree(buffer_lap) );
    // cusparseDestroySolveAnalysisInfo(info_L);
    // cusparseDestroyMatDescr(descr_L);
    // cusparseDestroySolveAnalysisInfo(info_U);
    // cusparseDestroyMatDescr(descr_U);
    // cusparseDestroy(cusparseHandle);

    return 0;
}

// void pcgSolver(
//     cusparseHandle_t cusparseHandle, cublasHandle_t cublasHandle,
//     const double *d_A, const int *d_csrRowPtrA, const int *d_csrColIndA,
//     double *d_x, const double *d_b, int n, int nnz) {

//     double alpha, beta, r0, r1;
//     int k = 0;

//     // Allocate GPU memory for required vectors
//     double *d_r, *d_p, *d_z, *d_Ap, *d_M;
//     cudaMalloc((void **)&d_r, n * sizeof(double));
//     cudaMalloc((void **)&d_p, n * sizeof(double));
//     cudaMalloc((void **)&d_z, n * sizeof(double));
//     cudaMalloc((void **)&d_Ap, n * sizeof(double));
//     cudaMalloc((void **)&d_M, n * sizeof(double));

//     // Initialize the Jacobi preconditioner
//     int blockSize = 256;
//     int gridSize = (n + blockSize - 1) / blockSize;
//     jacobiPreconditioner<<<gridSize, blockSize>>>(d_M, d_A, d_csrRowPtrA, d_csrColIndA, n);

//     // Initialize r = b - Ax
//     double minus_one = -1.0;
//     double one = 1.0;
//     cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                    n, n, nnz, &minus_one, nullptr, d_A, d_csrRowPtrA, d_csrColIndA,
//                    d_x, &one, d_r);
//     cudaMemcpy(d_r, d_b, n * sizeof(double), cudaMemcpyDeviceToDevice);
//     cublasDaxpy(cublasHandle, n, &minus_one, d_r, 1, d_x, 1);

//     // Apply preconditioner: z = M * r
//     applyPreconditioner<<<gridSize, blockSize>>>(d_z, d_r, d_M, n);

//     // Initialize p = z
//     cudaMemcpy(d_p, d_z, n * sizeof(double), cudaMemcpyDeviceToDevice);

//     // Compute r1 = dot(r, z)
//     cublasDdot(cublasHandle, n, d_r, 1, d_z, 1, &r1);

//     while (k < MAX_ITERS && r1 > TOL * TOL) {
//         // Ap = A * p
//         cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
//                        n, n, nnz, &one, nullptr, d_A, d_csrRowPtrA, d_csrColIndA,
//                        d_p, &zero, d_Ap);

//         // alpha = r1 / dot(p, Ap)
//         cublasDdot(cublasHandle, n, d_p, 1, d_Ap, 1, &alpha);
//         alpha = r1 / alpha;

//         // x = x + alpha * p
//         cublasDaxpy(cublasHandle, n, &alpha, d_p, 1, d_x, 1);

//         // r = r - alpha * Ap
//         double minus_alpha = -alpha;
//         cublasDaxpy(cublasHandle, n, &minus_alpha, d_Ap, 1, d_r, 1);

//         // Apply preconditioner: z = M * r
//         applyPreconditioner<<<gridSize, blockSize>>>(d_z, d_r, d_M, n);

//         // Compute r0 = r1, and then new r1 = dot(r, z)
//         r0 = r1;
//         cublasDdot(cublasHandle, n, d_r, 1, d_z, 1, &r1);

//         if (r1 < TOL * TOL) break;

//         // beta = r1 / r0
//         beta = r1 / r0;

//         // p = z + beta * p
//         cublasDscal(cublasHandle, n, &beta, d_p, 1);
//         cublasDaxpy(cublasHandle, n, &one, d_z, 1, d_p, 1);

//         k++;
//     }

//     // Clean up
//     cudaFree(d_r);
//     cudaFree(d_p);
//     cudaFree(d_z);
//     cudaFree(d_Ap);
//     cudaFree(d_M);
// }


/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
 #include <cublas_v2.h>
 #include <cuda_runtime.h>
 #include <cusparse.h>
 #include <stdio.h>  // fopen
 #include <stdlib.h> // EXIT_FAILURE
 #include <string.h> // strtok
 #include <assert.h>
 #include <fstream>
 #include <chrono>
 #include "fast_matrix_market/fast_matrix_market.hpp"  // Ensure this header is available in your include path
 #include <random>
 
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
         printf("cuSPARSE API failed at line %d with error: %s (%d)\n",         \
                __LINE__, cusparseGetErrorString(status), status);              \
         return EXIT_FAILURE;                                                   \
     }                                                                          \
 }
 
 #define CHECK_CUBLAS(func)                                                     \
 {                                                                              \
     cublasStatus_t status = (func);                                            \
     if (status != CUBLAS_STATUS_SUCCESS) {                                     \
         printf("CUBLAS API failed at line %d with error: %d\n",                \
                __LINE__, status);                                              \
         return EXIT_FAILURE;                                                   \
     }                                                                          \
 }
 
 #if defined(NDEBUG)
 #   define PRINT_INFO(var)
 #else
 #   define PRINT_INFO(var) printf("  " #var ": %f\n", var);
 #endif
 
 typedef struct VecStruct {
     cusparseDnVecDescr_t vec;
     double*              ptr;
 } Vec;
 
 //==============================================================================
 
 /// A 5-point Laplacian on a g x g grid with Dirichlet boundary conditions.
 /// This code allocates. The caller must free.
 void make_laplace_matrix(int * n_out,
                          int **row_offsets_out, 
                          int **columns_out, 
                          double **values_out) {
     int grid = 700; // grid resolution
 
     int n = grid * grid;
     *n_out = n;
     // vertices have 5 neighbors, 
     // but each vertex on the boundary loses 1. corners lose 2.
     int nnz = 5 * n - 4 * grid;
 
     printf("Creating 5-point time-dependent diffusion matrix.\n"
            " grid size: %d x %d\n"
            " matrix rows:   %d\n"
            " matrix cols:   %d\n"
            " nnz:         %d\n",
            grid, grid, n, n, nnz);
 
     int* row_offsets = *row_offsets_out = (int*)malloc((n + 1) * sizeof(int));
     int* columns     = *columns_out     = (int*)malloc(nnz * sizeof(int));
     double* values   = *values_out      = (double*)malloc(nnz * sizeof(double));
     assert(row_offsets);
     assert(columns);
     assert(values);
 
     // The Laplacian stencil looks like [-1;-1,4,-1;-1].
     // ICHOL doesn't work great with that stencil.
     // ICHOL is better suited when there's some more mass on the diagonal.
     double mass = 0.04;
 
     int it = 0; // next unused index into `columns`/`values`
 
 #define INSERT(u,v, x)                    \
     if(0<=(u) && (u)<grid &&              \
        0<=(v) && (v)<grid)                \
     {                                     \
         columns[it] = ((u) * grid + (v)); \
         values[it] = x;                   \
         ++it;                             \
     }
 
     int row = 0;
     row_offsets[row] = 0;
     for (int i = 0; i < grid; ++i) {
         for (int j = 0; j < grid; ++j)
         {
             INSERT(i - 1, j    , -1.0);
             INSERT(i    , j - 1, -1.0);
             INSERT(i    , j    ,  4.0 + mass);
             INSERT(i    , j + 1, -1.0);
             INSERT(i + 1, j    , -1.0);
             row_offsets[++row] = it;
         }
     }
     assert(it == nnz);
 #undef INSERT
 }

 int trim_csr_matrix(int m, int n, int** rows_ptr, int** cols_ptr, double** vals_ptr, int* nnz_ptr) {

    // --- Input Validation ---
    if (m <= 0 || n <= 0 || rows_ptr == NULL || *rows_ptr == NULL ||
        cols_ptr == NULL || *cols_ptr == NULL || vals_ptr == NULL ||
        *vals_ptr == NULL || nnz_ptr == NULL) {
        fprintf(stderr, "Error: Invalid input parameters.\n");
        return -1;
    }

    // --- Handle Edge Cases: Trimming results in an empty matrix ---
    if (m == 1 || n == 1) {
        free(*rows_ptr);
        free(*cols_ptr);
        free(*vals_ptr);
        *rows_ptr = (int*)calloc(1, sizeof(int)); // rows array for 0 rows is {0}
        *cols_ptr = NULL;
        *vals_ptr = NULL;
        *nnz_ptr = 0;
        if (*rows_ptr == NULL) { // Check allocation for the {0} array
             fprintf(stderr, "Error: Memory allocation failed for empty matrix row pointer.\n");
             // Pointers are already NULL or freed, nnz is 0. Signal error.
             return -1;
        }
        // If m=1, new_m=0. If n=1, new_n=0. In both cases, nnz becomes 0.
        return 0; // Success, resulted in an empty matrix
    }

    // --- Calculate New Dimensions ---
    int new_m = m - 1;
    int last_col_index = n - 1; // Column index to remove
    int original_nnz = *nnz_ptr;
    int* original_rows = *rows_ptr;
    int* original_cols = *cols_ptr;
    double* original_vals = *vals_ptr;

    // --- First Pass: Calculate new_nnz and fill new_rows ---
    int* new_rows = (int*)malloc((new_m + 1) * sizeof(int));
    if (new_rows == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for new_rows.\n");
        return -1;
    }

    int new_nnz = 0;
    new_rows[0] = 0;

    for (int i = 0; i < new_m; ++i) { // Iterate through rows to keep (all except the last)
        for (int k = original_rows[i]; k < original_rows[i + 1]; ++k) {
            if (original_cols[k] < last_col_index) { // Check if column is not the last one
                new_nnz++;
            }
        }
        new_rows[i + 1] = new_nnz; // Store the count (which is the start of the next row)
    }

    // --- Allocate new cols and vals arrays ---
    // Handle case where the resulting matrix might have zero non-zeros
    int* new_cols = NULL;
    double* new_vals = NULL;
    if (new_nnz > 0) {
        new_cols = (int*)malloc(new_nnz * sizeof(int));
        new_vals = (double*)malloc(new_nnz * sizeof(double));

        if (new_cols == NULL || new_vals == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for new_cols or new_vals.\n");
            free(new_rows); // Clean up allocated memory
            free(new_cols); // Free even if NULL
            free(new_vals); // Free even if NULL
            // Do NOT free original arrays here, as the caller might still have them.
            // The function fails, leaving original pointers unchanged.
            return -1;
        }
    } else {
         // If new_nnz is 0, new_cols and new_vals should be NULL
         new_cols = NULL;
         new_vals = NULL;
    }


    // --- Second Pass: Fill new_cols and new_vals ---
    int current_nnz_idx = 0;
    for (int i = 0; i < new_m; ++i) { // Iterate through rows to keep
        for (int k = original_rows[i]; k < original_rows[i + 1]; ++k) {
            if (original_cols[k] < last_col_index) { // If column is not the last one
                 if (current_nnz_idx < new_nnz) { // Bounds check (safety)
                    new_cols[current_nnz_idx] = original_cols[k];
                    new_vals[current_nnz_idx] = original_vals[k];
                    current_nnz_idx++;
                 } else {
                    // This should not happen if logic is correct, but indicates an error.
                    fprintf(stderr, "Error: Index out of bounds during fill. Logic error?\n");
                    free(new_rows);
                    free(new_cols);
                    free(new_vals);
                    return -1; // Indicate failure
                 }
            }
        }
    }

     // Defensive check: Did we fill the expected number of elements?
     if (current_nnz_idx != new_nnz) {
          fprintf(stderr, "Error: Mismatch between calculated and filled nnz. Logic error?\n");
          free(new_rows);
          free(new_cols);
          free(new_vals);
          return -1;
     }


    // --- Clean up original arrays ---
    free(original_rows);
    free(original_cols);
    free(original_vals);

    // --- Update caller's pointers and nnz ---
    *rows_ptr = new_rows;
    *cols_ptr = new_cols;
    *vals_ptr = new_vals;
    *nnz_ptr = new_nnz;

    

    return 0; // Success
}


// Function to read Matrix Market file and convert to CSR-style arrays
void read_matrix_market(const char *filename, int **rows, int **cols, double **vals, int *nrows, int *ncols, int *nnz, bool physics) {
    std::ifstream input_stream(filename);
    if (!input_stream.is_open()) {
        fprintf(stderr, "Could not open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    int m, n;
    std::vector<int> coo_rows, coo_cols;
    std::vector<double> coo_vals;

    fast_matrix_market::read_matrix_market_triplet(
        input_stream,
        m,
        n,
        coo_rows,
        coo_cols,
        coo_vals);

    *nrows = m;
    *ncols = n;
    *nnz = static_cast<int>(coo_vals.size());

    *rows = (int *)calloc(m + 1, sizeof(int));
    *cols = (int *)malloc(*nnz * sizeof(int));
    *vals = (double *)malloc(*nnz * sizeof(double));

    // Count non-zeros per row
    for (int i = 0; i < *nnz; ++i) {
        (*rows)[coo_rows[i] + 1]++;
    }

    // Cumulative sum to get CSR row pointers
    for (int i = 0; i < m; ++i) {
        (*rows)[i + 1] += (*rows)[i];
    }

    std::vector<int> row_copy((*rows), (*rows) + m);

    // Fill CSR col and val arrays
    for (int i = 0; i < *nnz; ++i) {
        int row = coo_rows[i];
        int dest = row_copy[row]++;
        (*cols)[dest] = coo_cols[i];
        (*vals)[dest] = coo_vals[i];
    }

    if(physics)
    {
        printf("detect physics problem, trimming: %d\n", *nrows - 1);
        *nrows = *nrows - 1;
        *ncols = *ncols - 1;
        trim_csr_matrix(m, n, rows, cols, vals, nnz);
    }
   
}
 
 //==============================================================================
 
 int gpu_CG(cublasHandle_t       cublasHandle,
            cusparseHandle_t     cusparseHandle,
            int                  m,
            cusparseSpMatDescr_t matA,
            cusparseSpMatDescr_t matL,
            Vec                  d_B,
            Vec                  d_X,
            Vec                  d_R,
            Vec                  d_R_aux,
            Vec                  d_P,
            Vec                  d_T,
            Vec                  d_tmp,
            void*                d_bufferMV,
            int                  maxIterations,
            double               tolerance) {
     const double zero      = 0.0;
     const double one       = 1.0;
     const double minus_one = -1.0;
     double norm_b;
     CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_B.ptr, 1, &norm_b) )
     //--------------------------------------------------------------------------
     // ### 1 ### R0 = b - A * X0 (using initial guess in X)
     //    (a) copy b in R0
     CHECK_CUDA( cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                            cudaMemcpyDeviceToDevice) )
     //    (b) compute R = -A * X0 + R
     CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &minus_one, matA, d_X.vec, &one, d_R.vec,
                                  CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                  d_bufferMV) )
     //--------------------------------------------------------------------------
     // ### 2 ### R_i_aux = L^-1 L^-T R_i
     size_t              bufferSizeL, bufferSizeLT;
     void*               d_bufferL, *d_bufferLT;
     cusparseSpSVDescr_t spsvDescrL, spsvDescrLT;
     //    (a) L^-1 tmp => R_i_aux    (triangular solver)
     CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescrL) )
     CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                         cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                         CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &bufferSizeL) )
     CHECK_CUDA( cudaMalloc(&d_bufferL, bufferSizeL) )
     CHECK_CUSPARSE( cusparseSpSV_analysis(
                         cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                         CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, d_bufferL) )
     CHECK_CUDA( cudaMemset(d_tmp.ptr, 0x0, m * sizeof(double)) )
     CHECK_CUSPARSE( cusparseSpSV_solve(
                         cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &one, matL, d_R.vec, d_tmp.vec, CUDA_R_64F,
                         CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL) )
 
     //    (b) L^-T R_i => tmp    (triangular solver)
     CHECK_CUSPARSE( cusparseSpSV_createDescr(&spsvDescrLT) )
     CHECK_CUSPARSE( cusparseSpSV_bufferSize(
                         cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                         &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                         CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, &bufferSizeLT) )
     CHECK_CUDA( cudaMalloc(&d_bufferLT, bufferSizeLT) )
     CHECK_CUSPARSE( cusparseSpSV_analysis(
                         cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                         &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                         CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT, d_bufferLT) )
     CHECK_CUDA( cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)) )
     CHECK_CUSPARSE( cusparseSpSV_solve(
                         cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE,
                         &one, matL, d_tmp.vec, d_R_aux.vec, CUDA_R_64F,
                         CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrLT) )
     //--------------------------------------------------------------------------
     // ### 3 ### P0 = R0_aux
     CHECK_CUDA( cudaMemcpy(d_P.ptr, d_R_aux.ptr, m * sizeof(double),
                            cudaMemcpyDeviceToDevice) )
     //--------------------------------------------------------------------------
     // nrm_R0 = ||R||
     double nrm_R;
     CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) )
     double threshold = tolerance * nrm_R;
     printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
     //--------------------------------------------------------------------------
     double delta;
     CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1, &delta) )
     //--------------------------------------------------------------------------
     // ### 4 ### repeat until convergence based on max iterations and
     //           and relative residual
     int track_iter = 0;
     for (int i = 0; i < maxIterations; i++) {
         track_iter++;
         //printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
         //----------------------------------------------------------------------
         // ### 5 ### alpha = (R_i, R_aux_i) / (A * P_i, P_i)
         //     (a) T  = A * P_i
         CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                      CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                      matA, d_P.vec, &zero, d_T.vec,
                                      CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                      d_bufferMV) )
         //     (b) denominator = (T, P_i)
         double denominator;
         CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_T.ptr, 1, d_P.ptr, 1,
                                  &denominator) )
         //     (c) alpha = delta / denominator
         double alpha = delta / denominator;
        //  PRINT_INFO(delta)
        //  PRINT_INFO(denominator)
        //  PRINT_INFO(alpha)
         //----------------------------------------------------------------------
         // ### 6 ###  X_i+1 = X_i + alpha * P
         //    (a) X_i+1 = -alpha * T + X_i
         CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &alpha, d_P.ptr, 1,
                                   d_X.ptr, 1) )
         //----------------------------------------------------------------------
         // ### 7 ###  R_i+1 = R_i - alpha * (A * P)
         //    (a) R_i+1 = -alpha * T + R_i
         double minus_alpha = -alpha;
         CHECK_CUBLAS( cublasDaxpy(cublasHandle, m, &minus_alpha, d_T.ptr, 1,
                                   d_R.ptr, 1) )
         //----------------------------------------------------------------------
         // ### 8 ###  check ||R_i+1|| < threshold
         CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) )
        // PRINT_INFO(nrm_R)
         if (nrm_R < threshold)
             break;
         //----------------------------------------------------------------------
         // ### 9 ### R_aux_i+1 = L^-1 L^-T R_i+1
         //    (a) L^-1 R_i+1 => tmp    (triangular solver)
         CHECK_CUDA( cudaMemset(d_tmp.ptr,   0x0, m * sizeof(double)) )
         CHECK_CUDA( cudaMemset(d_R_aux.ptr, 0x0, m * sizeof(double)) )
         CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &one, matL, d_R.vec, d_tmp.vec,
                                            CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsvDescrL) )
         //    (b) L^-T tmp => R_aux_i+1    (triangular solver)
         CHECK_CUSPARSE( cusparseSpSV_solve(cusparseHandle,
                                            CUSPARSE_OPERATION_TRANSPOSE,
                                            &one, matL, d_tmp.vec,
                                            d_R_aux.vec, CUDA_R_64F,
                                            CUSPARSE_SPSV_ALG_DEFAULT,
                                            spsvDescrLT) )
         //----------------------------------------------------------------------
         // ### 10 ### beta = (R_i+1, R_aux_i+1) / (R_i, R_aux_i)
         //    (a) delta_new => (R_i+1, R_aux_i+1)
         double delta_new;
         CHECK_CUBLAS( cublasDdot(cublasHandle, m, d_R.ptr, 1, d_R_aux.ptr, 1,
                                  &delta_new) )
         //    (b) beta => delta_new / delta
         double beta = delta_new / delta;
        // PRINT_INFO(delta_new)
        // PRINT_INFO(beta)
         delta       = delta_new;
         //----------------------------------------------------------------------
         // ### 11 ###  P_i+1 = R_aux_i+1 + beta * P_i
         //    (a) P = beta * P
         CHECK_CUBLAS(cublasDscal(cublasHandle, m, &beta, d_P.ptr, 1))
         //    (b) P = R_aux + P
         CHECK_CUBLAS(
             cublasDaxpy(cublasHandle, m, &one, d_R_aux.ptr, 1, d_P.ptr, 1))
     }
     //--------------------------------------------------------------------------
     printf("took %d iterations\n", track_iter);
     printf("Check Solution\n"); // ||R = b - A * X||
     //    (a) copy b in R
     CHECK_CUDA( cudaMemcpy(d_R.ptr, d_B.ptr, m * sizeof(double),
                            cudaMemcpyDeviceToDevice) )
     // R = -A * X + R
     CHECK_CUSPARSE( cusparseSpMV(cusparseHandle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, &minus_one,
                                  matA, d_X.vec, &one, d_R.vec, CUDA_R_64F,
                                  CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) )
     // check ||R||
     CHECK_CUBLAS( cublasDnrm2(cublasHandle, m, d_R.ptr, 1, &nrm_R) )
     
     printf("Final absolute error = %e\n", nrm_R);
     printf("Final relative error = %e\n", nrm_R / norm_b);
     //--------------------------------------------------------------------------
     CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescrL) )
     CHECK_CUSPARSE( cusparseSpSV_destroyDescr(spsvDescrLT) )
     CHECK_CUDA( cudaFree(d_bufferL) )
     CHECK_CUDA( cudaFree(d_bufferLT) )
     return EXIT_SUCCESS;
 }
 
 //==============================================================================
 //==============================================================================
 
 int main(int argc, char** argv) {
     const int    maxIterations = 10000;
     const double tolerance     = 1e-6;
    //  if (argc != 1) {
    //      printf("Wrong number of command line arguments. cg_example accepts no arguments.\n");
    //      return EXIT_FAILURE;
    //  }
     int     base        = 0;
     int     m           = -1;
     int*    h_A_rows    = NULL;
     int*    h_A_columns = NULL;
     double* h_A_values  = NULL;
     int num_nnz;
     //make_laplace_matrix(&m, &h_A_rows, &h_A_columns, &h_A_values);
     printf("problem name: %s\n", argv[1]);
     read_matrix_market(argv[1], &h_A_rows, &h_A_columns, &h_A_values, &m, &m, &num_nnz, atoi(argv[2]));
     int num_offsets = m + 1;
     int nnz = h_A_rows[m];
     double* h_X = (double*)malloc(m * sizeof(double));
 
     printf("Testing CG\n");
     std::mt19937 gen(0);                     // Mersenne Twister RNG
     std::uniform_real_distribution<double> dist(0.0, 1.0); // Range [-1, 1]
     double vec_sum = 0.0;
     for (int i = 0; i < m; i++)
     {
        h_X[i] = dist(gen);
        vec_sum += h_X[i];
     }
     double mean = vec_sum / m;
     for (int i = 0; i < m; i++) {
        h_X[i] -= mean;
    }
         
     //--------------------------------------------------------------------------
     // ### Device memory management ###
     int*    d_A_rows, *d_A_columns;
     double* d_A_values, *d_L_values;
     Vec     d_B, d_X, d_R, d_R_aux, d_P, d_T, d_tmp;
 
     // allocate device memory for CSR matrices
     CHECK_CUDA( cudaMalloc((void**) &d_A_rows,    num_offsets * sizeof(int)) )
     CHECK_CUDA( cudaMalloc((void**) &d_A_columns, nnz * sizeof(int)) )
     CHECK_CUDA( cudaMalloc((void**) &d_A_values,  nnz * sizeof(double)) )
     CHECK_CUDA( cudaMalloc((void**) &d_L_values,  nnz * sizeof(double)) )
 
     CHECK_CUDA( cudaMalloc((void**) &d_B.ptr,     m * sizeof(double)) )
     CHECK_CUDA( cudaMalloc((void**) &d_X.ptr,     m * sizeof(double)) )
     CHECK_CUDA( cudaMalloc((void**) &d_R.ptr,     m * sizeof(double)) )
     CHECK_CUDA( cudaMalloc((void**) &d_R_aux.ptr, m * sizeof(double)) )
     CHECK_CUDA( cudaMalloc((void**) &d_P.ptr,     m * sizeof(double)) )
     CHECK_CUDA( cudaMalloc((void**) &d_T.ptr,     m * sizeof(double)) )
     CHECK_CUDA( cudaMalloc((void**) &d_tmp.ptr,   m * sizeof(double)) )
 
     // copy the CSR matrices and vectors into device memory
     CHECK_CUDA( cudaMemcpy(d_A_rows, h_A_rows, num_offsets * sizeof(int),
                            cudaMemcpyHostToDevice) )
     CHECK_CUDA( cudaMemcpy(d_A_columns, h_A_columns, nnz *  sizeof(int),
                            cudaMemcpyHostToDevice) )
     CHECK_CUDA( cudaMemcpy(d_A_values, h_A_values, nnz * sizeof(double),
                            cudaMemcpyHostToDevice) )
     CHECK_CUDA( cudaMemcpy(d_L_values, h_A_values, nnz * sizeof(double),
                            cudaMemcpyHostToDevice) )
     CHECK_CUDA( cudaMemcpy(d_X.ptr, h_X, m * sizeof(double),
                            cudaMemcpyHostToDevice) )
     //--------------------------------------------------------------------------
     // ### cuSPARSE Handle and descriptors initialization ###
     // create the test matrix on the host
     cublasHandle_t   cublasHandle   = NULL;
     cusparseHandle_t cusparseHandle = NULL;
     CHECK_CUBLAS( cublasCreate(&cublasHandle) )
     CHECK_CUSPARSE( cusparseCreate(&cusparseHandle) )
     // Create dense vectors
     CHECK_CUSPARSE( cusparseCreateDnVec(&d_B.vec,     m, d_B.ptr, CUDA_R_64F) )
     CHECK_CUSPARSE( cusparseCreateDnVec(&d_X.vec,     m, d_X.ptr, CUDA_R_64F) )
     CHECK_CUSPARSE( cusparseCreateDnVec(&d_R.vec,     m, d_R.ptr, CUDA_R_64F) )
     CHECK_CUSPARSE( cusparseCreateDnVec(&d_R_aux.vec, m, d_R_aux.ptr,
                                         CUDA_R_64F) )
     CHECK_CUSPARSE( cusparseCreateDnVec(&d_P.vec,   m, d_P.ptr,   CUDA_R_64F) )
     CHECK_CUSPARSE( cusparseCreateDnVec(&d_T.vec,   m, d_T.ptr,   CUDA_R_64F) )
     CHECK_CUSPARSE( cusparseCreateDnVec(&d_tmp.vec, m, d_tmp.ptr, CUDA_R_64F) )
 
     cusparseIndexBase_t  baseIdx = CUSPARSE_INDEX_BASE_ZERO;
     cusparseSpMatDescr_t matA, matL;
     int*                 d_L_rows      = d_A_rows;
     int*                 d_L_columns   = d_A_columns;
     cusparseFillMode_t   fill_lower    = CUSPARSE_FILL_MODE_LOWER;
     cusparseDiagType_t   diag_non_unit = CUSPARSE_DIAG_TYPE_NON_UNIT;
     // A
     CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, m, nnz, d_A_rows,
                                       d_A_columns, d_A_values,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       baseIdx, CUDA_R_64F) )
     // L
     CHECK_CUSPARSE( cusparseCreateCsr(&matL, m, m, nnz, d_L_rows,
                                       d_L_columns, d_L_values,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       baseIdx, CUDA_R_64F) )
     CHECK_CUSPARSE( cusparseSpMatSetAttribute(matL,
                                               CUSPARSE_SPMAT_FILL_MODE,
                                               &fill_lower, sizeof(fill_lower)) )
     CHECK_CUSPARSE( cusparseSpMatSetAttribute(matL,
                                               CUSPARSE_SPMAT_DIAG_TYPE,
                                               &diag_non_unit,
                                               sizeof(diag_non_unit)) )
     //--------------------------------------------------------------------------
     // ### Preparation ### b = A * X
     const double alpha = 0.75;
     size_t       bufferSizeMV;
     void*        d_bufferMV;
     double       beta = 0.0;
     CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                         cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
                         CUSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV) )
     CHECK_CUDA( cudaMalloc(&d_bufferMV, bufferSizeMV) )
 
     CHECK_CUSPARSE( cusparseSpMV(
                         cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, matA, d_X.vec, &beta, d_B.vec, CUDA_R_64F,
                         CUSPARSE_SPMV_ALG_DEFAULT, d_bufferMV) )

     // overwrite d_B_vec with custom rhs
     std::vector<double> custom_rhs(m, 0.0);
     vec_sum = 0.0;
     for (int i = 0; i < m; i++)
     {
        custom_rhs[i] = dist(gen);
        vec_sum += custom_rhs[i];
     }
     mean = vec_sum / m;
     for (int i = 0; i < m; i++) {
        custom_rhs[i] -= mean;
    }
    CHECK_CUDA( cudaMemcpy(d_B.ptr, custom_rhs.data(), m * sizeof(double),
    cudaMemcpyHostToDevice) )

     // X0 = 0
     CHECK_CUDA( cudaMemset(d_X.ptr, 0x0, m * sizeof(double)) )
     //--------------------------------------------------------------------------
     // Perform Incomplete-Cholesky factorization of A (csric0) -> L, L^T
     cusparseMatDescr_t descrM;
     csric02Info_t      infoM        = NULL;
     int                bufferSizeIC = 0;
     void*              d_bufferIC;
     CHECK_CUSPARSE( cusparseCreateMatDescr(&descrM) )
     CHECK_CUSPARSE( cusparseSetMatIndexBase(descrM, baseIdx) )
     CHECK_CUSPARSE( cusparseSetMatType(descrM, CUSPARSE_MATRIX_TYPE_GENERAL) )
     CHECK_CUSPARSE( cusparseSetMatFillMode(descrM, CUSPARSE_FILL_MODE_LOWER) )
     CHECK_CUSPARSE( cusparseSetMatDiagType(descrM,
                                            CUSPARSE_DIAG_TYPE_NON_UNIT) )
     CHECK_CUSPARSE( cusparseCreateCsric02Info(&infoM) )
 
     CHECK_CUSPARSE( cusparseDcsric02_bufferSize(
                         cusparseHandle, m, nnz, descrM, d_L_values,
                         d_A_rows, d_A_columns, infoM, &bufferSizeIC) )
     CHECK_CUDA( cudaMalloc(&d_bufferIC, bufferSizeIC) )

     cudaEvent_t start, stop;
     float milliseconds = 0;
     
     CHECK_CUDA(cudaEventCreate(&start));
     CHECK_CUDA(cudaEventCreate(&stop));
     
     CHECK_CUDA(cudaEventRecord(start));
     CHECK_CUSPARSE( cusparseDcsric02_analysis(
                         cusparseHandle, m, nnz, descrM, d_L_values,
                         d_A_rows, d_A_columns, infoM,
                         CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferIC) )

    //CHECK_CUDA(cudaDeviceSynchronize());  // Ensure kernel execution is finished
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));   
    printf("analysis time: %f ms\n", milliseconds);

     int structural_zero;
     CHECK_CUSPARSE( cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
                                                &structural_zero) )
                                                

    
    // Incomplete Cholesky factorization call
    // M = L * L^T
    // CHECK_CUSPARSE( cusparseDcsric02(
    //     cusparseHandle, m, nnz, descrM, d_L_values,
    //     d_A_rows, d_A_columns, infoM,
    //     CUSPARSE_SOLVE_POLICY_NO_LEVEL, d_bufferIC) )
    CHECK_CUDA(cudaEventRecord(start));
        
    CHECK_CUSPARSE( cusparseDcsric02(
        cusparseHandle, m, nnz, descrM, d_L_values,
        d_A_rows, d_A_columns, infoM,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferIC) )
        
    
    //CHECK_CUDA(cudaDeviceSynchronize());  // Ensure kernel execution is finished
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
     
    printf("Incomplete Cholesky time: %f ms\n", milliseconds);
    
     // Find numerical zero
     int numerical_zero;
     CHECK_CUSPARSE( cusparseXcsric02_zeroPivot(cusparseHandle, infoM,
                                                &numerical_zero) )
    
     CHECK_CUSPARSE( cusparseDestroyCsric02Info(infoM) )
     CHECK_CUSPARSE( cusparseDestroyMatDescr(descrM) )
     CHECK_CUDA( cudaFree(d_bufferIC) )
     //--------------------------------------------------------------------------
     // ### Run CG computation ###
     printf("CG loop:\n");
     auto begin = std::chrono::high_resolution_clock::now();
     gpu_CG(cublasHandle, cusparseHandle, m,
            matA, matL, d_B, d_X, d_R, d_R_aux, d_P, d_T,
            d_tmp, d_bufferMV, maxIterations, tolerance);
     auto end = std::chrono::high_resolution_clock::now();

     // Calculate the duration in seconds
     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);

     std::cout << "solve time taken: " << duration.count() << " milliseconds" << std::endl;
     printf("\n\n\n\n\n\n\n\n");
     //--------------------------------------------------------------------------
     // ### Free resources ###
     CHECK_CUSPARSE( cusparseDestroyDnVec(d_B.vec) )
     CHECK_CUSPARSE( cusparseDestroyDnVec(d_X.vec) )
     CHECK_CUSPARSE( cusparseDestroyDnVec(d_R.vec) )
     CHECK_CUSPARSE( cusparseDestroyDnVec(d_R_aux.vec) )
     CHECK_CUSPARSE( cusparseDestroyDnVec(d_P.vec) )
     CHECK_CUSPARSE( cusparseDestroyDnVec(d_T.vec) )
     CHECK_CUSPARSE( cusparseDestroyDnVec(d_tmp.vec) )
     CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
     CHECK_CUSPARSE( cusparseDestroySpMat(matL) )
     CHECK_CUSPARSE( cusparseDestroy(cusparseHandle) )
     CHECK_CUBLAS( cublasDestroy(cublasHandle) )
 
     free(h_A_rows);
     free(h_A_columns);
     free(h_A_values);
     free(h_X);
 
     CHECK_CUDA( cudaFree(d_X.ptr) )
     CHECK_CUDA( cudaFree(d_B.ptr) )
     CHECK_CUDA( cudaFree(d_R.ptr) )
     CHECK_CUDA( cudaFree(d_R_aux.ptr) )
     CHECK_CUDA( cudaFree(d_P.ptr) )
     CHECK_CUDA( cudaFree(d_T.ptr) )
     CHECK_CUDA( cudaFree(d_tmp.ptr) )
     CHECK_CUDA( cudaFree(d_A_values) )
     CHECK_CUDA( cudaFree(d_A_columns) )
     CHECK_CUDA( cudaFree(d_A_rows) )
     CHECK_CUDA( cudaFree(d_L_values) )
     CHECK_CUDA( cudaFree(d_bufferMV) )
     return EXIT_SUCCESS;
 }

 // nvcc incomplete_chol.cu -I/pscratch/sd/t/tianyul/fast_matrix_market/include/ -L/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/math_libs/12.4/targets/x86_64-linux/lib/ -lcusparse -lcublas
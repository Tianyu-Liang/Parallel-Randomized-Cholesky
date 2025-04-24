
#include <iostream>
#include <stack>
#include <fstream>
#include <omp.h> 
#include "fast_matrix_market/fast_matrix_market.hpp"
#include <cassert>








template <typename type_int, typename type_data>
struct triplet_matrix {
    type_int nrows = 0, ncols = 0;
    std::vector<type_int> rows, cols;
    std::vector<type_data> vals;       // or int64_t, float, std::complex<double>, etc.
};

// Sparse Matrix Structure in CSC format
template <typename type_int, typename type_data>
struct sparse_matrix {
    type_int num_rows;                   // Number of rows in the matrix
    type_int num_cols;                   // Number of columns in the matrix
    std::vector<type_data> values;          // Non-zero values
    std::vector<type_int> row_indices; // Row indices for each non-zero value
    std::vector<type_int> col_ptrs;    // Column pointers
    std::vector<type_int> lower_tri_cptr_start; // used for accessing only lower triangular part
    std::vector<type_int> lower_tri_cptr_end; // used for accessing only lower triangular part

    // Constructor to initialize the matrix dimensions and column pointers
    sparse_matrix(type_int numRows, type_int numCols) : num_rows(numRows), num_cols(numCols) {
        col_ptrs.resize(num_cols + 1, 0);
    }

     // default constructor
    sparse_matrix() : num_rows(0), num_cols(0) {
        col_ptrs.resize(1, 0);
    }

    type_int rows() const 
    {
        return num_rows;
    }

    type_int cols() const 
    {
        return num_cols;
    }

    type_int outerSize() const
    {
        return num_cols;
    }

    size_t nonZeros()
    {
        return col_ptrs[num_cols];
    }
    

    // Method to add a non-zero element
    void add_value(type_int row, type_int col, type_data value) {
        if (col >= num_cols || row >= num_rows) {
            std::cerr << "Error: Index out of bounds" << std::endl;
            return;
        }

        // Find the position to insert the new element
        type_int insertPos = col_ptrs[col + 1];

        // Update the values and row indices
        values.insert(values.begin() + insertPos, value);
        row_indices.insert(row_indices.begin() + insertPos, row);

        // Update column pointers for subsequent columns
        for (type_int i = col + 1; i <= num_cols; ++i) {
            col_ptrs[i]++;
        }
    }

    // Iterator class to iterate over non-zero values
    // Inner iterator class to iterate over non-zero values of a specific column
    class InnerIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = type_data;
        using pointer           = value_type*;
        using reference         = value_type&;

        InnerIterator(const sparse_matrix<type_int, type_data>& matrix, type_int column)
            : m_values(&matrix.values), m_row_indices(&matrix.row_indices),
              m_index(matrix.col_ptrs[column]), m_end(matrix.col_ptrs[column + 1]), m_column(column) {}

        value_type value() const { return (*m_values)[m_index]; }
        type_int row() const { return (*m_row_indices)[m_index]; }
        type_int col() const { return m_column; }

        // Prefix increment
        InnerIterator& operator++() { 
            ++m_index;
            return *this;
        }

        // Postfix increment
        InnerIterator operator++(int) { 
            InnerIterator tmp = *this; 
            ++(*this); 
            return tmp; 
        }

        friend bool operator==(const InnerIterator& a, const InnerIterator& b) { return a.m_index == b.m_index; }
        friend bool operator!=(const InnerIterator& a, const InnerIterator& b) { return a.m_index != b.m_index; }

        bool isValid() const { return m_index < m_end; }

        // Custom operator to support logical negation
        bool operator!() const { return m_index >= m_end; }

        // Conversion operator to bool for iterator validity
        operator bool() const { return isValid(); }

    private:
        const std::vector<type_data>* m_values;
        const std::vector<type_int>* m_row_indices;
        type_int m_index;
        type_int m_end;
        type_int m_column;
    };

    // Begin and end functions for the iterator
    InnerIterator begin() {
        return InnerIterator(values.data(), row_indices.begin(), col_ptrs.begin(), values.data(), col_ptrs.begin(), col_ptrs.end());
    }

    InnerIterator end() {
        return InnerIterator(values.data() + values.size(), row_indices.end(), col_ptrs.end() - 1, values.data(), col_ptrs.begin(), col_ptrs.end());
    }

    // Method to print the matrix in CSC format
    void printCSC() const {
        std::cout << "Values: ";
        for (const auto& val : values) {
            std::cout << val << " ";
        }
        std::cout << "\nRow Indices: ";
        for (const auto& idx : row_indices) {
            std::cout << idx << " ";
        }
        std::cout << "\nColumn Pointers: ";
        for (const auto& ptr : col_ptrs) {
            std::cout << ptr << " ";
        }
        std::cout << std::endl;
    }
};



template <typename type_int, typename type_data>
class sparse_matrix_processor {

    private:
    // Function to check that the triplets are sorted by column, then by row
    void assert_sorted_triplets(const std::vector<type_int>& rows,
                            const std::vector<type_int>& cols) 
    {
        for (size_t i = 1; i < rows.size(); ++i) {
            if(!((cols[i-1] < cols[i]) ||
                (cols[i-1] == cols[i] && rows[i-1] <= rows[i])))
                {
                    std::cout << "rows[i-1]: " << rows[i - 1] << "\n";
                    std::cout << "rows[i]: " << rows[i] << "\n";
                     std::cout << "cols[i-1]: " << cols[i - 1] << "\n";
                    std::cout << "cols[i]: " << cols[i] << "\n";
                    std::cout << "i: " << i << "\n";
                }

            assert((cols[i-1] < cols[i]) ||
                (cols[i-1] == cols[i] && rows[i-1] <= rows[i]));
        }
    }

    // Function to sort row, column, and value vectors by column, then by row
    void sort_triplets_append_diagonal(std::vector<type_int>& rows,
                    std::vector<type_int>& cols,
                    std::vector<double>& values, size_t num_cols, bool add_diagonal) 
    {   

        size_t original_cols_size = cols.size();

        // make off diagonal entries negative
         for(size_t i = 0; i < values.size(); ++i)
        {
            if(rows[i] != cols[i])
            {
                values[i] = -fabs(values[i]);
            }
            else
            {
                values[i] = fabs(values[i]);
            }
            
        }

        // add the diagonal entries to the end, will be in correct location after sort
        if(add_diagonal)
        {
            for(size_t i = 0; i < num_cols; i++)
            {
                rows.push_back(i);
                cols.push_back(i);
                values.push_back(0.0);
            }

            // calculate diagonal entries by adding up off diagonals
            for(size_t i = 0; i < original_cols_size; ++i)
            {
                if(cols[i] == rows[i])
                {
                    printf("input already has diagonals added!!! No need to manually append diagonals!!!\n");
                    assert(cols[i] != rows[i]);
                }
                values[original_cols_size + cols[i]] += fabs(values[i]);
            }
        }
        
        
        // Create an index vector to store the initial indices
        std::vector<size_t> indices(rows.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            indices[i] = i;
        }


        // Sort the indices based on the column and row values
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return (cols[a] < cols[b]) || (cols[a] == cols[b] && rows[a] < rows[b]);
        });

        // Create temporary vectors to store sorted data
        std::vector<type_int> sorted_rows(rows.size());
        std::vector<type_int> sorted_cols(cols.size());
        std::vector<double> sorted_values(values.size());

        // Rearrange the original vectors based on sorted indices
        for (size_t i = 0; i < indices.size(); ++i) {
            sorted_rows[i] = rows[indices[i]];
            sorted_cols[i] = cols[indices[i]];
            sorted_values[i] = values[indices[i]];
        }

        

        // Assign sorted vectors back to original vectors
        rows = std::move(sorted_rows);
        cols = std::move(sorted_cols);
        values = std::move(sorted_values);
    }


    // Function to convert separate row, column, and value vectors to CSC format

    sparse_matrix<type_int, type_data> triplet_to_csc_with_diagonal(type_int numRows, type_int numCols,
                            const std::vector<type_int>& rows,
                            const std::vector<type_int>& cols,
                            const std::vector<type_data>& values) 
    {
        sparse_matrix<type_int, type_data> cscMatrix(numRows, numCols);

        // Step 1: Count the number of entries in each column
        for (const auto& col : cols) {
            cscMatrix.col_ptrs[col + 1]++;
        }

        // Step 2: Compute the cumulative sum to get the column pointers
        for (type_int col = 0; col < numCols; ++col) {
            cscMatrix.col_ptrs[col + 1] += cscMatrix.col_ptrs[col];
        }

        // Step 3: Fill the values and row_indices arrays
        cscMatrix.values.resize(values.size());
        cscMatrix.row_indices.resize(rows.size());

        for (size_t i = 0; i < values.size(); ++i) {
            type_int col = cols[i];
            type_int destPos = cscMatrix.col_ptrs[col];

            cscMatrix.values[destPos] = values[i];
            cscMatrix.row_indices[destPos] = rows[i];

            cscMatrix.col_ptrs[col]++;
        }

        // Step 4: Reset column pointers
        for (int64_t col = numCols; col > 0; --col) {
            cscMatrix.col_ptrs[col] = cscMatrix.col_ptrs[col - 1];
        }
        cscMatrix.col_ptrs[0] = 0;

        return cscMatrix;
    }

    public:

        
        std::string name; // Member variable
        sparse_matrix<type_int, type_data> mat; // Define a sparse matrix
        std::vector<type_int> etree;
        std::vector<std::vector<type_int>> ftree;
        std::vector<type_int> layer_summary;
        std::vector<type_int> subtree_node_count;
        std::vector<type_int> min_dependency_count;
      

    
        // Constructor
        sparse_matrix_processor(const std::string& name) : name(name)
        {
            

            // Load the matrix from file
            std::string path = "../data/";
            std::ifstream input_stream(path + name);
            triplet_matrix<type_int, type_data> input_triplet;
            input_triplet.nrows = 28924;
            input_triplet.ncols = 28924;
            // read input into triplet
            fast_matrix_market::read_matrix_market_triplet(
                input_stream, input_triplet.nrows, input_triplet.ncols, input_triplet.rows, input_triplet.cols, input_triplet.vals);
          
            //std::cout << "row size: " << input_triplet.rows.size() << "\n";
            // Sort the triplets
            bool add_diagonal = true;
            for(type_int i = 0; i < input_triplet.rows.size(); i++)
            {
                if(input_triplet.rows[i] == input_triplet.cols[i])
                {
                    add_diagonal = false;
                }
            }
            
            sort_triplets_append_diagonal(input_triplet.rows, input_triplet.cols, input_triplet.vals, input_triplet.ncols, add_diagonal);


            // Assert that triplets are sorted
            assert_sorted_triplets(input_triplet.rows, input_triplet.cols);

            // make sure it's a square matrix (i.e. legitimate graph)
            

            mat = triplet_to_csc_with_diagonal(input_triplet.nrows, input_triplet.ncols, input_triplet.rows, input_triplet.cols, input_triplet.vals);

            assert(mat.rows() == mat.cols());

        
            // Print the matrix
            //std::cout << "Matrix:\n" << mat << "\n";

            type_int n = mat.rows();
            subtree_node_count.resize(n + 1, 0);
            std::cout << "number of nodes: " << n << "\n";
            std::cout << "number of nonzeros: " << mat.nonZeros() << "\n";
            

            // compute elimination tree
            auto start = std::chrono::high_resolution_clock::now();
            etree = build_elimination_tree(mat);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            std::cout << "Time taken to build etree: " << duration.count() << " seconds" << std::endl;

            // start = std::chrono::high_resolution_clock::now();
            // etree = build_elimination_tree_slow(mat);
            // end = std::chrono::high_resolution_clock::now();
            // duration = end - start;
            // std::cout << "Time taken to build etree (slow): " << duration.count() << " seconds" << std::endl;


           // writeVectorToFile(etree, "etree.txt");

            // compute factorization tree
            start = std::chrono::high_resolution_clock::now();
            ftree = create_factorization_tree_from_etree(etree);
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            std::cout << "Time taken to build factorization tree: " << duration.count() << " seconds" << std::endl;
           // writeVectorOfVectorsToFile(ftree, "ftree.txt");

            // generate layer element summary
            // also gets subtree node count, has 1 extra space due to the dummy node at top of tree
            start = std::chrono::high_resolution_clock::now();
            layer_summary = layer_information(ftree);
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            std::cout << "Time taken to generate summary: " << duration.count() << " seconds" << std::endl;
         
            //writeVectorToFile(layer_summary, "layer_summary.txt");

            // calculate minimum dependency
            min_dependency_count.resize(n, 0);
            count_minimum_dependencies(mat);

             // print out summary
            std::cout << "depth of tree (has 1 extra depth due to placeholder): " << layer_summary.size() << "\n";
            type_int layer_sum = 0;
            for (auto it = layer_summary.begin(); it != layer_summary.end(); ++it) {
                layer_sum += *it;
            }
            assert(layer_sum - 1 == mat.rows());
            
            
        }
        

       
        void writeVectorToFile(const std::vector<type_int>& vec, const std::string& filename) {
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

        // Function to write a vector of vectors to a file

        void writeVectorOfVectorsToFile(const std::vector<std::vector<type_int>>& vecOfVecs, const std::string& filename) {
            std::ofstream outFile(filename);
            
            if (!outFile) {
                std::cerr << "Error opening file: " << filename << std::endl;
                return;
            }
            
            for (const auto& vec : vecOfVecs) {
                for (const auto& element : vec) {
                    outFile << element << " ";
                }
                outFile << "\n"; // New line after each inner vector
            }
            
            outFile.close();
        }

        // Function to print a single vector

        void printVector(const std::vector<type_int>& vec) {
            for (const type_int& element : vec) {
                std::cout << element << " ";
            }
            std::cout << std::endl;
        }

        // Function to print a list of lists (vector of vectors)

        void printListOfLists(const std::vector<std::vector<type_int>>& listOfLists) {
            for (const auto& list : listOfLists) {
                printVector(list);
            }
        }


        void count_minimum_dependencies(const sparse_matrix<type_int, type_data>& matrix) 
        {

            for (type_int k = 0; k < matrix.outerSize(); ++k) 
            {
                //std::cout << "k: " << k << "\n";
                typename sparse_matrix<type_int, type_data>::InnerIterator it(matrix, k);
                if (!it || it.row() >= k)
                {
                    continue;
                }

                
                for (; it && it.row() < k; ++it) 
                {
                    
                   min_dependency_count[k]++;
                    
                }

               
            }
        }




        std::vector<type_int> build_elimination_tree(const sparse_matrix<type_int, type_data>& matrix) 
        {
            /* do this inductively, everytime a new node is introduced, find the nodes with the smaller indices that connect to it, 
            then track down those paths and insert the new node in. This inductively creates the step n graph. */
            std::vector<type_int> tree(matrix.rows(), 0);
            
            type_int tree_size = tree.size();
            std::vector<int64_t> list_location(tree_size, -1);
            std::vector<int64_t> path_connection;
            path_connection.reserve(tree_size);
            std::vector<int64_t> dynamic_path_connection;
            dynamic_path_connection.reserve(tree_size);
            std::vector<std::vector<type_int>> path_list;
            path_list.reserve(tree_size);
            std::vector<type_int> actual_location;
            actual_location.reserve(1000);

            for (type_int k = 0; k < matrix.outerSize(); ++k) 
            {
                //std::cout << "k: " << k << "\n";
                typename sparse_matrix<type_int, type_data>::InnerIterator it(matrix, k);
                if (!it || it.row() >= k)
                {
                    continue;
                }

                
                for (; it && it.row() < k; ++it) 
                {
                    
                    if (list_location[it.row()] == -1)
                    {
                        std::vector<type_int> temp;
                        temp.reserve(100);
                        temp.push_back(it.row());
                        path_list.push_back(temp);
                        type_int idx = path_list.size() - 1;
                        path_connection.push_back(idx);
                        dynamic_path_connection.push_back(idx);
                        list_location[it.row()] = idx;
                    }
                
                    type_int idx = list_location[it.row()];
                    while(dynamic_path_connection[idx] != idx)
                    {
                        idx = dynamic_path_connection[idx];
                    }
                    actual_location.push_back(idx);


                    // update the dynamic connection so it's connecting to the newest
                    type_int latest_path_idx = idx;
                    idx = list_location[it.row()];
                    while(dynamic_path_connection[idx] != idx)
                    {
                        
                        type_int temp = dynamic_path_connection[idx];
                        dynamic_path_connection[idx] = latest_path_idx;
                        idx = temp;
                    }

                    
                }

                bool same_path = true;
                type_int reference = actual_location[0];
                for(type_int i = 1; i < actual_location.size() && same_path; i++)
                {
                    same_path = (reference == actual_location[i]);
                }

                if (!same_path)
                {
                    std::vector<type_int> temp;
                    temp.reserve(100);
                    temp.push_back(k);
                    path_list.push_back(temp);
                    type_int idx = path_list.size() - 1;
                    path_connection.push_back(idx);
                    dynamic_path_connection.push_back(idx);
                    list_location[k] = idx;
            

                    for(type_int i = 0; i < actual_location.size(); i++)
                    {
                        path_connection[actual_location[i]] = idx;
                        dynamic_path_connection[actual_location[i]] = idx;
                    }
                }
                else
                {
                    path_list[reference].push_back(k);
                    list_location[k] = reference;
                }
            


                actual_location.clear();
            }

            for(type_int i = 0; i < path_list.size(); i++)
            {
                std::vector<type_int> &cur_list = path_list[i];
                for(type_int j = 0; j < cur_list.size() - 1; j++)
                {
                    tree[cur_list[j]] = cur_list[j + 1];
                }
                // if the index of the next path is yourself, then that means it has reached the root
                if(i != path_connection[i])
                    tree[cur_list[cur_list.size() - 1]] = path_list[path_connection[i]][0];
                else
                    tree[cur_list[cur_list.size() - 1]] = 0;
            }


            std::cout << "num linear paths: " << path_list.size() << "\n";
            double sum = 0;
            for(int i = 0; i < path_list.size(); i++)
            {
                sum += path_list[i].size();
            }
            std::cout << "average per path: " << sum / path_list.size() << "\n";

            return tree;

        }


        std::vector<type_int> build_elimination_tree_slow(const sparse_matrix<type_int, type_data>& matrix) 
        {
            /* do this inductively, everytime a new node is introduced, find the nodes with the smaller indices that connect to it, 
            then track down those paths and insert the new node in. This inductively creates the step n graph. */
            
            std::vector<type_int> tree(matrix.rows(), 0);


            for (type_int k = 0; k < matrix.outerSize(); ++k) 
            {
                //std::cout << "k: " << k << "\n";
                for (typename sparse_matrix<type_int, type_data>::InnerIterator it(matrix, k); it; ++it) 
                {
                    type_int col = it.col();
                    type_int pt = it.row();
                    
                    if(pt < col)
                    {

                        while(tree[pt] != 0 && tree[pt] != col){
                            pt = tree[pt];
                        }
                        tree[pt] = col;
                        
                    }
                
                }
        
            }

            return tree;

        }



        // index 0 of factorization tree is the root node location
        std::vector<std::vector<type_int>> create_factorization_tree_from_etree(const std::vector<type_int> &etree) 
        {
            std::vector<std::vector<type_int>> factorization_tree(etree.size() + 1);
            for(type_int i = 0; i < etree.size(); i++)
            {
                factorization_tree[etree[i]].push_back(i);
            }
            factorization_tree[factorization_tree.size() - 1] = factorization_tree[0];
            factorization_tree[0].clear();

            return factorization_tree;

        }

        /* return a vector where the index corresponds to the depth and the value correspond to the number 
        of elements at the depth */
        std::vector<type_int> layer_information(const std::vector<std::vector<type_int>> &ftree)
        {
            std::vector<type_int> nodes_per_layer;
            std::stack<type_int> stack;
            std::stack<type_int> depth;
            std::stack<type_int> post_order;

            // do it iteratively to avoid stack overflow
            // depth 0 is a PLACE HOLDER NODE
            stack.push(ftree.size() - 1);
            depth.push(0);

            while(!stack.empty())
            {

                type_int index = stack.top();
                type_int cur_depth = depth.top();
                post_order.push(index);
                stack.pop();
                depth.pop();
                
                // update count at depth
                if(cur_depth >= nodes_per_layer.size())
                {
                    nodes_per_layer.push_back(1);
                }
                else
                {
                    nodes_per_layer[cur_depth] += 1;
                }

                const std::vector<type_int> &child = ftree[index];
                
                for (auto it = child.begin(); it != child.end(); ++it) 
                {
                    stack.push(*it);
                    depth.push(cur_depth + 1);
                }
            
            }

            // use post order to calculate the total number of nodes in each subtree
            while(!post_order.empty())
            {
                type_int index = post_order.top();
                post_order.pop();
                type_int node_sum = 1;

                // go through children nodes and sum the subtrees
                const std::vector<type_int> &child = ftree[index];
                for (auto it = child.begin(); it != child.end(); ++it) 
                {
                    node_sum += subtree_node_count[*it];
                }
                subtree_node_count[index] = node_sum;
            }

            return nodes_per_layer;

        }

        // create a slice to easily access lower triangular
        void insert_lower_triangular_info(sparse_matrix<type_int, type_data> &matrix)
        {
  
            std::vector<type_int> col_ptrs_begin(matrix.num_cols, 0);    // Column pointers
            std::vector<type_int> col_ptrs_end(matrix.num_cols, 0);



            for (type_int k = 0; k < matrix.outerSize(); ++k) 
            {
  
                typename sparse_matrix<type_int, type_data>::InnerIterator it(matrix, k);
                // if empty column
                if (!it)
                {
                    

                    continue;
                }

                // count number of lower triangular component
                type_int count = 0;
                for (; it; ++it) 
                {
                    
                    if(it.row() <= k)
                    {
                        count++;
                    }
                    
                }
                col_ptrs_begin[k] = matrix.col_ptrs[k] + count;
                col_ptrs_end[k] = matrix.col_ptrs[k + 1];
            }

            matrix.lower_tri_cptr_start = std::move(col_ptrs_begin);
            matrix.lower_tri_cptr_end = std::move(col_ptrs_end);
        }


        // create a lower triangular matrix out of the original input, doesn't preserve diagonal
        sparse_matrix<type_int, type_data> make_lower_triangular(const sparse_matrix<type_int, type_data> &matrix)
        {
            type_int num_rows = matrix.num_rows;                   // Number of rows in the matrix
            type_int num_cols = matrix.num_cols;                   // Number of columns in the matrix
            std::vector<type_data> values;          // Non-zero values
            std::vector<type_int> row_indices; // Row indices for each non-zero value
            std::vector<type_int> col_ptrs(matrix.num_cols + 1, 0);    // Column pointers
            sparse_matrix<type_int, type_data> ret;

            int announce = 0;
            for (type_int k = 0; k < matrix.outerSize(); ++k) 
            {
  
                typename sparse_matrix<type_int, type_data>::InnerIterator it(matrix, k);
                // if empty column
                if (!it)
                {
                    col_ptrs[k + 1] = col_ptrs[k];
                    continue;
                }

                // count number of lower triangular component
                type_int count = 0;
                for (; it; ++it) 
                {
                    
                    if(it.row() > k)
                    {
                        count++;
                    }
                    
                }
                col_ptrs[k + 1] = col_ptrs[k] + count;
            }

            values.resize(col_ptrs[num_cols], 0.0);
            row_indices.resize(col_ptrs[num_cols], 0);

            for (type_int k = 0; k < matrix.outerSize(); ++k) 
            {
  
                typename sparse_matrix<type_int, type_data>::InnerIterator it(matrix, k);
                // if empty column
                if (!it)
                {
                    continue;
                }

                type_int count = 0;
                for (; it; ++it) 
                {
                    
                    if(it.row() > k)
                    {
                        // write the entries, make sure to set to positive to make factorization more convenient
                        if(it.value() == 0 && announce == 0)
                        {
                            printf("explicit 0 in the matrix shouldn't be stored, they might break the algorithm\n");
                            announce++;
                        }
                        if(it.value() == 0.0)
                        {
                           values[col_ptrs[k] + count] = fabs(it.value());
                        }
                        else
                        {
                            values[col_ptrs[k] + count] = fabs(it.value());
                        }
                        
                        row_indices[col_ptrs[k] + count] = it.row();
                        count++;
                    }
                }
            }

            ret.num_cols = num_cols;
            ret.num_rows = num_rows;
            ret.values = values;
            ret.row_indices = row_indices;
            ret.col_ptrs = col_ptrs;

            return ret;
        }



    
};

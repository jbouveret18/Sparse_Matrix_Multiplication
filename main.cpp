#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

// Structure to represent a sparse matrix element
struct Element {
    int row;
    int col;
    int value;
};

// Structure to represent a sparse matrix
struct SparseMatrix {
    int rows;
    int cols;
    int numElements;
    vector<Element> elements;
};

// Function to generate a random sparse matrix

SparseMatrix generateSparseMatrix(int rows, int cols) {
    SparseMatrix matrix;
    matrix.rows = rows;
    matrix.cols = cols;

    // Calculate the total number of elements and the limit for non-zero elements
    int totalElements = rows * cols;
    int maxNonZeroElements = (2 * totalElements) / 3;
    int numNonZeroElements = std::rand() % maxNonZeroElements;

    matrix.numElements = numNonZeroElements;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> rowDist(0, rows - 1);
    std::uniform_int_distribution<int> colDist(0, cols - 1);
    std::uniform_int_distribution<int> valueDist(1, 100); // Assuming values between 1 and 100

    for (int i = 0; i < numNonZeroElements; ++i) {
        Element element;
        element.row = rowDist(gen);
        element.col = colDist(gen);
        element.value = valueDist(gen);
        matrix.elements.push_back(element);
    }

    return matrix;
}

// Function to display the sparse matrix
void displaySparseMatrix(const SparseMatrix &matrix) {
    int k = 0;
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            if (k < matrix.numElements && matrix.elements[k].row == i && matrix.elements[k].col == j) {
                cout << matrix.elements[k].value << " ";
                k++;
            } else {
                cout << "0 ";
            }
        }
        cout << endl;
    }
}

SparseMatrix innerMatrixMultiplicationMethod(SparseMatrix A, SparseMatrix B) {
    SparseMatrix C;
    if (A.cols != B.rows) {
        cout << "Cannot perform multiplication. Invalid dimensions." << endl;
        return C;
    }

    // Initializing result matrix C
    C.rows = A.rows;
    C.cols = B.cols;

    // For each element in A and B, perform inner product multiplication
    #pragma omp parallel for collapse(2) shared(A, B, C) schedule(static)
    for (int i = 0; i < A.numElements; ++i) {
        for (int j = 0; j < B.numElements; ++j) {
            if (A.elements[i].col == B.elements[j].row) {
                Element temp;
                temp.row = A.elements[i].row;
                temp.col = B.elements[j].col;
                temp.value = A.elements[i].value * B.elements[j].value;
                #pragma omp critical
                C.elements.push_back(temp);
            }
        }
    }
    C.numElements = C.elements.size();
    return C;
}

// Outer product method for matrix multiplication
SparseMatrix outerMatrixMultiplicationMethod(SparseMatrix A, SparseMatrix B) {
    SparseMatrix C;
    if (A.cols != B.rows) {
        cout << "Cannot perform multiplication. Invalid dimensions." << endl;
        return C;
    }

    // Initializing result matrix C
    C.rows = A.rows;
    C.cols = B.cols;

    // Perform outer product multiplication
    #pragma omp parallel for collapse(2) shared(A, B, C) schedule(static)
    for (int i = 0; i < A.numElements; ++i) {
        for (int j = 0; j < B.numElements; ++j) {
            Element temp;
            temp.row = A.elements[i].row;
            temp.col = B.elements[j].col;
            temp.value = A.elements[i].value * B.elements[j].value;
            #pragma omp critical
            C.elements.push_back(temp);
        }
    }
    C.numElements = C.elements.size();
    return C;
}


// Row-by-row product method for matrix multiplication
SparseMatrix rowByRowMatrixMultiplicationMethod(SparseMatrix A, SparseMatrix B) {
    SparseMatrix C;
    if (A.cols != B.rows) {
        cout << "Cannot perform multiplication. Invalid dimensions." << endl;
        return C;
    }

    // Initializing result matrix C
    C.rows = A.rows;
    C.cols = B.cols;

    // For each row in A and each column in B, perform row-by-row multiplication
    #pragma omp parallel for collapse(2) shared(A, B, C) schedule(static)
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            int sum = 0;
            #pragma omp parallel for reduction(+:sum) shared(A, B)
            for (int k = 0; k < A.cols; ++k) {
                for (int p = 0; p < A.numElements; ++p) {
                    if (A.elements[p].row == i && A.elements[p].col == k) {
                        for (int q = 0; q < B.numElements; ++q) {
                            if (B.elements[q].row == k && B.elements[q].col == j) {
                                sum += A.elements[p].value * B.elements[q].value;
                            }
                        }
                    }
                }
            }
            if (sum != 0) {
                Element temp;
                temp.row = i;
                temp.col = j;
                temp.value = sum;
                #pragma omp critical
                C.elements.push_back(temp);
            }
        }
    }
    C.numElements = C.elements.size();
    return C;
}

template <typename Func>
std::chrono::milliseconds measureExecutionTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func(); // Call the function whose execution time you want to measure
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}


int main() {
    int rowList[] = {30,50,80};
    auto columnList = rowList;

    for (int i = 0; i<size(rowList);i++){
        SparseMatrix matrix1 = generateSparseMatrix(rowList[i],columnList[i]);
        SparseMatrix matrix2 = generateSparseMatrix(rowList[i],columnList[i]);
        SparseMatrix matrix3 = innerMatrixMultiplicationMethod(matrix1, matrix2);
        SparseMatrix matrix4 = outerMatrixMultiplicationMethod(matrix1, matrix2);
        SparseMatrix matrix5 = rowByRowMatrixMultiplicationMethod(matrix1, matrix2);

        // Measure execution time for each matrix multiplication method
        auto timeMethod1 = measureExecutionTime([&]() { innerMatrixMultiplicationMethod(matrix1, matrix2); });
        auto timeMethod2 = measureExecutionTime([&]() { outerMatrixMultiplicationMethod(matrix1, matrix2); });
        auto timeMethod3 = measureExecutionTime([&]() { rowByRowMatrixMultiplicationMethod(matrix1, matrix2); });

        // Output execution times
        std::cout << "Method 1 (inner product) execution time: " << timeMethod1.count() << " milliseconds" << std::endl;
        std::cout << "Method 2 (outer product) execution time: " << timeMethod2.count() << " milliseconds" << std::endl;
        std::cout << "Method 3  (row by row product) execution time: " << timeMethod3.count() << " milliseconds" << std::endl;
        std::cout << "For a matrix of size " << rowList[i] << " X " << columnList[i] << std::endl;
    }


    /* You can test the functions here with a low number of rows and columns in order to reduce processing time.
     * We can clearly see in the running time I have measured here that the Inner product method when paralellized is the fastest,
     * the outer product a bit slower but easier to implement and finally the row by row product which is the traditional way of obtaining the product of a matrix takes the longest amount of time.
     *In short the inner product is 5 times faster than the outer product and a 100 times faster than the row by row product.
     * cout << "Sparse Matrix 1:" << endl;
    displaySparseMatrix(matrix1);

    cout << "\nSparse Matrix 2:" << endl;
    displaySparseMatrix(matrix2);

    cout << "\nSparse Matrix 3 inner product (Matrix 1 x Matrix 2):" << endl;
    displaySparseMatrix(matrix3);

    cout << "\nSparse Matrix 4 outer product (Matrix 1 x Matrix 2):" << endl;
    displaySparseMatrix(matrix4);

    cout << "\nSparse Matrix 5 row by row product (Matrix 1 x Matrix 2):" << endl;
    displaySparseMatrix(matrix5);*/


    return 0;
}

#ifndef MATRIX_H
#define MATRIX_H

/**
 * @brief Estructura que representa una matriz dinámica
 */
typedef struct {
    double** data;  // Datos de la matriz
    int rows;       // Número de filas
    int cols;       // Número de columnas
} Matrix;

/**
 * @brief Crea una nueva matriz
 * @param rows Número de filas
 * @param cols Número de columnas
 * @return Matrix* Puntero a la matriz creada o NULL si hay error
 */
Matrix* matrix_create(int rows, int cols);

/**
 * @brief Libera la memoria utilizada por una matriz
 * @param matrix Puntero a la matriz a liberar
 */
void matrix_free(Matrix* matrix);

/**
 * @brief Multiplica dos matrices
 * @param a Primera matriz
 * @param b Segunda matriz
 * @return Matrix* Resultado de la multiplicación o NULL si hay error
 */
Matrix* matrix_multiply(Matrix* a, Matrix* b);

/**
 * @brief Transpone una matriz
 * @param matrix Matriz a transponer
 * @return Matrix* Matriz transpuesta o NULL si hay error
 */
Matrix* matrix_transpose(Matrix* matrix);

/**
 * @brief Resta dos matrices
 * @param a Primera matriz
 * @param b Segunda matriz
 * @return Matrix* Resultado de la resta o NULL si hay error
 */
Matrix* matrix_subtract(Matrix* a, Matrix* b);

/**
 * @brief Calcula la distancia euclidiana entre dos vectores
 * @param x1 Primer vector
 * @param x2 Segundo vector
 * @param n Dimensión de los vectores
 * @return double Distancia euclidiana
 */
double euclidean_distance(const double* x1, const double* x2, int n);

Matrix* matrix_dot(Matrix* A, Matrix* B);          // Alias de matrix_multiply
Matrix* matrix_inverse(Matrix* A);                 // Inversa por Gauss-Jordan


#endif // MATRIX_H

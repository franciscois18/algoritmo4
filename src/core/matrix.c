#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"

Matrix* matrix_create(int rows, int cols) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    if (!matrix) return NULL;
    
    matrix->rows = rows;
    matrix->cols = cols;
    
    // Asignar memoria para filas
    matrix->data = (double**)malloc(rows * sizeof(double*));
    if (!matrix->data) {
        free(matrix);
        return NULL;
    }
    
    // Asignar memoria para columnas
    for (int i = 0; i < rows; i++) {
        matrix->data[i] = (double*)calloc(cols, sizeof(double));
        if (!matrix->data[i]) {
            // Liberar memoria ya asignada
            for (int j = 0; j < i; j++) {
                free(matrix->data[j]);
            }
            free(matrix->data);
            free(matrix);
            return NULL;
        }
    }
    
    return matrix;
}

void matrix_free(Matrix* matrix) {
    if (!matrix) return;
    
    if (matrix->data) {
        for (int i = 0; i < matrix->rows; i++) {
            if (matrix->data[i]) {
                free(matrix->data[i]);
            }
        }
        free(matrix->data);
    }
    
    free(matrix);
}

Matrix* matrix_multiply(Matrix* a, Matrix* b) {
    if (!a || !b || a->cols != b->rows) return NULL;
    
    Matrix* result = matrix_create(a->rows, b->cols);
    if (!result) return NULL;
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            result->data[i][j] = 0.0;
            for (int k = 0; k < a->cols; k++) {
                result->data[i][j] += a->data[i][k] * b->data[k][j];
            }
        }
    }
    
    return result;
}

Matrix* matrix_transpose(Matrix* matrix) {
    if (!matrix) return NULL;
    
    Matrix* result = matrix_create(matrix->cols, matrix->rows);
    if (!result) return NULL;
    
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            result->data[j][i] = matrix->data[i][j];
        }
    }
    
    return result;
}

Matrix* matrix_subtract(Matrix* a, Matrix* b) {
    if (!a || !b || a->rows != b->rows || a->cols != b->cols) return NULL;
    
    Matrix* result = matrix_create(a->rows, a->cols);
    if (!result) return NULL;
    
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            result->data[i][j] = a->data[i][j] - b->data[i][j];
        }
    }
    
    return result;
}

double euclidean_distance(const double* x1, const double* x2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = x1[i] - x2[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

Matrix* matrix_dot(Matrix* A, Matrix* B) {
    return matrix_multiply(A, B);
}

Matrix* matrix_inverse(Matrix* A) {
    if (!A || A->rows != A->cols) return NULL;

    int n = A->rows;
    Matrix* aug = matrix_create(n, 2 * n);
    if (!aug) return NULL;

    // Copiar A y añadir identidad
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug->data[i][j] = A->data[i][j];
        }
        for (int j = 0; j < n; j++) {
            aug->data[i][n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gauss-Jordan
    for (int i = 0; i < n; i++) {
        double pivot = aug->data[i][i];
        if (fabs(pivot) < 1e-12) {
            matrix_free(aug);
            return NULL; // no invertible
        }

        // Normalizar fila
        for (int j = 0; j < 2 * n; j++) {
            aug->data[i][j] /= pivot;
        }

        // Eliminar otras filas
        for (int k = 0; k < n; k++) {
            if (k == i) continue;
            double factor = aug->data[k][i];
            for (int j = 0; j < 2 * n; j++) {
                aug->data[k][j] -= factor * aug->data[i][j];
            }
        }
    }

    // Extraer parte derecha (la inversa)
    Matrix* inv = matrix_create(n, n);
    if (!inv) {
        matrix_free(aug);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inv->data[i][j] = aug->data[i][j + n];
        }
    }

    matrix_free(aug);
    return inv;
}

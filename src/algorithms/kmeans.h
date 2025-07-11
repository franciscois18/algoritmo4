#ifndef KMEANS_H
#define KMEANS_H

#include "../core/matrix.h"

/**
 * @brief Estructura para el Algoritmo K-Means
 */
typedef struct {
    int k;              // Número de clusters
    Matrix* centroids;  // Centroides (k x n_features)
} KMeans;

/**
 * @brief Crea un nuevo modelo K-Means
 * @param k Número de clusters
 * @return KMeans* Puntero al modelo o NULL si hay error
 */
KMeans* kmeans_create(int k);

/**
 * @brief Entrena el modelo K-Means con los datos proporcionados
 * @param model Modelo a entrenar
 * @param X Matriz de características
 * @param max_iter Número máximo de iteraciones
 * @param tol Tolerancia para convergencia
 * @return int 1 si éxito, 0 si error
 */
int kmeans_fit(KMeans* model, Matrix* X, int max_iter, double tol);

/**
 * @brief Predice los clusters para nuevos datos
 * @param model Modelo entrenado
 * @param X Matriz de características a predecir
 * @return Matrix* Vector de asignaciones de cluster o NULL si hay error
 */
Matrix* kmeans_predict(KMeans* model, Matrix* X);

/**
 * @brief Calcula la inercia del modelo (suma de distancias al cuadrado)
 * @param model Modelo entrenado
 * @param X Matriz de características
 * @return double Valor de inercia o -1 si hay error
 */
double kmeans_inertia(KMeans* model, Matrix* X);

/**
 * @brief Libera la memoria utilizada por el modelo
 * @param model Modelo a liberar
 */
void kmeans_free(KMeans* model);

/**
 * @brief Guarda el modelo KMeans a un archivo
 * @param model Modelo KMeans
 * @param filename Ruta del archivo de salida
 * @return int 1 si éxito, 0 si error
 */
int kmeans_save(KMeans* model, const char* filename);

/**
 * @brief Carga un modelo KMeans desde un archivo
 * @param filename Ruta del archivo
 * @return KMeans* Modelo cargado o NULL si error
 */
KMeans* kmeans_load(const char* filename);

/**
 * @brief Calcula el índice de silueta para evaluar el clustering
 * @param X Matriz de datos (n x d)
 * @param labels Vector de clusters asignados (n x 1)
 * @param k Número de clusters
 * @return double Índice de silueta promedio (entre -1 y 1)
 */
double kmeans_silhouette_score(Matrix* X, Matrix* labels, int k);

/**
 * @brief Ejecuta múltiples inicializaciones aleatorias de K-Means y devuelve el mejor modelo
 * @param X Matriz de datos
 * @param k Número de clusters
 * @param max_iter Iteraciones máximas por ejecución
 * @param tol Tolerancia de convergencia
 * @param n_init Número de reinicios aleatorios
 * @return KMeans* Modelo con menor inercia
 */
KMeans* kmeans_fit_best(Matrix* X, int k, int max_iter, double tol, int n_init);

void analyze_sensitivity(int max_iter, double tol, Matrix* X, int* k_values, int num_k_values);

#endif // KMEANS_H

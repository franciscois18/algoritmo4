#ifndef KNN_H
#define KNN_H

#include "../core/matrix.h"

/**
 * @brief Métricas de distancia disponibles para KNN
 */
typedef enum {
    EUCLIDEAN,   ///< Distancia Euclidiana
    MANHATTAN,   ///< Distancia de Manhattan
    COSINE       ///< Distancia del Coseno
} DistanceMetric;

/**
 * @brief Estructura para el clasificador K-Vecinos Más Cercanos
 */
typedef struct {
    Matrix* X_train;       ///< Datos de entrenamiento
    Matrix* y_train;       ///< Etiquetas de entrenamiento
    int k;                 ///< Número de vecinos
    int use_kdtree;        ///< Usar k-d tree (1) o búsqueda lineal (0)
    DistanceMetric metric; ///< Métrica de distancia a utilizar
} KNNClassifier;

/**
 * @brief Crea un nuevo clasificador K-Vecinos Más Cercanos
 * @param k Número de vecinos a considerar
 * @return KNNClassifier* Puntero al clasificador o NULL si hay error
 */
KNNClassifier* knn_create(int k);

/**
 * @brief Entrena el clasificador con los datos proporcionados
 * @param knn Clasificador a entrenar
 * @param X Matriz de características de entrenamiento
 * @param y Vector de etiquetas de entrenamiento
 */
void knn_fit(KNNClassifier* knn, Matrix* X, Matrix* y);

/**
 * @brief Realiza predicciones con el clasificador entrenado
 * @param knn Clasificador entrenado
 * @param X Matriz de características a predecir
 * @return Matrix* Vector de predicciones o NULL si hay error
 */
Matrix* knn_predict(KNNClassifier* knn, Matrix* X);

/**
 * @brief Libera la memoria utilizada por el clasificador
 * @param knn Clasificador a liberar
 */
void knn_free(KNNClassifier* knn);

/**
 * @brief Establece la función de distancia a utilizar en k-NN (modo texto)
 * @param name Nombre de la función de distancia: "euclidean", "manhattan" o "cosine"
 */
void knn_set_distance_function(const char* name);

/**
 * @brief Establece la métrica de distancia del clasificador directamente
 * @param knn Clasificador
 * @param metric Tipo de métrica a usar (EUCLIDEAN, MANHATTAN, COSINE)
 */
void knn_set_distance_metric(KNNClassifier* knn, DistanceMetric metric);

/**
 * @brief Calcula la distancia euclidiana entre dos vectores
 * @param a Vector 1
 * @param b Vector 2
 * @param length Longitud de los vectores
 * @return Distancia euclidiana
 */
double distance_euclidean(const double* a, const double* b, int length);

/**
 * @brief Calcula la distancia de Manhattan entre dos vectores
 * @param a Vector 1
 * @param b Vector 2
 * @param length Longitud de los vectores
 * @return Distancia de Manhattan
 */
double distance_manhattan(const double* a, const double* b, int length);

/**
 * @brief Calcula la distancia del coseno entre dos vectores
 * @param a Vector 1
 * @param b Vector 2
 * @param length Longitud de los vectores
 * @return Distancia del coseno
 */
double distance_cosine(const double* a, const double* b, int length);

/**
 * @brief Guarda un modelo k-NN entrenado en un archivo binario
 * @param knn Clasificador entrenado
 * @param filename Ruta del archivo de salida
 * @return 1 si tuvo éxito, 0 si falló
 */
int knn_save(KNNClassifier* knn, const char* filename);

/**
 * @brief Carga un modelo k-NN previamente guardado
 * @param filename Ruta del archivo con el modelo
 * @return KNNClassifier* Modelo cargado o NULL si hubo error
 */
KNNClassifier* knn_load(const char* filename);

/**
 * @brief Habilita o desactiva el uso de k-d tree (1 para usar, 0 para búsqueda lineal)
 * @param knn Clasificador
 * @param use_kdtree 1 o 0
 */
void knn_set_use_kdtree(KNNClassifier* knn, int use_kdtree);

#endif // KNN_H

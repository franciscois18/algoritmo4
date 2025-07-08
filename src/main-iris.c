/**
 * Ejemplo de Uso de Algoritmos de Machine Learning con el Conjunto de Datos Iris
 * 
 * Este programa muestra cómo implementar los algoritmos de:
 * - K-Vecinos Más Cercanos (KNN)
 * - K-Means
 * - Regresión Lineal
 * 
 * Utilizando el conjunto de datos Iris.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "algorithms/knn.h"
#include "algorithms/kmeans.h"
#include "algorithms/linear_regression.h"
#include "core/matrix.h"
#include "utils/csv_reader.h"

/**
 * Aplica el Algoritmo K-Vecinos Más Cercanos al Conjunto de Datos Iris
 */
void aplicar_knn(CSVData* csv_data) {
    printf("\n=== K-Vecinos Más Cercanos (KNN) ===\n");
    
    // Dividir en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    Matrix *X_train, *y_train, *X_test, *y_test;
    if (!train_test_split(csv_data->data, csv_data->labels, 0.2, 
                         &X_train, &y_train, &X_test, &y_test)) {
        printf("Error al dividir los datos\n");
        return;
    }
    
    printf("Conjunto de entrenamiento: %d muestras x %d características\n", 
           X_train->rows, X_train->cols);
    printf("Conjunto de prueba: %d muestras x %d características\n", 
           X_test->rows, X_test->cols);
    
    // Crear y entrenar el modelo KNN
    int k = 3;
    KNNClassifier* knn = knn_create(k);
    if (!knn) {
        printf("Error al crear el clasificador KNN\n");
        goto cleanup_knn;
    }
    
    // Entrenar el modelo (knn_fit es void, no devuelve valor)
    knn_fit(knn, X_train, y_train);
    
    // Realizar predicciones
    Matrix* y_pred = knn_predict(knn, X_test);
    if (!y_pred) {
        printf("Error al realizar predicciones con KNN\n");
        goto cleanup_knn;
    }
    
    // Calcular precisión (porcentaje de predicciones correctas)
    int correctas = 0;
    for (int i = 0; i < y_test->rows; i++) {
        if (y_test->data[i][0] == y_pred->data[i][0]) {
            correctas++;
        }
    }
    double precision = (double)correctas / y_test->rows;
    
    printf("Precisión del modelo KNN (k=%d): %.4f\n", k, precision);
    
    // Mostrar algunas predicciones
    printf("\nPrimeras 5 predicciones:\n");
    for (int i = 0; i < 5 && i < y_test->rows; i++) {
        printf("Real: %.0f, Predicción: %.0f\n", y_test->data[i][0], y_pred->data[i][0]);
    }
    
    // Liberar memoria
    matrix_free(y_pred);

cleanup_knn:
    knn_free(knn);
    matrix_free(X_train);
    matrix_free(y_train);
    matrix_free(X_test);
    matrix_free(y_test);
}

/**
 * Aplica el Algoritmo K-Means al Conjunto de Datos Iris
 */
void aplicar_kmeans(CSVData* csv_data) {
    printf("\n=== K-Means ===\n");
    
    Matrix* X = matrix_create(csv_data->data->rows, csv_data->data->cols);
    if (!X) {
        printf("Error al crear matriz para K-Means\n");
        return; // Salida temprana si falla la creación de X
    }
    
    for (int i = 0; i < csv_data->data->rows; i++) {
        for (int j = 0; j < csv_data->data->cols; j++) {
            X->data[i][j] = csv_data->data->data[i][j];
        }
    }
    
    // Inicializar punteros a NULL
    double* medias = NULL;
    double* desv_std = NULL;
    KMeans* kmeans = NULL;
    
    medias = (double*)malloc(X->cols * sizeof(double));
    desv_std = (double*)malloc(X->cols * sizeof(double));
    
    if (!medias || !desv_std) {
        printf("Error de memoria al normalizar datos\n");
        matrix_free(X);
        free(medias); // free(NULL) es seguro
        free(desv_std); // free(NULL) es seguro
        return;
    }
    
    // ... (resto del código de normalización sin cambios) ...
    for (int j = 0; j < X->cols; j++) {
        medias[j] = 0.0;
        desv_std[j] = 0.0;
    }
    
    for (int i = 0; i < X->rows; i++) {
        for (int j = 0; j < X->cols; j++) {
            medias[j] += X->data[i][j];
        }
    }
    
    for (int j = 0; j < X->cols; j++) {
        medias[j] /= X->rows;
    }
    
    for (int i = 0; i < X->rows; i++) {
        for (int j = 0; j < X->cols; j++) {
            double diff = X->data[i][j] - medias[j];
            desv_std[j] += diff * diff;
        }
    }
    
    for (int j = 0; j < X->cols; j++) {
        desv_std[j] = sqrt(desv_std[j] / X->rows);
        if (desv_std[j] < 1e-10) {
            desv_std[j] = 1.0;
        }
    }
    
    for (int i = 0; i < X->rows; i++) {
        for (int j = 0; j < X->cols; j++) {
            X->data[i][j] = (X->data[i][j] - medias[j]) / desv_std[j];
        }
    }
    
    int n_clusters = 3;
    int max_iter = 100;
    double tol = 1e-4;
    
    kmeans = kmeans_create(n_clusters);
    if (!kmeans) {
        printf("Error al crear el modelo K-Means\n");
        goto cleanup_kmeans;
    }
    
    if (!kmeans_fit(kmeans, X, max_iter, tol)) {
        printf("Error al entrenar el modelo K-Means\n");
        goto cleanup_kmeans;
    }
    
    Matrix* clusters = kmeans_predict(kmeans, X);
    if (!clusters) {
        printf("Error al realizar predicciones con K-Means\n");
        goto cleanup_kmeans;
    }
    
    double inercia = kmeans_inertia(kmeans, X);
    
    printf("Número de clusters: %d\n", n_clusters);
    printf("Inercia: %.4f\n", inercia);
    
    int* conteo_clusters = (int*)calloc(n_clusters, sizeof(int));
    if (conteo_clusters) {
        for (int i = 0; i < clusters->rows; i++) {
            int cluster = (int)clusters->data[i][0];
            if (cluster >= 0 && cluster < n_clusters) {
                conteo_clusters[cluster]++;
            }
        }
        
        printf("\nDistribución de clusters:\n");
        for (int i = 0; i < n_clusters; i++) {
            printf("Cluster %d: %d muestras\n", i, conteo_clusters[i]);
        }
        
        free(conteo_clusters);
    }
    
    matrix_free(clusters);
    
cleanup_kmeans:
    // Todas estas llamadas son seguras incluso si los punteros son NULL
    kmeans_free(kmeans);
    matrix_free(X);
    free(medias);
    free(desv_std);
}

/**
 * Aplica el Algoritmo de Regresión Lineal al Conjunto de Datos Iris
 */
void aplicar_regresion_lineal(CSVData* csv_data) {
    printf("\n=== Regresión Lineal ===\n");
    
    // Para la regresión, usaremos la longitud del pétalo para predecir el ancho del pétalo
    // Extraer características relevantes (longitud del pétalo = columna 2)
    Matrix* X_regresion = matrix_create(csv_data->data->rows, 1);
    // Variable objetivo (ancho del pétalo = columna 3)
    Matrix* y_regresion = matrix_create(csv_data->data->rows, 1);
    
    if (!X_regresion || !y_regresion) {
        printf("Error al crear matrices para regresión\n");
        matrix_free(X_regresion);
        matrix_free(y_regresion);
        return;
    }
    
    // Copiar datos
    for (int i = 0; i < csv_data->data->rows; i++) {
        X_regresion->data[i][0] = csv_data->data->data[i][2]; // Longitud del pétalo
        y_regresion->data[i][0] = csv_data->data->data[i][3]; // Ancho del pétalo
    }
    
    // Dividir en conjuntos de entrenamiento y prueba
    Matrix *X_train, *y_train, *X_test, *y_test;
    if (!train_test_split(X_regresion, y_regresion, 0.2, 
                         &X_train, &y_train, &X_test, &y_test)) {
        printf("Error al dividir los datos para regresión\n");
        matrix_free(X_regresion);
        matrix_free(y_regresion);
        return;
    }
    
    // Liberar las matrices originales ya que tenemos las divididas
    matrix_free(X_regresion);
    matrix_free(y_regresion);
    
    // Crear y entrenar el modelo de regresión lineal
    double learning_rate = 0.01;
    int max_iterations = 1000;
    double tolerance = 1e-6;
    LinearRegression* regresion = linear_regression_create(learning_rate, max_iterations, tolerance);
    if (!regresion) {
        printf("Error al crear el modelo de regresión lineal\n");
        goto cleanup_regresion;
    }
    
    // Entrenar el modelo
    if (!linear_regression_fit(regresion, X_train, y_train)) {
        printf("Error al entrenar el modelo de regresión lineal\n");
        goto cleanup_regresion;
    }
    
    // Realizar predicciones
    Matrix* y_pred = linear_regression_predict(regresion, X_test);
    if (!y_pred) {
        printf("Error al realizar predicciones con regresión lineal\n");
        goto cleanup_regresion;
    }
    
    // Calcular métricas
    double mse = linear_regression_mse(regresion, X_test, y_test);
    double r2 = linear_regression_r2_score(regresion, X_test, y_test);
    
    printf("Coeficiente: %.4f\n", regresion->weights->data[0][0]);
    printf("Intercepto: %.4f\n", regresion->bias);
    printf("Error Cuadrático Medio: %.4f\n", mse);
    printf("Coeficiente R²: %.4f\n", r2);
    
    // Mostrar algunas predicciones
    printf("\nPrimeras 5 predicciones:\n");
    for (int i = 0; i < 5 && i < y_test->rows; i++) {
        printf("X=%.2f, Real: %.2f, Predicción: %.2f\n", 
               X_test->data[i][0], y_test->data[i][0], y_pred->data[i][0]);
    }
    
    // Liberar memoria
    matrix_free(y_pred);
    
cleanup_regresion:
    linear_regression_free(regresion);
    matrix_free(X_train);
    matrix_free(y_train);
    matrix_free(X_test);
    matrix_free(y_test);
}

/**
 * Función Principal
 */
int main() {
    printf("Ejemplo de Machine Learning en C con el Conjunto de Datos Iris\n");
    printf("===========================================================\n");
    
    // Cargar el conjunto de datos Iris
    const char* filename = "data/iris.csv";
    printf("Cargando datos desde: %s\n", filename);
    
    // El archivo tiene encabezado y la columna de etiquetas es la última (índice 4)
    CSVData* csv_data = csv_read(filename, 1, 4, ',');
    
    if (!csv_data) {
        printf("Error al leer el archivo CSV\n");
        return 1;
    }
    
    // Mostrar información sobre los datos
    printf("Dimensiones de los datos: %d filas x %d columnas\n", 
           csv_data->data->rows, csv_data->data->cols);
    
    // Mostrar encabezados si están disponibles
    if (csv_data->has_header && csv_data->header) {
        printf("Encabezados: ");
        for (int i = 0; i < csv_data->data->cols; i++) {
            printf("%s ", csv_data->header[i]);
        }
        printf("%s (etiqueta)\n", csv_data->header[csv_data->label_col]);
    }
    
    // Mostrar algunas muestras
    printf("\nPrimeras 5 muestras:\n");
    for (int i = 0; i < 5 && i < csv_data->data->rows; i++) {
        printf("Muestra %d: [", i);
        for (int j = 0; j < csv_data->data->cols; j++) {
            printf("%.1f", csv_data->data->data[i][j]);
            if (j < csv_data->data->cols - 1) printf(", ");
        }
        printf("] -> Clase: %.0f\n", csv_data->labels->data[i][0]);
    }
    
    // Aplicar los algoritmos
    aplicar_knn(csv_data);
    aplicar_kmeans(csv_data);
    aplicar_regresion_lineal(csv_data);
    
    // Liberar memoria
    csv_free(csv_data);
    
    return 0;
}

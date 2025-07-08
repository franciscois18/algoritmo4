#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "core/matrix.h"
#include "algorithms/knn.h"
#include "algorithms/kmeans.h"
#include "algorithms/linear_regression.h"
#include "utils/metrics.h"
#include "utils/csv_reader.h"

/**
 * @brief Evalúa y muestra las métricas de un modelo de clasificación.
 * @param nombre Nombre del modelo a evaluar.
 * @param y_true Matriz de etiquetas verdaderas.
 * @param y_pred Matriz de etiquetas predichas.
 */
void evaluar_metricas_clasificacion(const char* nombre, Matrix* y_true, Matrix* y_pred) {
    double acc = accuracy_score(y_true, y_pred);
    double prec = precision_score(y_true, y_pred);
    double rec = recall_score(y_true, y_pred);
    double f1 = f1_score(y_true, y_pred);

    printf("\n--- Métricas para %s ---\n", nombre);
    printf("Accuracy : %.4f\n", acc);
    printf("Precision: %.4f\n", prec);
    printf("Recall   : %.4f\n", rec);
    printf("F1 Score : %.4f\n", f1);
}

/**
 * @brief Evalúa y muestra las métricas de un modelo de regresión.
 * @param y_true Matriz de valores verdaderos.
 * @param y_pred Matriz de valores predichos.
 */
void evaluar_metricas_regresion(const char* nombre, Matrix* y_true, Matrix* y_pred) {
    double mse = mean_squared_error(y_true, y_pred);
    double mae = mean_absolute_error(y_true, y_pred);
    double r2  = r2_score(y_true, y_pred);

    printf("\n--- Métricas para %s ---\n", nombre);
    printf("Error Cuadrático Medio (MSE) : %.4f\n", mse);
    printf("Error Absoluto Medio (MAE)   : %.4f\n", mae);
    printf("Coeficiente de Det. (R2)     : %.4f\n", r2);
}

int main() {
    printf("=== Prueba y Evaluación de Métricas de Machine Learning ===\n");

    // 1. Cargar el conjunto de datos Iris desde el archivo CSV
    // El archivo tiene encabezado (has_header=1) y la etiqueta está en la última columna (label_col=4)
    CSVData* data = csv_read("data/iris.csv", 1, 4, ',');
    if (!data) {
        fprintf(stderr, "Error: No se pudo leer el archivo 'data/iris.csv'.\n");
        return 1;
    }

    Matrix* X_full = data->data;
    Matrix* y_full = data->labels;

    // --- Prueba de Clasificación con K-Vecinos Más Cercanos (KNN) ---
    Matrix *X_train_knn, *y_train_knn, *X_test_knn, *y_test_knn;
    train_test_split(X_full, y_full, 0.2, &X_train_knn, &y_train_knn, &X_test_knn, &y_test_knn);

    KNNClassifier* knn = knn_create(5);
    knn_fit(knn, X_train_knn, y_train_knn);
    Matrix* y_pred_knn = knn_predict(knn, X_test_knn);

    evaluar_metricas_clasificacion("K-Vecinos Más Cercanos (KNN)", y_test_knn, y_pred_knn);

    knn_free(knn);
    matrix_free(X_train_knn);
    matrix_free(y_train_knn);
    matrix_free(X_test_knn);
    matrix_free(y_test_knn);
    matrix_free(y_pred_knn);

    // --- Prueba de Agrupamiento con K-Means ---
    int k_clusters = 3;
    KMeans* kmeans = kmeans_create(k_clusters);
    kmeans_fit(kmeans, X_full, 100, 1e-4);
    Matrix* kmeans_labels = kmeans_predict(kmeans, X_full);

    double inertia = kmeans_inertia(kmeans, X_full);
    double silhouette = kmeans_silhouette_score(X_full, kmeans_labels, k_clusters);

    printf("\n--- Métricas para K-Means ---\n");
    printf("Inercia          : %.4f\n", inertia);
    printf("Índice de Silueta: %.4f\n", silhouette);
    
    kmeans_free(kmeans);
    matrix_free(kmeans_labels);

    // --- Prueba de Regresión Lineal ---
    // Tarea corregida: Predecir el ancho del pétalo (col 3) a partir del largo del pétalo (col 2)
    Matrix* Xr = matrix_create(X_full->rows, 1);
    Matrix* yr = matrix_create(X_full->rows, 1);
    for (int i = 0; i < X_full->rows; i++) {
        Xr->data[i][0] = X_full->data[i][2]; // Largo del pétalo
        yr->data[i][0] = X_full->data[i][3]; // Ancho del pétalo
    }

    Matrix *Xr_train, *yr_train, *Xr_test, *yr_test;
    train_test_split(Xr, yr, 0.2, &Xr_train, &yr_train, &Xr_test, &yr_test);

    LinearRegression* lr = linear_regression_create(0.01, 1000, 1e-6);
    linear_regression_fit(lr, Xr_train, yr_train);
    Matrix* y_pred_lr = linear_regression_predict(lr, Xr_test);

    evaluar_metricas_regresion("Regresión Lineal", yr_test, y_pred_lr);

    // Liberación final de memoria
    linear_regression_free(lr);
    matrix_free(Xr);
    matrix_free(yr);
    matrix_free(Xr_train);
    matrix_free(yr_train);
    matrix_free(Xr_test);
    matrix_free(yr_test);
    matrix_free(y_pred_lr);
    
    csv_free(data);

    return 0;
}
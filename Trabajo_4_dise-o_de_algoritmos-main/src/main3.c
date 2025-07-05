#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "core/matrix.h"
#include "algorithms/knn.h"
#include "algorithms/kmeans.h"
#include "utils/metrics.h"

void preparar_datos_knn(Matrix** X_train, Matrix** y_train, Matrix** X_test, Matrix** y_test) {
    *X_train = matrix_create(6, 2);
    *y_train = matrix_create(6, 1);
    *X_test = matrix_create(4, 2);
    *y_test = matrix_create(4, 1);

    // Clase 0
    (*X_train)->data[0][0] = 1.0; (*X_train)->data[0][1] = 2.0; (*y_train)->data[0][0] = 0;
    (*X_train)->data[1][0] = 1.5; (*X_train)->data[1][1] = 1.8; (*y_train)->data[1][0] = 0;

    // Clase 1
    (*X_train)->data[2][0] = 5.0; (*X_train)->data[2][1] = 8.0; (*y_train)->data[2][0] = 1;
    (*X_train)->data[3][0] = 6.0; (*X_train)->data[3][1] = 9.0; (*y_train)->data[3][0] = 1;

    // Clase 2
    (*X_train)->data[4][0] = 9.0; (*X_train)->data[4][1] = 1.0; (*y_train)->data[4][0] = 2;
    (*X_train)->data[5][0] = 8.0; (*X_train)->data[5][1] = 2.0; (*y_train)->data[5][0] = 2;

    // Test
    (*X_test)->data[0][0] = 1.2; (*X_test)->data[0][1] = 2.1; (*y_test)->data[0][0] = 0;
    (*X_test)->data[1][0] = 5.5; (*X_test)->data[1][1] = 8.5; (*y_test)->data[1][0] = 1;
    (*X_test)->data[2][0] = 8.5; (*X_test)->data[2][1] = 1.0; (*y_test)->data[2][0] = 2;
    (*X_test)->data[3][0] = 5.0; (*X_test)->data[3][1] = 1.0; (*y_test)->data[3][0] = 2;
}

void evaluar_metricas(const char* nombre, Matrix* y_true, Matrix* y_pred) {
    double acc = accuracy_score(y_true, y_pred);
    double prec = precision_score(y_true, y_pred);  // Precisión macro
    double rec = recall_score(y_true, y_pred);
    double f1 = f1_score(y_true, y_pred);

    printf("\n--- Métricas para %s ---\n", nombre);
    printf("Accuracy : %.4f\n", acc);
    printf("Precision: %.4f\n", prec);
    printf("Recall   : %.4f\n", rec);
    printf("F1 Score : %.4f\n", f1);
}

int main() {
    printf("=== Prueba de Métricas de Clasificación ===\n");

    // --- KNN ---
    Matrix *X_train, *y_train, *X_test, *y_test;
    preparar_datos_knn(&X_train, &y_train, &X_test, &y_test);

    knn_set_distance_function("euclidean");
    KNNClassifier* knn = knn_create(3);  // sin k-d tree ni extra args
    knn_fit(knn, X_train, y_train);
    Matrix* y_pred_knn = knn_predict(knn, X_test);

    evaluar_metricas("KNN", y_test, y_pred_knn);

    knn_free(knn);
    matrix_free(X_train);
    matrix_free(y_train);
    matrix_free(X_test);
    matrix_free(y_test);
    matrix_free(y_pred_knn);

    // --- K-Means ---
    printf("\n=== Prueba con K-Means (Evaluación tentativa) ===\n");
    int n_samples = 30, n_features = 2;
    Matrix* X = matrix_create(n_samples, n_features);
    Matrix* y_true_kmeans = matrix_create(n_samples, 1);

    for (int i = 0; i < 10; i++) {
        X->data[i][0] = 1.0 + i * 0.1;
        X->data[i][1] = 1.0 + i * 0.2;
        y_true_kmeans->data[i][0] = 0;
    }
    for (int i = 10; i < 20; i++) {
        X->data[i][0] = 5.0 + (i - 10) * 0.2;
        X->data[i][1] = 5.0 + (i - 10) * 0.1;
        y_true_kmeans->data[i][0] = 1;
    }
    for (int i = 20; i < 30; i++) {
        X->data[i][0] = 9.0 + (i - 20) * 0.1;
        X->data[i][1] = 1.0 + (i - 20) * 0.2;
        y_true_kmeans->data[i][0] = 2;
    }

    KMeans* kmeans = kmeans_create(3);
    kmeans_fit(kmeans, X, 100, 1e-4);
    Matrix* y_pred_kmeans = kmeans_predict(kmeans, X);

    evaluar_metricas("K-Means", y_true_kmeans, y_pred_kmeans);

    matrix_free(X);
    matrix_free(y_true_kmeans);
    matrix_free(y_pred_kmeans);
    kmeans_free(kmeans);

    return 0;
}

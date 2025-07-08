#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils/csv_reader.h"
#include "core/matrix.h"
#include "algorithms/knn.h"

void evaluar(KNNClassifier* knn, Matrix* X_test, Matrix* y_test, const char* mensaje) {
    Matrix* y_pred = knn_predict(knn, X_test);
    if (!y_pred) {
        fprintf(stderr, "Error al predecir.\n");
        return;
    }

    int correctas = 0;
    for (int i = 0; i < y_test->rows; i++) {
        if ((int)y_test->data[i][0] == (int)y_pred->data[i][0]) {
            correctas++;
        }
    }

    double acc = (double)correctas / y_test->rows;
    printf("Precisión (%s): %.4f\n", mensaje, acc);

    for (int i = 0; i < 5 && i < y_test->rows; i++) {
        printf("Real: %.0f, Predicción: %.0f\n", y_test->data[i][0], y_pred->data[i][0]);
    }

    matrix_free(y_pred);
}

DistanceMetric obtener_metrica(const char* nombre) {
    if (strcmp(nombre, "euclidean") == 0) return EUCLIDEAN;
    if (strcmp(nombre, "manhattan") == 0) return MANHATTAN;
    if (strcmp(nombre, "cosine") == 0) return COSINE;

    fprintf(stderr, "Métrica desconocida: %s. Usando EUCLIDEAN por defecto.\n", nombre);
    return EUCLIDEAN;
}

void probar_con_distancia(const char* nombre_distancia, Matrix* X_train, Matrix* y_train, Matrix* X_test, Matrix* y_test, int k, int use_kdtree) {
    printf("\n=== Distancia: %s | k-d tree: %s ===\n", nombre_distancia, use_kdtree ? "ON" : "OFF");

    KNNClassifier* knn = knn_create(k);
    if (!knn) {
        fprintf(stderr, "Error al crear el clasificador KNN.\n");
        return;
    }

    knn_set_use_kdtree(knn, use_kdtree);
    knn_set_distance_metric(knn, obtener_metrica(nombre_distancia));
    knn_fit(knn, X_train, y_train);

    evaluar(knn, X_test, y_test, "Original");

    const char* modelo_path = "knn_model_temp.dat";
    if (knn_save(knn, modelo_path)) {
        printf("Modelo guardado en '%s'\n", modelo_path);
    } else {
        fprintf(stderr, "Error al guardar el modelo.\n");
    }

    knn_free(knn);
    knn = knn_load(modelo_path);
    if (!knn) {
        fprintf(stderr, "Error al cargar el modelo desde archivo.\n");
        return;
    }

    evaluar(knn, X_test, y_test, "Cargado");
    knn_free(knn);
}

int main() {
    printf("=== Prueba Unitaria: KNN Extendido ===\n");

    CSVData* data = csv_read("data/iris.csv", 1, 4, ',');
    if (!data) {
        fprintf(stderr, "Error al cargar el dataset.\n");
        return 1;
    }

    Matrix *X_train, *y_train, *X_test, *y_test;
    if (!train_test_split(data->data, data->labels, 0.2, &X_train, &y_train, &X_test, &y_test)) {
        fprintf(stderr, "Error en train_test_split.\n");
        csv_free(data);
        return 1;
    }

    int k = 3;
    const char* distancias[] = {"euclidean", "manhattan", "cosine"};
    for (int i = 0; i < 3; i++) {
        probar_con_distancia(distancias[i], X_train, y_train, X_test, y_test, k, 1);
        probar_con_distancia(distancias[i], X_train, y_train, X_test, y_test, k, 0);
    }

    matrix_free(X_train);
    matrix_free(y_train);
    matrix_free(X_test);
    matrix_free(y_test);
    csv_free(data);

    return 0;
}

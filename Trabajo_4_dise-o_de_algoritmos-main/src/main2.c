#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "core/matrix.h"
#include "algorithms/kmeans.h"

void normalizar(Matrix* X) {
    for (int j = 0; j < X->cols; j++) {
        double suma = 0.0, suma2 = 0.0;
        for (int i = 0; i < X->rows; i++) {
            suma += X->data[i][j];
            suma2 += X->data[i][j] * X->data[i][j];
        }
        double media = suma / X->rows;
        double std = sqrt((suma2 / X->rows) - (media * media));
        if (std < 1e-10) std = 1.0;

        for (int i = 0; i < X->rows; i++) {
            X->data[i][j] = (X->data[i][j] - media) / std;
        }
    }
}

int main() {
    printf("=== Prueba Unitaria: K-Means Extendida ===\n");

    int n_samples = 30, n_features = 2;
    Matrix* X = matrix_create(n_samples, n_features);

    // Generar 3 clusters
    for (int i = 0; i < 10; i++) {
        X->data[i][0] = 1.0 + 0.1 * i;
        X->data[i][1] = 1.0 + 0.2 * i;
    }
    for (int i = 10; i < 20; i++) {
        X->data[i][0] = 5.0 + 0.2 * (i - 10);
        X->data[i][1] = 5.0 + 0.1 * (i - 10);
    }
    for (int i = 20; i < 30; i++) {
        X->data[i][0] = 9.0 + 0.1 * (i - 20);
        X->data[i][1] = 1.0 + 0.2 * (i - 20);
    }

    normalizar(X);

    int k = 3, max_iter = 100, n_init = 10;
    double tol = 1e-4;

    printf("Entrenando con %d reinicios aleatorios...\n", n_init);
    KMeans* model = kmeans_fit_best(X, k, max_iter, tol, n_init);
    if (!model) {
        printf("❌ Error al entrenar K-Means\n");
        matrix_free(X);
        return 1;
    }

    Matrix* clusters = kmeans_predict(model, X);
    double inertia = kmeans_inertia(model, X);
    double score = kmeans_silhouette_score(X, clusters, k);

    printf("✅ Entrenamiento exitoso\n");
    printf("Inercia final (mejor): %.4f\n", inertia);
    printf("Índice de silueta: %.4f\n", score);

    // Guardar el modelo
    const char* filename = "modelo_kmeans.dat";
    if (kmeans_save(model, filename)) {
        printf("Modelo guardado en '%s'\n", filename);
    } else {
        printf("❌ Error al guardar el modelo\n");
    }

    // Cargar el modelo
    KMeans* loaded = kmeans_load(filename);
    if (loaded) {
        printf("Modelo cargado desde '%s'\n", filename);
    } else {
        printf("❌ Error al cargar el modelo\n");
        matrix_free(X); kmeans_free(model); return 1;
    }

    // Predicción con modelo cargado
    Matrix* clusters_loaded = kmeans_predict(loaded, X);
    double inertia_loaded = kmeans_inertia(loaded, X);
    printf("Inercia (modelo cargado): %.4f\n", inertia_loaded);

    // Comparación entre predicciones
    int coincidencias = 0;
    for (int i = 0; i < X->rows; i++) {
        if ((int)clusters->data[i][0] == (int)clusters_loaded->data[i][0]) {
            coincidencias++;
        }
    }
    printf("Coincidencias entre predicción original y cargada: %d de %d\n", coincidencias, X->rows);

    // Distribución de clusters
    int* conteo = (int*)calloc(k, sizeof(int));
    for (int i = 0; i < clusters->rows; i++) {
        conteo[(int)clusters->data[i][0]]++;
    }
    for (int i = 0; i < k; i++) {
        printf("Cluster %d: %d muestras\n", i, conteo[i]);
    }

    int valido = 1;
    for (int i = 0; i < k; i++) {
        if (conteo[i] < 5 || conteo[i] > 15) {
            valido = 0;
            break;
        }
    }
    printf("Distribución %s\n", valido ? "válida" : "no válida");

    // Limpieza
    free(conteo);
    matrix_free(X);
    matrix_free(clusters);
    matrix_free(clusters_loaded);
    kmeans_free(model);
    kmeans_free(loaded);

    return 0;
}

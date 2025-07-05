#include "kmeans.h"
#include "../core/matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <string.h>

// --- K-Means++ initialization ---
static void initialize_kmeanspp_centroids(KMeans* model, Matrix* X) {
    int n_samples = X->rows;
    int n_features = X->cols;
    model->centroids = matrix_create(model->k, n_features);

    int first = rand() % n_samples;
    for (int j = 0; j < n_features; j++) {
        model->centroids->data[0][j] = X->data[first][j];
    }

    double* dist = (double*)malloc(sizeof(double) * n_samples);

    for (int c = 1; c < model->k; c++) {
        double total = 0.0;

        for (int i = 0; i < n_samples; i++) {
            double min_dist = DBL_MAX;
            for (int j = 0; j < c; j++) {
                double d = euclidean_distance(X->data[i], model->centroids->data[j], n_features);
                if (d < min_dist) min_dist = d;
            }
            dist[i] = min_dist * min_dist;
            total += dist[i];
        }

        double r = ((double)rand() / RAND_MAX) * total;
        double sum = 0.0;
        int next = 0;
        for (int i = 0; i < n_samples; i++) {
            sum += dist[i];
            if (sum >= r) {
                next = i;
                break;
            }
        }

        for (int j = 0; j < n_features; j++) {
            model->centroids->data[c][j] = X->data[next][j];
        }
    }

    free(dist);
}

KMeans* kmeans_create(int k) {
    KMeans* model = (KMeans*)malloc(sizeof(KMeans));
    if (!model) return NULL;
    model->k = k;
    model->centroids = NULL;
    return model;
}

int kmeans_fit(KMeans* model, Matrix* X, int max_iter, double tol) {
    if (!model || !X || model->k <= 0) return 0;

    int n_samples = X->rows;
    int n_features = X->cols;

    if (!model->centroids)
        initialize_kmeanspp_centroids(model, X);

    int* labels = (int*)malloc(n_samples * sizeof(int));
    Matrix* new_centroids = matrix_create(model->k, n_features);

    for (int iter = 0; iter < max_iter; iter++) {
        // Asignar clusters
        for (int i = 0; i < n_samples; i++) {
            double min_dist = DBL_MAX;
            int best = 0;
            for (int j = 0; j < model->k; j++) {
                double dist = euclidean_distance(X->data[i], model->centroids->data[j], n_features);
                if (dist < min_dist) {
                    min_dist = dist;
                    best = j;
                }
            }
            labels[i] = best;
        }

        // Calcular nuevos centroides
        int* counts = (int*)calloc(model->k, sizeof(int));
        for (int i = 0; i < model->k; i++)
            for (int j = 0; j < n_features; j++)
                new_centroids->data[i][j] = 0.0;

        for (int i = 0; i < n_samples; i++) {
            int c = labels[i];
            counts[c]++;
            for (int j = 0; j < n_features; j++) {
                new_centroids->data[c][j] += X->data[i][j];
            }
        }

        for (int i = 0; i < model->k; i++) {
            if (counts[i] > 0) {
                for (int j = 0; j < n_features; j++) {
                    new_centroids->data[i][j] /= counts[i];
                }
            }
        }

        // Verificar convergencia
        double shift = 0.0;
        for (int i = 0; i < model->k; i++) {
            shift += euclidean_distance(model->centroids->data[i], new_centroids->data[i], n_features);
        }

        // Actualizar centroides
        for (int i = 0; i < model->k; i++)
            for (int j = 0; j < n_features; j++)
                model->centroids->data[i][j] = new_centroids->data[i][j];

        free(counts);
        if (shift < tol) break;
    }

    matrix_free(new_centroids);
    free(labels);
    return 1;
}

Matrix* kmeans_predict(KMeans* model, Matrix* X) {
    if (!model || !X || !model->centroids) return NULL;

    Matrix* labels = matrix_create(X->rows, 1);
    for (int i = 0; i < X->rows; i++) {
        double min_dist = DBL_MAX;
        int best = 0;
        for (int j = 0; j < model->k; j++) {
            double dist = euclidean_distance(X->data[i], model->centroids->data[j], X->cols);
            if (dist < min_dist) {
                min_dist = dist;
                best = j;
            }
        }
        labels->data[i][0] = (double)best;
    }
    return labels;
}

double kmeans_inertia(KMeans* model, Matrix* X) {
    if (!model || !X || !model->centroids) return -1.0;

    double inertia = 0.0;
    for (int i = 0; i < X->rows; i++) {
        double min_dist = DBL_MAX;
        for (int j = 0; j < model->k; j++) {
            double dist = euclidean_distance(X->data[i], model->centroids->data[j], X->cols);
            if (dist < min_dist) {
                min_dist = dist;
            }
        }
        inertia += min_dist * min_dist;
    }
    return inertia;
}

// Función para el análisis de sensibilidad al número de clusters
void analyze_sensitivity(int max_iter, double tol, Matrix* X, int* k_values, int num_k_values) {
    for (int i = 0; i < num_k_values; i++) {
        int k = k_values[i];
        KMeans* model = kmeans_create(k);
        kmeans_fit(model, X, max_iter, tol);
        double inertia = kmeans_inertia(model, X);
        printf("Inercia para k=%d: %f\n", k, inertia);
        kmeans_free(model);
    }
}

int kmeans_save(KMeans* model, const char* filename) {
    if (!model || !model->centroids) return 0;

    FILE* f = fopen(filename, "wb");
    if (!f) return 0;

    // Guardar número de clusters y dimensiones
    fwrite(&model->k, sizeof(int), 1, f);
    fwrite(&model->centroids->rows, sizeof(int), 1, f);
    fwrite(&model->centroids->cols, sizeof(int), 1, f);

    // Guardar los centroides
    for (int i = 0; i < model->centroids->rows; i++) {
        fwrite(model->centroids->data[i], sizeof(double), model->centroids->cols, f);
    }

    fclose(f);
    return 1;
}

KMeans* kmeans_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    int k, rows, cols;
    fread(&k, sizeof(int), 1, f);
    fread(&rows, sizeof(int), 1, f);
    fread(&cols, sizeof(int), 1, f);

    KMeans* model = kmeans_create(k);
    if (!model) {
        fclose(f);
        return NULL;
    }

    model->centroids = matrix_create(rows, cols);
    if (!model->centroids) {
        kmeans_free(model);
        fclose(f);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        fread(model->centroids->data[i], sizeof(double), cols, f);
    }

    fclose(f);
    return model;
}

void kmeans_free(KMeans* model) {
    if (!model) return;
    matrix_free(model->centroids);
    free(model);
}

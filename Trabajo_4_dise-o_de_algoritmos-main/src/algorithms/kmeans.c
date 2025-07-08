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
    if (fread(&k, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&rows, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&cols, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }

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
        if (fread(model->centroids->data[i], sizeof(double), cols, f) != (size_t)cols) {
            kmeans_free(model);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    return model;
}

static double euclidean_distance_row(const double* a, const double* b, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double kmeans_silhouette_score(Matrix* X, Matrix* labels, int k) {
    if (!X || !labels || X->rows != labels->rows) return -1;

    int n = X->rows;
    int d = X->cols;
    double* silhouettes = malloc(n * sizeof(double));
    if (!silhouettes) return -1;

    for (int i = 0; i < n; i++) {
        int label_i = (int)labels->data[i][0];

        double a = 0.0, b = DBL_MAX;
        int same_cluster_count = 0;

        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double dist = euclidean_distance_row(X->data[i], X->data[j], d);
            int label_j = (int)labels->data[j][0];

            if (label_j == label_i) {
                a += dist;
                same_cluster_count++;
            }
        }

        a = (same_cluster_count > 0) ? a / same_cluster_count : 0.0;

        for (int cluster = 0; cluster < k; cluster++) {
            if (cluster == label_i) continue;

            double avg_dist = 0.0;
            int count = 0;

            for (int j = 0; j < n; j++) {
                if ((int)labels->data[j][0] == cluster) {
                    avg_dist += euclidean_distance_row(X->data[i], X->data[j], d);
                    count++;
                }
            }

            if (count > 0) {
                avg_dist /= count;
                if (avg_dist < b) b = avg_dist;
            }
        }

        double s = 0.0;
        if (a < b) s = 1.0 - a / b;
        else if (a > b) s = b / a - 1.0;
        silhouettes[i] = s;
    }

    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += silhouettes[i];
    free(silhouettes);
    return mean / n;
}

KMeans* kmeans_fit_best(Matrix* X, int k, int max_iter, double tol, int n_init) {
    if (!X || k <= 0 || n_init <= 0) return NULL;

    KMeans* best_model = NULL;
    double best_inertia = DBL_MAX;

    for (int i = 0; i < n_init; i++) {
        KMeans* model = kmeans_create(k);
        if (!model) continue;

        if (!kmeans_fit(model, X, max_iter, tol)) {
            kmeans_free(model);
            continue;
        }

        double inertia = kmeans_inertia(model, X);
        if (inertia < best_inertia) {
            if (best_model) kmeans_free(best_model);
            best_model = model;
            best_inertia = inertia;
        } else {
            kmeans_free(model);
        }
    }

    return best_model;
}


void kmeans_free(KMeans* model) {
    if (!model) return;
    matrix_free(model->centroids);
    free(model);
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "knn.h"

// ==== Distancias ====

double distance_euclidean(const double* a, const double* b, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

double distance_manhattan(const double* a, const double* b, int length) {
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        sum += fabs(a[i] - b[i]);
    }
    return sum;
}

double distance_cosine(const double* a, const double* b, int length) {
    double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
    for (int i = 0; i < length; i++) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0 || norm_b == 0) return 1.0;
    return 1.0 - (dot / (sqrt(norm_a) * sqrt(norm_b)));
}

static double compute_distance_metric(DistanceMetric metric, const double* a, const double* b, int length) {
    switch (metric) {
        case MANHATTAN:
            return distance_manhattan(a, b, length);
        case COSINE:
            return distance_cosine(a, b, length);
        case EUCLIDEAN:
        default:
            return distance_euclidean(a, b, length);
    }
}

void knn_set_distance_metric(KNNClassifier* knn, DistanceMetric metric) {
    if (knn) {
        knn->metric = metric;
    }
}

// ==== K-D Tree ====

typedef struct KDNode {
    int index;
    int axis;
    struct KDNode* left;
    struct KDNode* right;
} KDNode;

static KDNode* build_kdtree(Matrix* X, int* indices, int n, int depth) {
    if (n <= 0) return NULL;

    int axis = depth % X->cols;

    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (X->data[indices[i]][axis] > X->data[indices[j]][axis]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }

    int mid = n / 2;
    KDNode* node = malloc(sizeof(KDNode));
    node->index = indices[mid];
    node->axis = axis;
    node->left = build_kdtree(X, indices, mid, depth + 1);
    node->right = build_kdtree(X, indices + mid + 1, n - mid - 1, depth + 1);

    return node;
}

static void free_kdtree(KDNode* root) {
    if (!root) return;
    free_kdtree(root->left);
    free_kdtree(root->right);
    free(root);
}

typedef struct {
    double distance;
    int index;
} NeighborKD;

static void search_kdtree(KDNode* node, Matrix* X, const double* query, int k,
                          NeighborKD* neighbors, int* neighbor_count, int features, DistanceMetric metric) {
    if (!node) return;

    double dist = compute_distance_metric(metric, X->data[node->index], query, features);

    if (*neighbor_count < k) {
        neighbors[*neighbor_count].distance = dist;
        neighbors[*neighbor_count].index = node->index;
        (*neighbor_count)++;
    } else {
        int max_idx = 0;
        for (int i = 1; i < k; i++) {
            if (neighbors[i].distance > neighbors[max_idx].distance) {
                max_idx = i;
            }
        }
        if (dist < neighbors[max_idx].distance) {
            neighbors[max_idx].distance = dist;
            neighbors[max_idx].index = node->index;
        }
    }

    int axis = node->axis;
    double diff = query[axis] - X->data[node->index][axis];

    KDNode* near = diff < 0 ? node->left : node->right;
    KDNode* far = diff < 0 ? node->right : node->left;

    search_kdtree(near, X, query, k, neighbors, neighbor_count, features, metric);

    double best_dist = 0.0;
    for (int i = 0; i < *neighbor_count; i++) {
        if (i == 0 || neighbors[i].distance < best_dist)
            best_dist = neighbors[i].distance;
    }

    if (*neighbor_count < k || fabs(diff) < best_dist) {
        search_kdtree(far, X, query, k, neighbors, neighbor_count, features, metric);
    }
}

static double weighted_vote_kdtree(Matrix* y, NeighborKD* neighbors, int k) {
    double weights[256] = {0};
    for (int i = 0; i < k; i++) {
        int label = (int)y->data[neighbors[i].index][0];
        double dist = neighbors[i].distance;
        double w = dist < 1e-5 ? 1e6 : 1.0 / dist;
        weights[label] += w;
    }

    int best = 0;
    for (int i = 1; i < 256; i++) {
        if (weights[i] > weights[best]) best = i;
    }
    return (double)best;
}

// ==== API ====

KNNClassifier* knn_create(int k) {
    KNNClassifier* knn = malloc(sizeof(KNNClassifier));
    if (!knn) return NULL;
    knn->k = k;
    knn->X_train = NULL;
    knn->y_train = NULL;
    knn->use_kdtree = 1;
    knn->metric = EUCLIDEAN;
    return knn;
}

void knn_fit(KNNClassifier* knn, Matrix* X, Matrix* y) {
    knn->X_train = X;
    knn->y_train = y;
}

Matrix* knn_predict(KNNClassifier* knn, Matrix* X) {
    Matrix* pred = matrix_create(X->rows, 1);

    if (knn->use_kdtree) {
        int* indices = malloc(knn->X_train->rows * sizeof(int));
        for (int i = 0; i < knn->X_train->rows; i++) indices[i] = i;
        KDNode* tree = build_kdtree(knn->X_train, indices, knn->X_train->rows, 0);
        free(indices);

        for (int i = 0; i < X->rows; i++) {
            NeighborKD* neighbors = malloc(sizeof(NeighborKD) * knn->k);
            int count = 0;
            search_kdtree(tree, knn->X_train, X->data[i], knn->k, neighbors, &count, X->cols, knn->metric);
            pred->data[i][0] = weighted_vote_kdtree(knn->y_train, neighbors, knn->k);
            free(neighbors);
        }

        free_kdtree(tree);
    } else {
        for (int i = 0; i < X->rows; i++) {
            NeighborKD* neighbors = malloc(sizeof(NeighborKD) * knn->k);
            for (int j = 0; j < knn->k; j++) {
                neighbors[j].distance = DBL_MAX;
                neighbors[j].index = -1;
            }

            for (int j = 0; j < knn->X_train->rows; j++) {
                double dist = compute_distance_metric(knn->metric, knn->X_train->data[j], X->data[i], X->cols);

                int max_idx = 0;
                for (int l = 1; l < knn->k; l++) {
                    if (neighbors[l].distance > neighbors[max_idx].distance)
                        max_idx = l;
                }

                if (dist < neighbors[max_idx].distance) {
                    neighbors[max_idx].distance = dist;
                    neighbors[max_idx].index = j;
                }
            }

            pred->data[i][0] = weighted_vote_kdtree(knn->y_train, neighbors, knn->k);
            free(neighbors);
        }
    }

    return pred;
}

void knn_set_use_kdtree(KNNClassifier* knn, int use_kdtree) {
    if (knn) {
        knn->use_kdtree = use_kdtree;
    }
}

int knn_save(KNNClassifier* knn, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return 0;

    fwrite(&knn->k, sizeof(int), 1, f);
    fwrite(&knn->use_kdtree, sizeof(int), 1, f);
    fwrite(&knn->metric, sizeof(int), 1, f);
    fwrite(&knn->X_train->rows, sizeof(int), 1, f);
    fwrite(&knn->X_train->cols, sizeof(int), 1, f);

    for (int i = 0; i < knn->X_train->rows; i++)
        fwrite(knn->X_train->data[i], sizeof(double), knn->X_train->cols, f);

    for (int i = 0; i < knn->y_train->rows; i++)
        fwrite(knn->y_train->data[i], sizeof(double), knn->y_train->cols, f);

    fclose(f);
    return 1;
}

KNNClassifier* knn_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    int k, use_kdtree, metric_int, rows, cols;
    
    // Verificar cada lectura
    if (fread(&k, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&use_kdtree, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&metric_int, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&rows, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&cols, sizeof(int), 1, f) != 1) { fclose(f); return NULL; }

    Matrix* X = matrix_create(rows, cols);
    Matrix* y = matrix_create(rows, 1);
    if (!X || !y) {
        fclose(f);
        if (X) matrix_free(X);
        if (y) matrix_free(y);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        if (fread(X->data[i], sizeof(double), cols, f) != (size_t)cols) {
            fclose(f);
            matrix_free(X);
            matrix_free(y);
            return NULL;
        }
    }
    for (int i = 0; i < rows; i++) {
        if (fread(y->data[i], sizeof(double), 1, f) != 1) {
            fclose(f);
            matrix_free(X);
            matrix_free(y);
            return NULL;
        }
    }

    fclose(f);

    KNNClassifier* knn = knn_create(k);
    if (!knn) {
        matrix_free(X);
        matrix_free(y);
        return NULL;
    }

    knn->X_train = X;
    knn->y_train = y;
    knn->use_kdtree = use_kdtree;
    knn->metric = (DistanceMetric)metric_int;
    return knn;
}

void knn_free(KNNClassifier* knn) {
    if (knn) free(knn);
}

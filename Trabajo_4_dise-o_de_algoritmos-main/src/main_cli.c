#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/matrix.h"
#include "utils/csv_reader.h"
#include "algorithms/knn.h"
#include "algorithms/kmeans.h"
#include "algorithms/linear_regression.h"
#include "utils/metrics.h"

// ==================== Ayuda General ====================
void print_general_help() {
    printf("=== ML CLI - Interfaz de Línea de Comandos ===\n");
    printf("Uso: ./mlcli <comando> [opciones]\n\n");
    printf("Comandos disponibles:\n");
    printf("  knn      Clasificador K-Nearest Neighbors\n");
    printf("  kmeans   Agrupamiento con K-Means\n");
    printf("  linear   Regresión Lineal (petal_width ~ petal_length en Iris)\n");
    printf("  metrics  Cálculo de métricas de evaluación\n");
    printf("\nUse './mlcli <comando> --help' para más detalles.\n");
}

// ==================== KNN ====================
void run_knn(int argc, char* argv[]) {
    char *file = NULL, *output_file = NULL;
    int k = 3;
    int use_kdtree = 0;
    DistanceMetric metric = EUCLIDEAN;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--file") == 0) file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0) output_file = argv[++i];
        else if (strcmp(argv[i], "--k") == 0) k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--distance") == 0) {
            char* d = argv[++i];
            if (strcmp(d, "euclidean") == 0) metric = EUCLIDEAN;
            else if (strcmp(d, "manhattan") == 0) metric = MANHATTAN;
            else if (strcmp(d, "cosine") == 0) metric = COSINE;
        } else if (strcmp(argv[i], "--kdtree") == 0) use_kdtree = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Uso: ./mlcli knn --file data/iris.csv --k 3 --distance euclidean --output pred.csv\n");
            return;
        }
    }

    if (!file || !output_file) {
        fprintf(stderr, "Faltan argumentos. Usa --help.\n");
        return;
    }

    CSVData* csv = csv_read(file, 1, 4, ',');
    if (!csv || !csv->data || !csv->labels) {
        fprintf(stderr, "Error al leer el archivo CSV: %s\n", file);
        return;
    }

    Matrix *X_train, *y_train, *X_test, *y_test;
    if (!train_test_split(csv->data, csv->labels, 0.2, &X_train, &y_train, &X_test, &y_test)) {
        fprintf(stderr, "❌ Error en train_test_split\n");
        return;
    }

    KNNClassifier* knn = knn_create(k);
    knn_set_use_kdtree(knn, use_kdtree);
    knn_set_distance_metric(knn, metric);
    knn_fit(knn, X_train, y_train);

    Matrix* y_pred = knn_predict(knn, X_test);
    write_csv(output_file, y_pred);

    printf("KNN completado. Resultados:\n");
    printf("Accuracy : %.4f\n", accuracy_score(y_test, y_pred));
    printf("Precision: %.4f\n", precision_score(y_test, y_pred));
    printf("Recall   : %.4f\n", recall_score(y_test, y_pred));
    printf("F1 Score : %.4f\n", f1_score(y_test, y_pred));

    knn_free(knn);
    matrix_free(X_train); matrix_free(y_train);
    matrix_free(X_test); matrix_free(y_test);
    matrix_free(y_pred);
    csv_free(csv);
}

// ==================== KMeans ====================
void run_kmeans(int argc, char* argv[]) {
    const char* file = NULL;
    const char* output_file = NULL;
    int k = 3;
    int max_iter = 100;
    double tol = 1e-4;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--file") == 0) file = argv[++i];
        else if (strcmp(argv[i], "--output") == 0) output_file = argv[++i];
        else if (strcmp(argv[i], "--k") == 0) k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max_iter") == 0) max_iter = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tol") == 0) tol = atof(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Uso: ./mlcli kmeans --file data.csv --k 3 --max_iter 200 --tol 0.0001 --output out.csv\n");
            return;
        }
    }

    if (!file || !output_file) {
        fprintf(stderr, "Faltan archivos. Usa --help.\n");
        return;
    }

    CSVData* csv = csv_read(file, 1, 4, ',');
    if (!csv) {
        fprintf(stderr, "Error al leer archivo: %s\n", file);
        return;
    }

    Matrix *X_train, *X_test, *y_train, *y_test;
    train_test_split(csv->data, csv->labels, 0.2, &X_train, &y_train, &X_test, &y_test);

    KMeans* model = kmeans_create(k);
    kmeans_fit(model, X_train, max_iter, tol);
    Matrix* labels = kmeans_predict(model, X_test);

    double inertia = kmeans_inertia(model, X_test);
    printf("K-Means completado. Inercia: %.4f\n", inertia);

    if (y_test) {
        printf("Evaluación tentativa con etiquetas:\n");
        printf("Accuracy : %.4f\n", accuracy_score(y_test, labels));
        printf("Precision: %.4f\n", precision_score(y_test, labels));
        printf("Recall   : %.4f\n", recall_score(y_test, labels));
        printf("F1 Score : %.4f\n", f1_score(y_test, labels));
    }

    int* counts = calloc(k, sizeof(int));
    for (int i = 0; i < labels->rows; i++) {
        int cluster = (int)labels->data[i][0];
        if (cluster >= 0 && cluster < k) counts[cluster]++;
    }

    printf("Distribución por cluster:\n");
    for (int i = 0; i < k; i++) {
        printf("Cluster %d: %d muestras\n", i, counts[i]);
    }
    free(counts);

    write_csv(output_file, labels);

    matrix_free(labels);
    kmeans_free(model);
    matrix_free(X_train); matrix_free(X_test);
    if (y_train) matrix_free(y_train);
    if (y_test) matrix_free(y_test);
    csv_free(csv);
}

// ==================== Linear Regression ====================
void run_linear(int argc, char* argv[]) {
    char* file = NULL;
    char* feature_str = NULL;
    int target_col = -1;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0) file = argv[++i];
        else if (strcmp(argv[i], "--features") == 0) feature_str = argv[++i];
        else if (strcmp(argv[i], "--target") == 0) target_col = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Uso: ./mlcli linear --train data.csv --features 0,1,2 --target 3\n");
            return;
        }
    }

    if (!file || !feature_str || target_col < 0) {
        fprintf(stderr, "❌ Faltan argumentos requeridos. Usa --help.\n");
        return;
    }

    CSVData* csv = csv_read(file, 1, -1, ',');
    if (!csv || !csv->data) {
        fprintf(stderr, "❌ Error al leer el archivo CSV: %s\n", file);
        return;
    }

    int num_cols = csv->data->cols;

    // Contar cuántos features vienen en la cadena
    int feature_count = 1;
    for (char* p = feature_str; *p; p++) {
        if (*p == ',') feature_count++;
    }

    int* feature_indices = malloc(feature_count * sizeof(int));
    char* token = strtok(feature_str, ",");
    for (int i = 0; i < feature_count && token; i++) {
        int col = atoi(token);
        if (col < 0 || col >= num_cols) {
            fprintf(stderr, "❌ Índice de columna inválido en --features: %d\n", col);
            free(feature_indices);
            csv_free(csv);
            return;
        }
        feature_indices[i] = col;
        token = strtok(NULL, ",");
    }

    if (target_col >= num_cols) {
        fprintf(stderr, "❌ Índice de columna de salida (--target) fuera de rango: %d\n", target_col);
        free(feature_indices);
        csv_free(csv);
        return;
    }

    // Crear matrices X e y con las columnas seleccionadas
    int rows = csv->data->rows;
    Matrix* X = matrix_create(rows, feature_count);
    Matrix* y = matrix_create(rows, 1);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < feature_count; j++) {
            X->data[i][j] = csv->data->data[i][feature_indices[j]];
        }
        y->data[i][0] = csv->data->data[i][target_col];
    }

    free(feature_indices);

    Matrix *X_train, *y_train, *X_test, *y_test;
    if (!train_test_split(X, y, 0.2, &X_train, &y_train, &X_test, &y_test)) {
        fprintf(stderr, "❌ train_test_split falló.\n");
        return;
    }

    LinearRegression* model = linear_regression_create(0.01, 1000, 1e-6);
    linear_regression_fit(model, X_train, y_train);

    Matrix* y_pred = linear_regression_predict(model, X_test);

    printf("\n--- Métricas para Regresión Lineal ---\n");
    printf("MSE : %.4f\n", mean_squared_error(y_test, y_pred));
    printf("MAE : %.4f\n", mean_absolute_error(y_test, y_pred));
    printf("R2  : %.4f\n", r2_score(y_test, y_pred));

    linear_regression_free(model);
    matrix_free(X); matrix_free(y);
    matrix_free(X_train); matrix_free(y_train);
    matrix_free(X_test); matrix_free(y_test);
    matrix_free(y_pred);
    csv_free(csv);
}


// ==================== Métricas ====================
void run_metrics(int argc, char* argv[]) {
    char *pred_file = NULL, *true_file = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--pred") == 0) pred_file = argv[++i];
        else if (strcmp(argv[i], "--true") == 0) true_file = argv[++i];
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Uso: ./mlcli metrics --pred y_pred.csv --true y_true.csv\n");
            return;
        }
    }

    if (!pred_file || !true_file) {
        fprintf(stderr, "Faltan archivos para métricas. Usa --help.\n");
        return;
    }

    Matrix* y_pred = read_csv(pred_file);
    Matrix* y_true = read_csv(true_file);

    printf("Accuracy : %.4f\n", accuracy_score(y_true, y_pred));
    printf("Precision: %.4f\n", precision_score(y_true, y_pred));
    printf("Recall   : %.4f\n", recall_score(y_true, y_pred));
    printf("F1 Score : %.4f\n", f1_score(y_true, y_pred));

    matrix_free(y_pred);
    matrix_free(y_true);
}

// ==================== MAIN ====================
int main(int argc, char* argv[]) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_general_help();
        return 0;
    }

    if (strcmp(argv[1], "knn") == 0) run_knn(argc, argv);
    else if (strcmp(argv[1], "kmeans") == 0) run_kmeans(argc, argv);
    else if (strcmp(argv[1], "linear") == 0) run_linear(argc, argv);
    else if (strcmp(argv[1], "metrics") == 0) run_metrics(argc, argv);
    else {
        fprintf(stderr, "Comando no reconocido: %s\n", argv[1]);
        print_general_help();
        return 1;
    }

    return 0;
}

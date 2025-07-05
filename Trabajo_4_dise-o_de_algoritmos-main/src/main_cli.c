#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core/matrix.h"
#include "utils/csv_reader.h"
#include "algorithms/knn.h"
#include "algorithms/kmeans.h"
#include "algorithms/linear_regression.h"
#include "utils/metrics.h"

void print_general_help() {
    printf("=== ML CLI - Interfaz de Línea de Comandos ===\n");
    printf("Uso: ./mlcli <comando> [opciones]\n\n");
    printf("Comandos disponibles:\n");
    printf("  knn      Clasificador K-Nearest Neighbors\n");
    printf("  kmeans   Agrupamiento con K-Means\n");
    printf("  linear   Regresión lineal (normal, ridge, lasso)\n");
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
        }
        else if (strcmp(argv[i], "--kdtree") == 0) use_kdtree = 1;
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Uso: ./mlcli knn --file data/iris.csv --k 3 --distance euclidean --output pred.csv\n");
            return;
        }
    }

    if (!file || !output_file) {
        fprintf(stderr, "Faltan argumentos. Usa --help.\n");
        return;
    }

    CSVData* csv = csv_read(file, 0, -1, ',');  // Supone que la última columna es la etiqueta
    if (!csv || !csv->data) {
        fprintf(stderr, "Error al leer el archivo CSV: %s\n", file);
        return;
    }

    int last_col = csv->data->cols - 1;

    // Separar features y etiquetas
    Matrix* X_all = matrix_create(csv->data->rows, last_col);
    Matrix* y_all = matrix_create(csv->data->rows, 1);

    for (int i = 0; i < csv->data->rows; i++) {
        for (int j = 0; j < last_col; j++) {
            X_all->data[i][j] = csv->data->data[i][j];
        }
        y_all->data[i][0] = csv->data->data[i][last_col];
    }

    Matrix *X_train, *y_train, *X_test, *y_test;
    if (!train_test_split(X_all, y_all, 0.2, &X_train, &y_train, &X_test, &y_test)) {
        fprintf(stderr, "❌ Error en train_test_split\n");
        return;
    }

    KNNClassifier* knn = knn_create(k);
    knn_set_use_kdtree(knn, use_kdtree);
    knn_set_distance_metric(knn, metric);
    knn_fit(knn, X_train, y_train);

    Matrix* y_pred = knn_predict(knn, X_test);
    write_csv(output_file, y_pred);

    // Evaluación
    printf("KNN completado. Resultados:\n");
    printf("Accuracy : %.4f\n", accuracy_score(y_test, y_pred));
    printf("Precision: %.4f\n", precision_score(y_test, y_pred));
    printf("Recall   : %.4f\n", recall_score(y_test, y_pred));
    printf("F1 Score : %.4f\n", f1_score(y_test, y_pred));

    // Limpieza
    knn_free(knn);
    matrix_free(X_all); matrix_free(y_all);
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

    // Parseo de argumentos
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            file = argv[++i];
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--max_iter") == 0 && i + 1 < argc) {
            max_iter = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tol") == 0 && i + 1 < argc) {
            tol = atof(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Uso: %s kmeans --file data.csv --k 3 --max_iter 200 --tol 0.0001 --output out.csv\n", argv[0]);
            return;
        }
    }

    if (!file || !output_file) {
        fprintf(stderr, "Falta el archivo de entrada o salida. Usa --help.\n");
        return;
    }

    // Leer datos con etiquetas
    CSVData* csv = csv_read(file, 1, -1, ',');
    if (!csv) {
        fprintf(stderr, "Error al leer el archivo: %s\n", file);
        return;
    }

    Matrix *X_train = NULL, *X_test = NULL;
    Matrix *y_train = NULL, *y_test = NULL;

    // División 80/20
    if (!train_test_split(csv->data, csv->labels, 0.2, &X_train, &y_train, &X_test, &y_test)) {
        fprintf(stderr, "Error al dividir los datos.\n");
        csv_free(csv);
        return;
    }

    // Entrenar modelo
    KMeans* model = kmeans_create(k);
    kmeans_fit(model, X_train, max_iter, tol);

    // Predecir en test
    Matrix* labels = kmeans_predict(model, X_test);

    // Inercia
    double inertia = kmeans_inertia(model, X_test);
    printf("K-Means completado. Inercia: %.4f\n", inertia);

    // Calcular métricas si se tienen etiquetas
    if (y_test) {
        printf("\n--- Evaluación Tentativa (si las etiquetas son reales) ---\n");
        printf("Accuracy : %.4f\n", accuracy_score(y_test, labels));
        printf("Precision: %.4f\n", precision_score(y_test, labels));
        printf("Recall   : %.4f\n", recall_score(y_test, labels));
        printf("F1 Score : %.4f\n", f1_score(y_test, labels));
    }

    // Contar cantidad por cluster
    int* counts = calloc(k, sizeof(int));
    for (int i = 0; i < labels->rows; i++) {
        int cluster = (int)labels->data[i][0];
        if (cluster >= 0 && cluster < k) counts[cluster]++;
    }

    printf("\nDistribución por cluster:\n");
    for (int i = 0; i < k; i++) {
        printf("Cluster %d: %d muestras\n", i, counts[i]);
    }
    free(counts);

    // Guardar resultados
    write_csv(output_file, labels);

    // Liberar memoria
    matrix_free(labels);
    kmeans_free(model);
    matrix_free(X_train); matrix_free(X_test);
    if (y_train) matrix_free(y_train);
    if (y_test) matrix_free(y_test);
    csv_free(csv);
}




// ==================== Linear Regression ====================
void run_linear(int argc, char* argv[]) {
    char* train_file = NULL;
    char* type = "normal";
    double lambda = 0.0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--train") == 0) train_file = argv[++i];
        else if (strcmp(argv[i], "--type") == 0) type = argv[++i];
        else if (strcmp(argv[i], "--lambda") == 0) lambda = atof(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Uso: ./mlcli linear --train data.csv --type ridge --lambda 0.5\n");
            return;
        }
    }

    if (!train_file) {
        fprintf(stderr, "Falta el archivo de entrenamiento para regresión. Usa --help.\n");
        return;
    }

    Matrix *X, *y;
    read_csv_xy(train_file, &X, &y);
    LinearRegression* model = linear_regression_create(0.01, 1000, 1e-6);

    if (strcmp(type, "ridge") == 0)
        linear_regression_fit_ridge(model, X, y, lambda);
    else if (strcmp(type, "lasso") == 0)
        linear_regression_fit_lasso(model, X, y, lambda);
    else
        linear_regression_fit(model, X, y);

    printf("Modelo ajustado. Coef: %.4f | Bias: %.4f\n",
           model->weights->data[0][0], model->bias);

    linear_regression_free(model);
    matrix_free(X);
    matrix_free(y);
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

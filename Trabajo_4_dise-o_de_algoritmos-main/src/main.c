#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "algorithms/knn.h"
#include "algorithms/kmeans.h"
#include "algorithms/linear_regression.h"
#include "core/matrix.h"
#include "utils/csv_reader.h"

// Función para generar datos de ejemplo
Matrix* generate_sample_data(int n_samples, int n_features, int n_clusters) {
    Matrix* data = matrix_create(n_samples, n_features);
    if (!data) return NULL;
    
    // Inicializar semilla aleatoria
    srand(time(NULL));
    
    // Generar puntos alrededor de centros de clusters
    for (int i = 0; i < n_samples; i++) {
        int cluster = rand() % n_clusters;
        double center_x = (cluster + 1) * 5.0;
        double center_y = (cluster + 1) * 5.0;
        
        data->data[i][0] = center_x + ((rand() % 1000) / 1000.0 - 0.5) * 3.0;
        data->data[i][1] = center_y + ((rand() % 1000) / 1000.0 - 0.5) * 3.0;
    }
    
    return data;
}

void test_knn() {
    printf("\n=== Prueba de K-Vecinos Más Cercanos ===\n");
    
    // Generar datos de prueba
    int n_samples = 100;
    int n_features = 2;
    int n_classes = 3;
    
    Matrix* X = generate_sample_data(n_samples, n_features, n_classes);
    Matrix* y = matrix_create(n_samples, 1);
    
    // Asignar etiquetas basadas en la proximidad a centros
    for (int i = 0; i < n_samples; i++) {
        int closest_class = 0;
        double min_dist = 1000.0;
        
        for (int c = 0; c < n_classes; c++) {
            double center_x = (c + 1) * 5.0;
            double center_y = (c + 1) * 5.0;
            
            double dx = X->data[i][0] - center_x;
            double dy = X->data[i][1] - center_y;
            double dist = dx*dx + dy*dy;
            
            if (dist < min_dist) {
                min_dist = dist;
                closest_class = c;
            }
        }
        
        y->data[i][0] = closest_class;
    }
    
    // Crear y entrenar modelo KNN
    KNNClassifier* knn = knn_create(3);
    if (knn) {
        knn_fit(knn, X, y);
        
        // Predecir para los mismos datos (solo para prueba)
        Matrix* y_pred = knn_predict(knn, X);
        
        if (y_pred) {
            // Calcular precisión
            int correct = 0;
            for (int i = 0; i < n_samples; i++) {
                if (y_pred->data[i][0] == y->data[i][0]) {
                    correct++;
                }
            }
            
            printf("Precisión KNN: %.2f%%\n", 100.0 * correct / n_samples);
            
            matrix_free(y_pred);
        }
        
        knn_free(knn);
    }
    
    matrix_free(X);
    matrix_free(y);
}

void test_kmeans() {
    printf("\n=== Prueba de K-Means ===\n");
    
    // Generar datos de prueba
    int n_samples = 100;
    int n_features = 2;
    int n_clusters = 3;
    
    Matrix* X = generate_sample_data(n_samples, n_features, n_clusters);
    
    // Crear y entrenar modelo K-Means
    KMeans* kmeans = kmeans_create(n_clusters);
    if (kmeans) {
        if (kmeans_fit(kmeans, X, 100, 1e-4)) {
            // Predecir clusters
            Matrix* labels = kmeans_predict(kmeans, X);
            
            if (labels) {
                // Mostrar algunos resultados
                printf("Primeros 5 puntos y sus clusters:\n");
                printf("   X       Y     Cluster\n");
                for (int i = 0; i < 5; i++) {
                    printf("%7.2f %7.2f     %d\n", 
                           X->data[i][0], 
                           X->data[i][1], 
                           (int)labels->data[i][0]);
                }
                
                // Calcular inercia
                double inertia = kmeans_inertia(kmeans, X);
                printf("Inercia: %.4f\n", inertia);
                
                matrix_free(labels);
            }
        }
        
        kmeans_free(kmeans);
    }
    
    matrix_free(X);
}

void test_linear_regression() {
    printf("\n=== Prueba de Regresión Lineal ===\n");
    
    // Generar datos de prueba
    int n_samples = 50;
    double slope = 2.5;
    double intercept = 3.0;
    double noise_level = 2.0;
    
    Matrix* X = matrix_create(n_samples, 1);
    Matrix* y = matrix_create(n_samples, 1);
    
    // Inicializar semilla aleatoria
    srand(time(NULL));
    
    // Generar datos con relación lineal y algo de ruido
    for (int i = 0; i < n_samples; i++) {
        X->data[i][0] = i * 0.5;
        y->data[i][0] = slope * X->data[i][0] + intercept + 
                      (((rand() % 1000) / 1000.0) - 0.5) * noise_level;
    }
    
    // Crear y entrenar modelo de regresión lineal
    LinearRegression* model = linear_regression_create(0.01, 1000, 1e-6);
    if (model) {
        if (linear_regression_fit(model, X, y)) {
            // Predecir para los mismos datos
            Matrix* y_pred = linear_regression_predict(model, X);
            
            if (y_pred) {
                // Calcular error cuadrático medio
                double mse = linear_regression_mse(model, X, y);
                double r2 = linear_regression_r2_score(model, X, y);
                
                printf("Coeficiente: %.4f (Valor real: %.4f)\n", 
                       model->weights->data[0][0], slope);
                printf("Intercepto: %.4f (Valor real: %.4f)\n", 
                       model->bias, intercept);
                printf("Error cuadrático medio: %.4f\n", mse);
                printf("Coeficiente R²: %.4f\n", r2);
                
                matrix_free(y_pred);
            }
        }
        
        linear_regression_free(model);
    }
    
    matrix_free(X);
    matrix_free(y);
}

void test_csv_reader() {
    printf("\n=== Prueba de Lectura de CSV ===\n");
    
    // Leer el conjunto de datos Iris
    const char* filename = "data/iris.csv";
    printf("Leyendo archivo: %s\n", filename);
    
    // El archivo tiene encabezado y la columna de etiquetas es la última (índice 4)
    CSVData* csv_data = csv_read(filename, 1, 4, ',');
    
    if (!csv_data) {
        printf("Error al leer el archivo CSV\n");
        return;
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
    
    // Dividir en conjuntos de entrenamiento y prueba
    Matrix *X_train, *y_train, *X_test, *y_test;
    if (train_test_split(csv_data->data, csv_data->labels, 0.2, 
                         &X_train, &y_train, &X_test, &y_test)) {
        
        printf("\nDivisión en conjuntos de entrenamiento y prueba:\n");
        printf("X_train: %d muestras x %d características\n", X_train->rows, X_train->cols);
        printf("X_test: %d muestras x %d características\n", X_test->rows, X_test->cols);
        
        // Liberar memoria
        matrix_free(X_train);
        matrix_free(y_train);
        matrix_free(X_test);
        matrix_free(y_test);
    }
    
    // Liberar memoria
    csv_free(csv_data);
}

int main() {
    printf("Biblioteca de Machine Learning en C\n");
    printf("==================================\n");
    
    // Probar la lectura de CSV primero
    test_csv_reader();
    
    // Probar los algoritmos
    test_knn();
    test_kmeans();
    test_linear_regression();
    
    return 0;
}

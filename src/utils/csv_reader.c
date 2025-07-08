#include "csv_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Implementación propia de strdup para evitar advertencias
static char* my_strdup(const char* s) {
    if (s == NULL) return NULL;
    size_t len = strlen(s) + 1;
    char* new_str = (char*)malloc(len);
    if (new_str == NULL) return NULL;
    return (char*)memcpy(new_str, s, len);
}

#define MAX_LINE_LENGTH 4096
#define MAX_FIELDS 256

// Función auxiliar para contar líneas y columnas en un archivo CSV
static int csv_dimensions(const char* filename, int has_header, char delimiter, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (!file) return 0;
    
    char line[MAX_LINE_LENGTH];
    *rows = 0;
    *cols = 0;
    
    // Leer la primera línea para determinar el número de columnas
    if (fgets(line, MAX_LINE_LENGTH, file)) {
        // Crear una cadena con el delimitador para strtok
        char delim_str[2] = {delimiter, '\0'};
        
        char* token = strtok(line, delim_str);
        while (token) {
            (*cols)++;
            token = strtok(NULL, delim_str);
        }
        
        // Contar el número de filas
        *rows = 1; // Ya leímos la primera línea
        while (fgets(line, MAX_LINE_LENGTH, file)) {
            (*rows)++;
        }
        
        // Si hay encabezado, ajustar el número de filas
        if (has_header) {
            (*rows)--;
        }
    }
    
    fclose(file);
    return 1;
}

CSVData* csv_read(const char* filename, int has_header, int label_col, char delimiter) {
    int rows, cols;
    if (!csv_dimensions(filename, has_header, delimiter, &rows, &cols)) {
        return NULL;
    }
    
    // Crear estructura para almacenar los datos
    CSVData* csv_data = (CSVData*)malloc(sizeof(CSVData));
    if (!csv_data) return NULL;
    
    csv_data->has_header = has_header;
    csv_data->label_col = label_col;
    csv_data->header = NULL;
    
    // Determinar dimensiones de las matrices de datos y etiquetas
    int data_cols = (label_col >= 0) ? cols - 1 : cols;
    
    // Crear matrices para datos y etiquetas
    csv_data->data = matrix_create(rows, data_cols);
    if (!csv_data->data) {
        free(csv_data);
        return NULL;
    }
    
    if (label_col >= 0) {
        csv_data->labels = matrix_create(rows, 1);
        if (!csv_data->labels) {
            matrix_free(csv_data->data);
            free(csv_data);
            return NULL;
        }
    } else {
        csv_data->labels = NULL;
    }
    
    // Si hay encabezado, reservar memoria para los nombres de columnas
    if (has_header) {
        csv_data->header = (char**)malloc(cols * sizeof(char*));
        if (!csv_data->header) {
            if (csv_data->labels) matrix_free(csv_data->labels);
            matrix_free(csv_data->data);
            free(csv_data);
            return NULL;
        }
        
        for (int i = 0; i < cols; i++) {
            csv_data->header[i] = NULL;
        }
    }
    
    // Abrir el archivo para leer los datos
    FILE* file = fopen(filename, "r");
    if (!file) {
        if (csv_data->header) free(csv_data->header);
        if (csv_data->labels) matrix_free(csv_data->labels);
        matrix_free(csv_data->data);
        free(csv_data);
        return NULL;
    }
    
    char line[MAX_LINE_LENGTH];
    char* fields[MAX_FIELDS];
    int row = 0;
    
    // Leer encabezado si existe
    if (has_header && fgets(line, MAX_LINE_LENGTH, file)) {
        // Eliminar el salto de línea
        line[strcspn(line, "\r\n")] = 0;
        
        // Crear una cadena con el delimitador para strtok
        char delim_str[2] = {delimiter, '\0'};
        
        // Dividir la línea en campos
        char* token = strtok(line, delim_str);
        int col = 0;
        
        while (token && col < cols) {
            // Eliminar espacios en blanco al inicio y final
            while (*token == ' ' || *token == '\t') token++;
            char* end = token + strlen(token) - 1;
            while (end > token && (*end == ' ' || *end == '\t')) end--;
            *(end + 1) = 0;
            
            // Guardar el nombre de la columna
            csv_data->header[col] = my_strdup(token);
            token = strtok(NULL, delim_str);
            col++;
        }
    }
    
    // Leer datos
    while (fgets(line, MAX_LINE_LENGTH, file) && row < rows) {
        // Eliminar el salto de línea
        line[strcspn(line, "\r\n")] = 0;
        
        // Crear una cadena con el delimitador para strtok
        char delim_str[2] = {delimiter, '\0'};
        
        // Dividir la línea en campos
        char* token = strtok(line, delim_str);
        int field = 0;
        
        while (token && field < MAX_FIELDS) {
            // Eliminar espacios en blanco al inicio y final
            while (*token == ' ' || *token == '\t') token++;
            char* end = token + strlen(token) - 1;
            while (end > token && (*end == ' ' || *end == '\t')) end--;
            *(end + 1) = 0;
            
            fields[field] = token;
            token = strtok(NULL, delim_str);
            field++;
        }
        
        // Llenar matrices de datos y etiquetas
        int data_col = 0;
        for (int col = 0; col < cols && col < field; col++) {
            if (col == label_col) {
                // Esta columna contiene etiquetas
                if (csv_data->labels) {
                    csv_data->labels->data[row][0] = atof(fields[col]);
                }
            } else {
                // Esta columna contiene datos
                csv_data->data->data[row][data_col] = atof(fields[col]);
                data_col++;
            }
        }
        
        row++;
    }
    
    fclose(file);
    return csv_data;
}

void csv_free(CSVData* csv_data) {
    if (!csv_data) return;
    
    if (csv_data->data) {
        matrix_free(csv_data->data);
    }
    
    if (csv_data->labels) {
        matrix_free(csv_data->labels);
    }
    
    if (csv_data->header) {
        // Calcular el número total de columnas (datos + etiqueta si existe)
        int total_cols = csv_data->data->cols;
        if (csv_data->label_col >= 0) total_cols++;
        
        // Liberar cada cadena de encabezado
        for (int i = 0; i < total_cols; i++) {
            if (csv_data->header[i]) {
                free(csv_data->header[i]);
                csv_data->header[i] = NULL;
            }
        }
        
        // Liberar el array de punteros
        free(csv_data->header);
        csv_data->header = NULL;
    }
    
    free(csv_data);
}

int train_test_split(Matrix* data, Matrix* labels, double test_ratio,
                    Matrix** X_train, Matrix** y_train, 
                    Matrix** X_test, Matrix** y_test) {
    if (!data || test_ratio < 0.0 || test_ratio > 1.0) return 0;
    
    int n_samples = data->rows;
    int n_features = data->cols;
    
    // Calcular tamaños de conjuntos de entrenamiento y prueba
    int test_size = (int)(test_ratio * n_samples);
    int train_size = n_samples - test_size;
    
    if (train_size <= 0 || test_size <= 0) return 0;
    
    // Crear matrices para los conjuntos de entrenamiento y prueba
    *X_train = matrix_create(train_size, n_features);
    *X_test = matrix_create(test_size, n_features);
    
    if (!*X_train || !*X_test) {
        if (*X_train) matrix_free(*X_train);
        if (*X_test) matrix_free(*X_test);
        return 0;
    }
    
    // Si hay etiquetas, crear matrices para ellas también
    if (labels) {
        *y_train = matrix_create(train_size, 1);
        *y_test = matrix_create(test_size, 1);
        
        if (!*y_train || !*y_test) {
            matrix_free(*X_train);
            matrix_free(*X_test);
            if (*y_train) matrix_free(*y_train);
            if (*y_test) matrix_free(*y_test);
            return 0;
        }
    } else {
        *y_train = NULL;
        *y_test = NULL;
    }
    
    // Crear un arreglo de índices y mezclarlo aleatoriamente
    int* indices = (int*)malloc(n_samples * sizeof(int));
    if (!indices) {
        matrix_free(*X_train);
        matrix_free(*X_test);
        if (*y_train) matrix_free(*y_train);
        if (*y_test) matrix_free(*y_test);
        return 0;
    }
    
    for (int i = 0; i < n_samples; i++) {
        indices[i] = i;
    }
    
    // Mezclar índices (algoritmo de Fisher-Yates)
    srand(time(NULL));
    for (int i = n_samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    
    // Llenar conjuntos de entrenamiento y prueba
    for (int i = 0; i < train_size; i++) {
        int idx = indices[i];
        for (int j = 0; j < n_features; j++) {
            (*X_train)->data[i][j] = data->data[idx][j];
        }
        
        if (labels && *y_train) {
            (*y_train)->data[i][0] = labels->data[idx][0];
        }
    }
    
    for (int i = 0; i < test_size; i++) {
        int idx = indices[train_size + i];
        for (int j = 0; j < n_features; j++) {
            (*X_test)->data[i][j] = data->data[idx][j];
        }
        
        if (labels && *y_test) {
            (*y_test)->data[i][0] = labels->data[idx][0];
        }
    }
    
    free(indices);
    return 1;
}

// Lectura simple: sólo datos
Matrix* read_csv(const char* filename) {
    CSVData* data = csv_read(filename, 0, -1, ',');
    if (!data) return NULL;
    Matrix* m = data->data;
    data->data = NULL;  // Transferimos propiedad
    csv_free(data);
    return m;
}

// Lectura de datos + etiquetas
void read_csv_xy(const char* filename, Matrix** X, Matrix** y) {
    CSVData* data = csv_read(filename, 0, -1, ',');
    if (!data) return;
    *X = data->data;
    *y = data->labels;
    data->data = NULL;
    data->labels = NULL;
    csv_free(data);
}

// Guardar matriz como CSV
void write_csv(const char* filename, Matrix* m) {
    FILE* f = fopen(filename, "w");
    if (!f) return;
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            fprintf(f, "%g", m->data[i][j]);
            if (j < m->cols - 1) fprintf(f, ",");
        }
        fprintf(f, "\n");
    }
    fclose(f);
}
#ifndef CSV_READER_H
#define CSV_READER_H

#include "../core/matrix.h"

/**
 * @brief Estructura para almacenar datos leídos de un archivo CSV
 */
typedef struct {
    Matrix* data;    // Matriz con los datos numéricos
    Matrix* labels;  // Vector con las etiquetas (opcional)
    char** header;   // Nombres de las columnas (opcional)
    int has_header;  // Indica si el CSV tenía encabezado
    int label_col;   // Índice de la columna de etiquetas (-1 si no hay)
} CSVData;

/**
 * @brief Lee un archivo CSV y devuelve los datos como matrices
 * @param filename Nombre del archivo CSV a leer
 * @param has_header Indica si el archivo tiene encabezado (1) o no (0)
 * @param label_col Índice de la columna que contiene las etiquetas (-1 si no hay)
 * @param delimiter Carácter delimitador (normalmente ',' o ';')
 * @return CSVData* Estructura con los datos leídos o NULL si hay error
 */
CSVData* csv_read(const char* filename, int has_header, int label_col, char delimiter);

/**
 * @brief Libera la memoria utilizada por la estructura CSVData
 * @param csv_data Estructura a liberar
 */
void csv_free(CSVData* csv_data);

/**
 * @brief Divide los datos en conjuntos de entrenamiento y prueba
 * @param data Datos originales
 * @param labels Etiquetas originales (puede ser NULL)
 * @param test_ratio Proporción de datos para prueba (0.0-1.0)
 * @param X_train Puntero donde se almacenará la matriz de características de entrenamiento
 * @param y_train Puntero donde se almacenará la matriz de etiquetas de entrenamiento (puede ser NULL)
 * @param X_test Puntero donde se almacenará la matriz de características de prueba
 * @param y_test Puntero donde se almacenará la matriz de etiquetas de prueba (puede ser NULL)
 * @return int 1 si éxito, 0 si error
 */
int train_test_split(Matrix* data, Matrix* labels, double test_ratio,
                    Matrix** X_train, Matrix** y_train, 
                    Matrix** X_test, Matrix** y_test);

/**
 * @brief Lee un archivo CSV sin etiquetas y devuelve una matriz con los datos
 * @param filename Nombre del archivo CSV
 * @return Matrix* Matriz de datos leída (n x m), o NULL si hay error
 */
Matrix* read_csv(const char* filename);

/**
 * @brief Lee un archivo CSV que contiene características y etiquetas
 * @param filename Nombre del archivo CSV
 * @param X Puntero donde se almacenará la matriz de características (n x m)
 * @param y Puntero donde se almacenará la matriz de etiquetas (n x 1)
 */
void read_csv_xy(const char* filename, Matrix** X, Matrix** y);

/**
 * @brief Guarda una matriz en un archivo CSV
 * @param filename Nombre del archivo de salida
 * @param m Matriz a guardar (n x m)
 */
void write_csv(const char* filename, Matrix* m);

#endif // CSV_READER_H

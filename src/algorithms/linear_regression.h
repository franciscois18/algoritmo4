#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include "../core/matrix.h"

/**
 * @brief Estructura para el Modelo de Regresión Lineal
 */
typedef struct {
    Matrix* weights;     // Coeficientes del modelo
    double bias;        // Término independiente
    double learning_rate;    // Tasa de aprendizaje
    int max_iterations;      // Número máximo de iteraciones
    double tolerance;        // Tolerancia para convergencia
} LinearRegression;

/**
 * @brief Crea un nuevo modelo de Regresión Lineal
 * @param learning_rate Tasa de aprendizaje para descenso de gradiente
 * @param max_iterations Número máximo de iteraciones
 * @param tolerance Tolerancia para convergencia
 * @return LinearRegression* Puntero al modelo o NULL si hay error
 */
LinearRegression* linear_regression_create(double learning_rate, int max_iterations, double tolerance);

/**
 * @brief Entrena el modelo de Regresión Lineal con los datos proporcionados
 * @param model Modelo a entrenar
 * @param X Matriz de características
 * @param y Vector de valores objetivo
 * @return int 1 si éxito, 0 si error
 */
int linear_regression_fit(LinearRegression* model, Matrix* X, Matrix* y);

/**
 * @brief Realiza predicciones con el modelo entrenado
 * @param model Modelo entrenado
 * @param X Matriz de características a predecir
 * @return Matrix* Vector de predicciones o NULL si hay error
 */
Matrix* linear_regression_predict(LinearRegression* model, Matrix* X);

/**
 * @brief Calcula el error cuadrático medio (MSE)
 * @param model Modelo entrenado
 * @param X Matriz de características
 * @param y Vector de valores reales
 * @return double Valor de MSE o -1 si hay error
 */
double linear_regression_mse(LinearRegression* model, Matrix* X, Matrix* y);

/**
 * @brief Calcula el coeficiente de determinación R²
 * @param model Modelo entrenado
 * @param X Matriz de características
 * @param y Vector de valores reales
 * @return double Valor de R² o -1 si hay error
 */
double linear_regression_r2_score(LinearRegression* model, Matrix* X, Matrix* y);

/**
 * @brief Libera la memoria utilizada por el modelo
 * @param model Modelo a liberar
 */
void linear_regression_free(LinearRegression* model);

double linear_regression_mae(LinearRegression* model, Matrix* X, Matrix* y);
int linear_regression_fit_ridge(LinearRegression* model, Matrix* X, Matrix* y, double lambda);
int linear_regression_fit_lasso(LinearRegression* model, Matrix* X, Matrix* y, double lambda);
int linear_regression_fit_closed_form(LinearRegression* model, Matrix* X, Matrix* y);
int linear_regression_save(LinearRegression* model, const char* filename);
LinearRegression* linear_regression_load(const char* filename);



#endif // LINEAR_REGRESSION_H

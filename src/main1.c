#include <stdio.h>
#include <stdlib.h>
#include "core/matrix.h"
#include "algorithms/linear_regression.h"

void evaluar_modelo(const char* nombre, LinearRegression* model, Matrix* X, Matrix* y) {
    double mse = linear_regression_mse(model, X, y);
    double r2 = linear_regression_r2_score(model, X, y);
    double mae = linear_regression_mae(model, X, y);

    printf("\n--- %s ---\n", nombre);
    printf("Coeficiente: %.4f | Bias: %.4f\n", model->weights->data[0][0], model->bias);
    printf("MSE: %.4f | R²: %.4f | MAE: %.4f\n", mse, r2, mae);
}

int main() {
    printf("=== Prueba Unitaria: Regresión Lineal Extendida ===\n");

    // Crear dataset simulado: y = 2x + 3
    int n = 10;
    Matrix* X = matrix_create(n, 1);
    Matrix* y = matrix_create(n, 1);
    for (int i = 0; i < n; i++) {
        X->data[i][0] = i;
        y->data[i][0] = 2.0 * i + 3.0;
    }

    // ========== GD ==========
    LinearRegression* model_gd = linear_regression_create(0.01, 1000, 1e-6);
    linear_regression_fit(model_gd, X, y);
    evaluar_modelo("Descenso por Gradiente", model_gd, X, y);

    // ========== Closed-Form ==========
    LinearRegression* model_closed = linear_regression_create(0.01, 1000, 1e-6);
    if (linear_regression_fit_closed_form(model_closed, X, y)) {
        evaluar_modelo("Ecuaciones Normales (Closed-form)", model_closed, X, y);
    } else {
        printf("❌ No se pudo resolver por ecuaciones normales (posible matriz no invertible).\n");
    }

    // ========== RIDGE ==========
    LinearRegression* model_ridge = linear_regression_create(0.01, 1000, 1e-6);
    linear_regression_fit_ridge(model_ridge, X, y, 0.5);
    evaluar_modelo("Regresión Ridge (λ=0.5)", model_ridge, X, y);

    // ========== LASSO ==========
    LinearRegression* model_lasso = linear_regression_create(0.01, 1000, 1e-6);
    linear_regression_fit_lasso(model_lasso, X, y, 0.5);
    evaluar_modelo("Regresión Lasso (λ=0.5)", model_lasso, X, y);

    // ========== SERIALIZACIÓN ==========
    printf("\n--- Guardar y Cargar modelo (GD) ---\n");
    const char* path = "modelo_lr_gd.dat";
    if (linear_regression_save(model_gd, path)) {
        printf("Modelo guardado correctamente en %s\n", path);
    } else {
        printf("❌ Error al guardar el modelo.\n");
    }

    LinearRegression* model_loaded = linear_regression_load(path);
    if (model_loaded) {
        printf("Modelo cargado correctamente.\n");
        evaluar_modelo("Modelo Cargado", model_loaded, X, y);
    } else {
        printf("❌ Error al cargar el modelo desde archivo.\n");
    }

    // ========== Comparación visual ==========
    printf("\n--- Comparación Real vs Predicción (GD) ---\n");
    Matrix* y_pred = linear_regression_predict(model_gd, X);
    for (int i = 0; i < n; i++) {
        printf("X=%.1f, Real=%.1f, Pred=%.2f\n",
               X->data[i][0], y->data[i][0], y_pred->data[i][0]);
    }

    // Limpieza
    matrix_free(X);
    matrix_free(y);
    matrix_free(y_pred);
    linear_regression_free(model_gd);
    linear_regression_free(model_closed);
    linear_regression_free(model_ridge);
    linear_regression_free(model_lasso);
    linear_regression_free(model_loaded);

    return 0;
}

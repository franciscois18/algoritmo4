#include "linear_regression.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

LinearRegression* linear_regression_create(double learning_rate, int max_iterations, double tolerance) {
    LinearRegression* model = malloc(sizeof(LinearRegression));
    if (!model) return NULL;

    model->weights = NULL;
    model->bias = 0.0;
    model->learning_rate = learning_rate;
    model->max_iterations = max_iterations;
    model->tolerance = tolerance;

    return model;
}

int linear_regression_fit(LinearRegression* model, Matrix* X, Matrix* y) {
    int n = X->rows;
    int m = X->cols;

    // Inicializar pesos en 0
    model->weights = matrix_create(m, 1);
    if (!model->weights) return 0;
    for (int j = 0; j < m; j++) {
        model->weights->data[j][0] = 0.0;
    }

    model->bias = 0.0;

    for (int iter = 0; iter < model->max_iterations; iter++) {
        // y_pred = X * weights + bias
        Matrix* y_pred = matrix_create(n, 1);
        for (int i = 0; i < n; i++) {
            double dot = 0.0;
            for (int j = 0; j < m; j++) {
                dot += X->data[i][j] * model->weights->data[j][0];
            }
            y_pred->data[i][0] = dot + model->bias;
        }

        // Gradientes
        double grad_b = 0.0;
        for (int j = 0; j < m; j++) {
            double grad_w = 0.0;
            for (int i = 0; i < n; i++) {
                double error = y_pred->data[i][0] - y->data[i][0];
                grad_w += error * X->data[i][j];
                if (j == 0) grad_b += error;
            }
            grad_w /= n;
            model->weights->data[j][0] -= model->learning_rate * grad_w;
        }

        grad_b /= n;
        model->bias -= model->learning_rate * grad_b;

        matrix_free(y_pred);
    }

    return 1;
}

Matrix* linear_regression_predict(LinearRegression* model, Matrix* X) {
    int n = X->rows;
    int m = X->cols;

    Matrix* y_pred = matrix_create(n, 1);
    if (!y_pred) return NULL;

    for (int i = 0; i < n; i++) {
        double dot = 0.0;
        for (int j = 0; j < m; j++) {
            dot += X->data[i][j] * model->weights->data[j][0];
        }
        y_pred->data[i][0] = dot + model->bias;
    }

    return y_pred;
}

double linear_regression_mse(LinearRegression* model, Matrix* X, Matrix* y) {
    Matrix* y_pred = linear_regression_predict(model, X);
    if (!y_pred) return -1;

    double mse = 0.0;
    for (int i = 0; i < y->rows; i++) {
        double diff = y_pred->data[i][0] - y->data[i][0];
        mse += diff * diff;
    }
    matrix_free(y_pred);
    return mse / y->rows;
}

double linear_regression_r2_score(LinearRegression* model, Matrix* X, Matrix* y) {
    Matrix* y_pred = linear_regression_predict(model, X);
    if (!y_pred) return -1;

    double ss_res = 0.0;
    double ss_tot = 0.0;
    double mean_y = 0.0;

    for (int i = 0; i < y->rows; i++) {
        mean_y += y->data[i][0];
    }
    mean_y /= y->rows;

    for (int i = 0; i < y->rows; i++) {
        double diff = y_pred->data[i][0] - y->data[i][0];
        ss_res += diff * diff;

        double tot = y->data[i][0] - mean_y;
        ss_tot += tot * tot;
    }

    matrix_free(y_pred);
    return 1.0 - (ss_res / ss_tot);
}

void linear_regression_free(LinearRegression* model) {
    if (model) {
        if (model->weights) matrix_free(model->weights);
        free(model);
    }
}

int linear_regression_fit_ridge(LinearRegression* model, Matrix* X, Matrix* y, double lambda) {
    int n = X->rows;
    int m = X->cols;

    model->weights = matrix_create(m, 1);
    if (!model->weights) return 0;

    for (int j = 0; j < m; j++) {
        model->weights->data[j][0] = 0.0;
    }
    model->bias = 0.0;

    for (int iter = 0; iter < model->max_iterations; iter++) {
        // y_pred = X * weights + bias
        Matrix* y_pred = matrix_create(n, 1);
        for (int i = 0; i < n; i++) {
            double dot = 0.0;
            for (int j = 0; j < m; j++) {
                dot += X->data[i][j] * model->weights->data[j][0];
            }
            y_pred->data[i][0] = dot + model->bias;
        }

        // Gradientes
        double grad_b = 0.0;
        for (int j = 0; j < m; j++) {
            double grad_w = 0.0;
            for (int i = 0; i < n; i++) {
                double error = y_pred->data[i][0] - y->data[i][0];
                grad_w += error * X->data[i][j];
                if (j == 0) grad_b += error;
            }
            grad_w /= n;

            // Ridge penalty (L2)
            grad_w += lambda * model->weights->data[j][0];

            model->weights->data[j][0] -= model->learning_rate * grad_w;
        }

        grad_b /= n;
        model->bias -= model->learning_rate * grad_b;

        matrix_free(y_pred);
    }

    return 1;
}

int linear_regression_fit_lasso(LinearRegression* model, Matrix* X, Matrix* y, double lambda) {
    int n = X->rows;
    int m = X->cols;

    model->weights = matrix_create(m, 1);
    if (!model->weights) return 0;
    for (int j = 0; j < m; j++) {
        model->weights->data[j][0] = 0.0;
    }

    model->bias = 0.0;

    for (int iter = 0; iter < model->max_iterations; iter++) {
        Matrix* y_pred = matrix_create(n, 1);
        for (int i = 0; i < n; i++) {
            double dot = 0.0;
            for (int j = 0; j < m; j++) {
                dot += X->data[i][j] * model->weights->data[j][0];
            }
            y_pred->data[i][0] = dot + model->bias;
        }

        double grad_b = 0.0;
        for (int j = 0; j < m; j++) {
            double grad_w = 0.0;
            for (int i = 0; i < n; i++) {
                double error = y_pred->data[i][0] - y->data[i][0];
                grad_w += error * X->data[i][j];
                if (j == 0) grad_b += error;
            }
            grad_w /= n;

            grad_w += lambda * (model->weights->data[j][0] >= 0 ? 1 : -1);

            model->weights->data[j][0] -= model->learning_rate * grad_w;
        }

        grad_b /= n;
        model->bias -= model->learning_rate * grad_b;

        matrix_free(y_pred);
    }

    return 1;
}

double linear_regression_mae(LinearRegression* model, Matrix* X, Matrix* y) {
    if (!model || !X || !y || X->rows != y->rows) return -1;

    Matrix* y_pred = linear_regression_predict(model, X);
    if (!y_pred) return -1;

    double error_sum = 0.0;
    for (int i = 0; i < y->rows; i++) {
        double pred = y_pred->data[i][0];
        double actual = y->data[i][0];
        double abs_error = fabs(pred - actual);
        error_sum += abs_error;

        // Añadido para depuración:
        printf("  [%2d] y=%.2f, ŷ=%.2f, |error|=%.4f\n", i, actual, pred, abs_error);
    }

    matrix_free(y_pred);
    return error_sum / y->rows;
}

int linear_regression_fit_closed_form(LinearRegression* model, Matrix* X, Matrix* y) {
    // X^T * X
    Matrix* Xt = matrix_transpose(X);
    Matrix* XtX = matrix_dot(Xt, X);
    if (!XtX) return 0;

    // Inversa de X^T * X
    Matrix* XtX_inv = matrix_inverse(XtX);
    if (!XtX_inv) {
        matrix_free(Xt); matrix_free(XtX);
        return 0;
    }

    // X^T * y
    Matrix* Xty = matrix_dot(Xt, y);
    if (!Xty) {
        matrix_free(Xt); matrix_free(XtX); matrix_free(XtX_inv);
        return 0;
    }

    // weights = (X^T X)^-1 X^T y
    if (model->weights) matrix_free(model->weights);
    model->weights = matrix_dot(XtX_inv, Xty);
    model->bias = 0.0;

    matrix_free(Xt);
    matrix_free(XtX);
    matrix_free(XtX_inv);
    matrix_free(Xty);
    return 1;
}

int linear_regression_save(LinearRegression* model, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return 0;

    fwrite(&model->bias, sizeof(double), 1, f);
    fwrite(&model->learning_rate, sizeof(double), 1, f);
    fwrite(&model->max_iterations, sizeof(int), 1, f);
    fwrite(&model->tolerance, sizeof(double), 1, f);

    fwrite(&model->weights->rows, sizeof(int), 1, f);
    fwrite(&model->weights->cols, sizeof(int), 1, f);
    for (int i = 0; i < model->weights->rows; i++) {
        fwrite(model->weights->data[i], sizeof(double), model->weights->cols, f);
    }

    fclose(f);
    return 1;
}

LinearRegression* linear_regression_load(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    LinearRegression* model = malloc(sizeof(LinearRegression));
    if (!model) {
        fclose(f);
        return NULL;
    }
    
    // Inicializar punteros a NULL
    model->weights = NULL;

    if (fread(&model->bias, sizeof(double), 1, f) != 1) { free(model); fclose(f); return NULL; }
    if (fread(&model->learning_rate, sizeof(double), 1, f) != 1) { free(model); fclose(f); return NULL; }
    if (fread(&model->max_iterations, sizeof(int), 1, f) != 1) { free(model); fclose(f); return NULL; }
    if (fread(&model->tolerance, sizeof(double), 1, f) != 1) { free(model); fclose(f); return NULL; }

    int rows, cols;
    if (fread(&rows, sizeof(int), 1, f) != 1) { free(model); fclose(f); return NULL; }
    if (fread(&cols, sizeof(int), 1, f) != 1) { free(model); fclose(f); return NULL; }
    
    model->weights = matrix_create(rows, cols);
    if (!model->weights) {
        free(model);
        fclose(f);
        return NULL;
    }

    for (int i = 0; i < rows; i++) {
        if (fread(model->weights->data[i], sizeof(double), cols, f) != (size_t)cols) {
            matrix_free(model->weights);
            free(model);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    return model;
}

#include "metrics.h"
#include <stdlib.h>
#include <math.h>

double accuracy_score(Matrix* y_true, Matrix* y_pred) {
    if (!y_true || !y_pred || y_true->rows != y_pred->rows) return -1;

    int correct = 0;
    for (int i = 0; i < y_true->rows; i++) {
        if ((int)y_true->data[i][0] == (int)y_pred->data[i][0]) {
            correct++;
        }
    }
    return (double)correct / y_true->rows;
}

double precision_score(Matrix* y_true, Matrix* y_pred) {
    if (!y_true || !y_pred || y_true->rows != y_pred->rows) return -1;

    int tp = 0, fp = 0;
    for (int i = 0; i < y_true->rows; i++) {
        int y_t = (int)y_true->data[i][0];
        int y_p = (int)y_pred->data[i][0];
        if (y_p == 1) {
            if (y_t == 1) tp++;
            else fp++;
        }
    }

    if (tp + fp == 0) return 0.0;
    return (double)tp / (tp + fp);
}

double recall_score(Matrix* y_true, Matrix* y_pred) {
    if (!y_true || !y_pred || y_true->rows != y_pred->rows) return -1;

    int tp = 0, fn = 0;
    for (int i = 0; i < y_true->rows; i++) {
        int y_t = (int)y_true->data[i][0];
        int y_p = (int)y_pred->data[i][0];
        if (y_t == 1) {
            if (y_p == 1) tp++;
            else fn++;
        }
    }

    if (tp + fn == 0) return 0.0;
    return (double)tp / (tp + fn);
}

double f1_score(Matrix* y_true, Matrix* y_pred) {
    double precision = precision_score(y_true, y_pred);
    double recall = recall_score(y_true, y_pred);

    if (precision + recall == 0) return 0.0;
    return 2 * (precision * recall) / (precision + recall);
}

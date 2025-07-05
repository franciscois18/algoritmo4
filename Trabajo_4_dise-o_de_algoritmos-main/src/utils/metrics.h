#ifndef METRICS_H
#define METRICS_H

#include "../core/matrix.h"

/**
 * @brief Calcula la precisión (accuracy) entre etiquetas verdaderas y predichas.
 * @param y_true Etiquetas verdaderas (n x 1)
 * @param y_pred Etiquetas predichas (n x 1)
 * @return double Precisión (entre 0 y 1), o -1 si error
 */
double accuracy_score(Matrix* y_true, Matrix* y_pred);

/**
 * @brief Calcula la precisión (precision) binaria: TP / (TP + FP)
 * @param y_true Etiquetas verdaderas (n x 1)
 * @param y_pred Etiquetas predichas (n x 1)
 * @return double Precisión binaria, o -1 si error
 */
double precision_score(Matrix* y_true, Matrix* y_pred);

/**
 * @brief Calcula la sensibilidad (recall) binaria: TP / (TP + FN)
 * @param y_true Etiquetas verdaderas (n x 1)
 * @param y_pred Etiquetas predichas (n x 1)
 * @return double Sensibilidad binaria, o -1 si error
 */
double recall_score(Matrix* y_true, Matrix* y_pred);

/**
 * @brief Calcula la puntuación F1 binaria: 2 * (precision * recall) / (precision + recall)
 * @param y_true Etiquetas verdaderas (n x 1)
 * @param y_pred Etiquetas predichas (n x 1)
 * @return double F1 Score, o -1 si error
 */
double f1_score(Matrix* y_true, Matrix* y_pred);

#endif // METRICS_H

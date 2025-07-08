#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ejemplo Mínimo de Machine Learning en Python

Este script muestra cómo implementar los algoritmos de:
- K-Vecinos Más Cercanos (KNN)
- K-Means
- Regresión Lineal

Utilizando el conjunto de datos Iris.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configuración para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

def cargar_datos(ruta_archivo):
    """
    Carga el conjunto de datos desde un archivo CSV.
    
    Args:
        ruta_archivo: Ruta al archivo CSV
        
    Returns:
        X: Características
        y: Etiquetas
    """
    print(f"Cargando datos desde: {ruta_archivo}")
    
    # Cargar el conjunto de datos
    datos = pd.read_csv(ruta_archivo)
    
    # Mostrar información sobre el conjunto de datos
    print(f"Dimensiones del conjunto de datos: {datos.shape}")
    print("\nPrimeras 5 filas:")
    print(datos.head())
    
    # Separar características y etiquetas
    X = datos.iloc[:, :-1].values  # Todas las columnas excepto la última
    y = datos.iloc[:, -1].values   # Última columna
    
    return X, y

def aplicar_knn(X, y):
    """
    Aplica el algoritmo K-Vecinos Más Cercanos.
    
    Args:
        X: Características
        y: Etiquetas
    """
    print("\n=== K-Vecinos Más Cercanos (KNN) ===")
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    print(f"Conjunto de entrenamiento: {X_train.shape}")
    print(f"Conjunto de prueba: {X_test.shape}")
    
    # Crear y entrenar el modelo
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = knn.predict(X_test)
    
    # Evaluar el modelo
    precision = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo KNN (k={k}): {precision:.4f}")
    
    # Visualizar las primeras predicciones
    print("\nPrimeras 5 predicciones:")
    for i in range(min(5, len(y_test))):
        print(f"Real: {y_test[i]}, Predicción: {y_pred[i]}")

def aplicar_kmeans(X):
    """
    Aplica el algoritmo K-Means.
    
    Args:
        X: Características
    """
    print("\n=== K-Means ===")
    
    # Normalizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Crear y entrenar el modelo
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X_scaled)
    
    # Calcular inercia
    inercia = kmeans.inertia_
    print(f"Número de clusters: {n_clusters}")
    print(f"Inercia: {inercia:.4f}")
    
    # Visualizar los centroides y clusters (usando las dos primeras características)
    plt.figure()
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='viridis', alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=100, c='red', marker='X', label='Centroides')
    plt.title('Clusters K-Means y Centroides')
    plt.xlabel('Característica 1 (normalizada)')
    plt.ylabel('Característica 2 (normalizada)')
    plt.legend()
    plt.savefig('kmeans_clusters.png')
    print("Gráfico de clusters guardado como 'kmeans_clusters.png'")

def aplicar_regresion_lineal(X, y):
    """
    Aplica el algoritmo de Regresión Lineal.
    
    Args:
        X: Características
        y: Etiquetas
    """
    print("\n=== Regresión Lineal ===")
    
    # Para la regresión, usaremos la longitud del pétalo para predecir el ancho del pétalo
    X_regresion = X[:, 2].reshape(-1, 1)  # Longitud del pétalo
    y_regresion = X[:, 3]                 # Ancho del pétalo
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_regresion, y_regresion, test_size=0.2, random_state=42)
    
    # Crear y entrenar el modelo
    regresion = LinearRegression()
    regresion.fit(X_train, y_train)
    
    # Realizar predicciones
    y_pred = regresion.predict(X_test)
    
    # Evaluar el modelo
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Coeficiente: {regresion.coef_[0]:.4f}")
    print(f"Intercepto: {regresion.intercept_:.4f}")
    print(f"Error Cuadrático Medio: {mse:.4f}")
    print(f"Coeficiente R²: {r2:.4f}")
    
    # Visualizar la regresión
    plt.figure()
    plt.scatter(X_test, y_test, color='blue', label='Datos reales')
    plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regresión')
    plt.title('Regresión Lineal: Longitud vs Ancho del Pétalo')
    plt.xlabel('Longitud del Pétalo')
    plt.ylabel('Ancho del Pétalo')
    plt.legend()
    plt.savefig('regresion_lineal.png')
    print("Gráfico de regresión guardado como 'regresion_lineal.png'")

def main():
    """Función principal"""
    print("Ejemplo Mínimo de Machine Learning en Python")
    print("===========================================")
    
    # Cargar datos
    ruta_archivo = "data/iris.csv"
    X, y = cargar_datos(ruta_archivo)
    
    # Aplicar algoritmos
    aplicar_knn(X, y)
    aplicar_kmeans(X)
    aplicar_regresion_lineal(X, y)

if __name__ == "__main__":
    main()

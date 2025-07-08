# Código Base para Algoritmos de Machine Learning

Este directorio contiene el código base mínimo necesario para implementar tres algoritmos fundamentales de Machine Learning:

1. **K-Vecinos Más Cercanos (KNN)** - Un algoritmo de clasificación supervisada
2. **K-Means** - Un algoritmo de clustering no supervisado
3. **Regresión Lineal** - Un algoritmo de regresión supervisada

Además, se incluye una implementación completa de un lector de archivos CSV para facilitar la carga de conjuntos de datos.

## Estructura del Proyecto

```
basico/
├── Makefile
├── data/
│   └── iris.csv           # Conjunto de datos Iris de ejemplo
├── src/
│   ├── algorithms/
│   │   ├── knn.c           # Implementación de KNN (a completar)
│   │   ├── knn.h           # Interfaz de KNN
│   │   ├── kmeans.c        # Implementación de K-Means (a completar)
│   │   ├── kmeans.h        # Interfaz de K-Means
│   │   ├── linear_regression.c  # Implementación de Regresión Lineal (a completar)
│   │   └── linear_regression.h  # Interfaz de Regresión Lineal
│   ├── core/
│   │   ├── matrix.c        # Implementación de operaciones con matrices
│   │   └── matrix.h        # Interfaz de operaciones con matrices
│   ├── utils/
│   │   ├── csv_reader.c    # Implementación del lector de CSV
│   │   └── csv_reader.h    # Interfaz del lector de CSV
│   └── main.c              # Programa principal con ejemplos
```

## Instrucciones para los Alumnos

1. **Objetivo**: Implementar los tres algoritmos de Machine Learning siguiendo las interfaces proporcionadas.

2. **Archivos a Modificar**:
   - `src/algorithms/knn.c`
   - `src/algorithms/kmeans.c`
   - `src/algorithms/linear_regression.c`

3. **Archivos que NO deben Modificarse**:
   - Todos los archivos `.h` (contienen las interfaces)
   - `src/core/matrix.c` (implementación de matrices)
   - `src/main.c` (programa de prueba)

## Compilación y Ejecución

### Versión en C

1. Compilar el proyecto:
   ```bash
   make
   ```

2. Ejecutar el programa de demostración con datos aleatorios:
   ```bash
   ./ml_demo
   ```

3. Ejecutar el ejemplo con el conjunto de datos Iris:
   ```bash
   ./ml_iris
   ```
   o usando el Makefile:
   ```bash
   make run-iris
   ```

## Conjunto de Datos Iris

El proyecto incluye el famoso conjunto de datos Iris, que contiene mediciones de 150 flores de iris de tres especies diferentes:

- **Características**: 
  - Longitud del sépalo (cm)
  - Ancho del sépalo (cm)
  - Longitud del pétalo (cm)
  - Ancho del pétalo (cm)

- **Etiquetas**: 
  - Clase 0: Iris Setosa
  - Clase 1: Iris Versicolor
  - Clase 2: Iris Virginica

Este conjunto de datos es ideal para probar algoritmos de clasificación (KNN), clustering (K-Means) y regresión lineal.

## Ejemplos Incluidos

### Ejemplo Básico (`ml_demo`)

Este ejemplo muestra el uso de los tres algoritmos con datos generados aleatoriamente:

- **KNN**: Clasificación de puntos en clusters aleatorios
- **K-Means**: Agrupamiento de datos en clusters
- **Regresión Lineal**: Predicción de valores basados en una relación lineal

### Ejemplo con Iris (`ml_iris`)

Este ejemplo utiliza el conjunto de datos Iris para mostrar aplicaciones reales de los algoritmos:

- **KNN**: Clasificación de especies de iris basada en las medidas de sépalos y pétalos
- **K-Means**: Agrupamiento no supervisado para descubrir patrones naturales en los datos
- **Regresión Lineal**: Predicción del ancho del pétalo a partir de su longitud

El ejemplo incluye:
- Carga de datos desde CSV
- División en conjuntos de entrenamiento y prueba
- Cálculo de métricas de rendimiento (precisión, inercia, error cuadrático medio, R²)
- Visualización de resultados

### Ejemplo en Python

Se incluye un ejemplo en Python que implementa los mismos algoritmos usando scikit-learn para comparación:

1. Crear entorno virtual e instalar dependencias:
   ```bash
   make install-py-reqs
   ```

2. Ejecutar el ejemplo de Python:
   ```bash
   make run-py
   ```

Esto creará un entorno virtual, instalará las dependencias necesarias y ejecutará el ejemplo que muestra la implementación de los tres algoritmos con el conjunto de datos Iris.

## Descripción de los Componentes

### Utilidades Proporcionadas

#### Operaciones con Matrices

Se proporciona una implementación completa de operaciones con matrices, incluyendo creación, liberación, multiplicación, transposición y cálculo de distancias.

#### Lector de CSV

Se incluye una implementación completa para leer conjuntos de datos desde archivos CSV:

**Funciones disponibles**:
- `csv_read`: Lee un archivo CSV y devuelve los datos como matrices
- `csv_free`: Libera la memoria utilizada por la estructura CSVData
- `train_test_split`: Divide los datos en conjuntos de entrenamiento y prueba

### Algoritmos a Implementar

#### K-Vecinos Más Cercanos (KNN)

Algoritmo de clasificación que asigna a un nuevo punto la clase mayoritaria entre sus k vecinos más cercanos en el conjunto de entrenamiento.

**Funciones a implementar**:
- `knn_create`: Crear un nuevo clasificador
- `knn_fit`: Entrenar el clasificador con datos
- `knn_predict`: Predecir clases para nuevos datos
- `knn_free`: Liberar memoria

#### K-Means

Algoritmo de clustering que agrupa datos en k clusters, minimizando la distancia de cada punto al centroide de su cluster.

**Funciones a implementar**:
- `kmeans_create`: Crear un nuevo modelo
- `kmeans_fit`: Entrenar el modelo con datos
- `kmeans_predict`: Asignar clusters a nuevos datos
- `kmeans_inertia`: Calcular la inercia del modelo
- `kmeans_free`: Liberar memoria

#### Regresión Lineal

Algoritmo que modela la relación entre variables mediante una función lineal, minimizando el error cuadrático.

**Funciones a implementar**:
- `linear_regression_create`: Crear un nuevo modelo
- `linear_regression_fit`: Entrenar el modelo con datos
- `linear_regression_predict`: Predecir valores para nuevos datos
- `linear_regression_mse`: Calcular el error cuadrático medio
- `linear_regression_r2_score`: Calcular el coeficiente de determinación
- `linear_regression_free`: Liberar memoria

## Evaluación

La implementación se evaluará según:
1. Correctitud de los algoritmos
2. Eficiencia computacional
3. Manejo adecuado de memoria
4. Calidad del código

## Referencias

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.

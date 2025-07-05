CC = gcc
CFLAGS = -Wall -Wextra -std=c99 -pedantic -g -O2
LDFLAGS = -lm

SRC_DIR = src
BUILD_DIR = build
VENV_DIR = venv

# Archivos fuente
CORE_SRC = $(SRC_DIR)/core/matrix.c
UTILS_SRC = $(SRC_DIR)/utils/csv_reader.c
ALGO_SRC = $(SRC_DIR)/algorithms/knn.c $(SRC_DIR)/algorithms/kmeans.c $(SRC_DIR)/algorithms/linear_regression.c
MAIN_SRC = $(SRC_DIR)/main.c
IRIS_SRC = $(SRC_DIR)/main-iris.c

# Objetos
CORE_OBJ = $(CORE_SRC:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
UTILS_OBJ = $(UTILS_SRC:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
ALGO_OBJ = $(ALGO_SRC:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
MAIN_OBJ = $(MAIN_SRC:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
IRIS_OBJ = $(IRIS_SRC:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Ejecutables
TARGET = ml_demo
IRIS_TARGET = ml_iris
PY_EXAMPLE = ejemplo_python.py

# Regla principal
all: $(BUILD_DIR) $(TARGET) $(IRIS_TARGET)

# Crear directorio de compilación
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)/core
	mkdir -p $(BUILD_DIR)/utils
	mkdir -p $(BUILD_DIR)/algorithms

# Compilar los ejecutables
$(TARGET): $(CORE_OBJ) $(UTILS_OBJ) $(ALGO_OBJ) $(MAIN_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(IRIS_TARGET): $(CORE_OBJ) $(UTILS_OBJ) $(ALGO_OBJ) $(IRIS_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compilar archivos objeto
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Crear entorno virtual de Python
$(VENV_DIR):
	python3 -m venv $(VENV_DIR)
	touch $(VENV_DIR)

# Instalar requisitos de Python
.PHONY: install-py-reqs
install-py-reqs: $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r requirements.txt

# Ejecutar ejemplo de Python
.PHONY: run-py
run-py: install-py-reqs
	$(VENV_DIR)/bin/python $(PY_EXAMPLE)

# Ejecutar ejemplo de Iris en C
.PHONY: run-iris
run-iris: $(IRIS_TARGET)
	./$(IRIS_TARGET)

# Limpiar
clean:
	rm -rf $(BUILD_DIR) $(TARGET) $(IRIS_TARGET)

# Limpiar todo (incluyendo entorno virtual y archivos generados por Python)
clean-all: clean
	rm -rf $(VENV_DIR) *.png

.PHONY: all clean clean-all

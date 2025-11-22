# Entrega 3 - Clasificador de Actividades Humanas en Tiempo Real

## 📋 Descripción

Sistema de clasificación de actividades humanas en tiempo real utilizando MediaPipe Pose y un modelo de Random Forest optimizado. El sistema detecta y clasifica las siguientes actividades:

- **caminar hacia al frente**
- **caminar hacia atras**
- **giro 180**
- **ponerse de pie**
- **sentarse**

## 🎯 Rendimiento del Modelo

- **F1-Score**: 0.7067 ± 0.0507
- **Accuracy**: 69.37% ± 3.64%

### F1-Score por Actividad:
- caminar hacia al frente: 0.50
- caminar hacia atras: 0.86
- giro 180: 0.73
- ponerse de pie: 0.71
- sentarse: 0.67

## 📦 Archivos Incluidos

```
Entrega 3/
├── clasificador_final.py          # Script principal del clasificador
├── balanced_rf_model.pkl          # Modelo Random Forest entrenado
├── balanced_label_encoder.pkl     # Codificador de etiquetas
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

## 🚀 Instalación

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

O instalar manualmente:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn joblib
```

## 💻 Uso

### Ejecutar el clasificador

```bash
python clasificador_final.py
```

### Requisitos

- **Cámara web**: El sistema requiere una cámara web conectada
- **Python 3.7+**: Compatible con Python 3.7 o superior
- **Sistema operativo**: Windows, macOS o Linux

### Controles

- **Presiona 'q' o ESC**: Para salir del clasificador
- **Calibración**: El sistema necesita 15 frames para calibrarse antes de comenzar a clasificar

## 🔧 Configuración

El archivo `clasificador_final.py` contiene los siguientes parámetros configurables:

```python
MIN_HISTORY = 15          # Frames mínimos para calibración
WINDOW_SIZE = 10          # Ventana de suavizado temporal
CONFIDENCE_THRESHOLD = 0.35  # Umbral de confianza
FEATURES_WINDOW = 30      # Ventana para calcular features temporales
```

## 📊 Características del Sistema

### Extracción de Features

El sistema extrae 17 características biomecánicas:

1. **Velocidades**: mean_speed, std_speed, p75_speed, p90_speed
2. **Postura**: trunk_angle_deg, trunk_angle_var
3. **Altura**: head_y_mean, head_y_min, head_y_range, hip_y_mean
4. **Desplazamiento**: hip_x_disp, hip_x_path
5. **Temporales**: seg_len, avg_vertical_speed_hip, avg_horizontal_speed_hip
6. **Distancia**: normalized_head_hip_distance_mean

### Normalización

Los landmarks se normalizan por la distancia entre hombros, haciendo el modelo invariante al tamaño corporal y distancia a la cámara.

## 🏗️ Arquitectura

- **MediaPipe Pose**: Detección de landmarks corporales (33 puntos)
- **Random Forest**: Modelo de clasificación con:
  - `n_estimators`: 100
  - `max_depth`: 10
  - `min_samples_split`: 4
  - `class_weight`: balanced

## 📝 Notas

- El sistema muestra un mensaje de "CALIBRANDO..." durante los primeros 15 frames
- La confianza se muestra con colores:
  - **Verde**: > 70% de confianza
  - **Naranja**: 50-70% de confianza
  - **Amarillo**: 35-50% de confianza
- Al finalizar, se muestran estadísticas de la sesión

## 👥 Integrantes

- Juan Esteban Eraso
- Damy Villegas
- Cristian Molina
- Carlos Sanchez

## 📚 Proyecto Final APO III

Sistema de Análisis de Movimiento Humano - Universidad ICESI


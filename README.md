# Sistema de Análisis de Movimiento Humano

**Estado del Proyecto**: Activo

## 👥 Miembros Colaboradores

*   Damy Villegas
*   Cristian Molina
*   Carlos Sanchez
*   Juan Esteban Eraso

## 📖 Introducción/Objetivo del Proyecto

Este proyecto busca crear un sistema automático para analizar videos de movimiento humano y detectar signos de movilidad reducida. Utilizaremos MediaPipe Pose para rastrear las articulaciones y modelos de aprendizaje automático para identificar patrones anormales en la amplitud, simetría y velocidad del movimiento. El objetivo es ofrecer una herramienta complementaria para el monitoreo y apoyo preventivo en la detección temprana de limitaciones motoras.

## 🔬 Métodos Utilizados

*   MediaPipe Pose
*   Aprendizaje Automático Supervisado
*   EDA (Análisis Exploratorio de Datos)

## 🛠️ Tecnologías

*   Python
*   MediaPipe
*   scikit-learn
*   OpenCV
*   Streamlit
*   Pandas
*   Matplotlib

## 📝 Descripción del Proyecto

El proyecto aborda la clasificación de actividades y la detección de movilidad reducida a partir de videos. Se sigue la metodología CRISP-DM, cubriendo desde la recolección y preparación de datos (extracción de coordenadas con MediaPipe, generación de características biomecánicas) hasta el modelado con algoritmos supervisados (Random Forest, SVM, XGBoost) y la evaluación. Se consideran principios éticos para garantizar un uso seguro y responsable de la tecnología. Los datos consisten en videos de actividades básicas (sentarse, levantarse, caminar, girar) con etiquetas de segmento.

---

## 🚀 Cómo Probar el Clasificador de Actividades (Entrega 3)

### 📋 Requisitos Previos

- **Python 3.8, 3.9, 3.10 o 3.11** (⚠️ **Importante**: Python 3.12+ no es compatible con MediaPipe)
- **Cámara web** conectada y funcionando
- **Sistema operativo**: Windows, macOS o Linux

### 🔧 Instalación y Configuración

#### 1. Verificar Versión de Python

**⚠️ IMPORTANTE**: MediaPipe requiere Python 3.8-3.11. Python 3.12+ no es compatible.

Verifica tu versión de Python:
```bash
python3 --version
```

Si tienes Python 3.12 o superior, necesitas instalar una versión compatible:

**En macOS (usando Homebrew):**
```bash
# Instalar Python 3.11
brew install python@3.11

# Usar Python 3.11 para el entorno virtual
python3.11 -m venv venv
```

**En Linux:**
```bash
# Instalar Python 3.11
sudo apt-get install python3.11 python3.11-venv

# Usar Python 3.11 para el entorno virtual
python3.11 -m venv venv
```

**En Windows:**
- Descarga Python 3.11 desde [python.org](https://www.python.org/downloads/)
- Durante la instalación, marca "Add Python to PATH"
- Usa `py -3.11` para especificar la versión

#### 2. Crear un Entorno Virtual

Es recomendable crear un entorno virtual para aislar las dependencias del proyecto:

**En Windows:**
```bash
# Crear el entorno virtual (usar Python 3.11 si es necesario)
python -m venv venv
# O si instalaste Python 3.11 específicamente:
py -3.11 -m venv venv

# Activar el entorno virtual
venv\Scripts\activate
```

**En macOS/Linux:**
```bash
# Crear el entorno virtual (usar Python 3.11 si es necesario)
python3 -m venv venv
# O si instalaste Python 3.11 específicamente:
python3.11 -m venv venv

# Activar el entorno virtual
source venv/bin/activate
```

#### 3. Navegar a la Carpeta de Entrega 3

```bash
cd Entrega\ 3/
```

#### 4. Instalar Dependencias

Una vez activado el entorno virtual, instalar las dependencias:

```bash
pip install -r requirements.txt
```

Esto instalará:
- `opencv-python` - Procesamiento de video y cámara
- `mediapipe` - Detección de landmarks corporales
- `numpy` - Operaciones numéricas
- `pandas` - Manipulación de datos
- `scikit-learn` - Modelos de machine learning
- `joblib` - Serialización de modelos

#### 5. Verificar Archivos Necesarios

Asegúrarse de tener estos archivos en la carpeta `Entrega 3/`:
- ✅ `clasificador_final.py`
- ✅ `balanced_rf_model.pkl`
- ✅ `balanced_label_encoder.pkl`

### ▶️ Ejecutar el Clasificador

```bash
python clasificador_final.py
```

### 🎮 Controles

- **Presiona 'q' o ESC**: Para salir del clasificador
- **Calibración**: El sistema necesita 15 frames para calibrarse antes de comenzar a clasificar (verás "CALIBRANDO..." en pantalla)

### 📊 Actividades Detectadas

El sistema clasifica las siguientes actividades:
- **caminar hacia al frente**
- **caminar hacia atras**
- **giro 180**
- **ponerse de pie**
- **sentarse**

### 🎨 Indicadores Visuales

- **Verde**: Confianza > 70%
- **Naranja**: Confianza 50-70%
- **Amarillo**: Confianza 35-50%
- **Rojo**: No se detecta persona

### 📈 Rendimiento del Modelo

- **F1-Score**: 0.7067 ± 0.0507
- **Accuracy**: 69.37% ± 3.64%

### ⚠️ Solución de Problemas

**Error: "No se pudo abrir la cámara"**
- Verifica que la cámara web esté conectada y no esté siendo usada por otra aplicación
- En Linux, puede ser necesario instalar `v4l-utils`

**Error: "Archivo no encontrado"**
- Asegúrate de estar en la carpeta `Entrega 3/`
- Verifica que los archivos `.pkl` estén presentes

**Error al importar módulos**
- Asegúrate de haber activado el entorno virtual
- Reinstala las dependencias: `pip install -r requirements.txt`

**Error: "Could not find a version that satisfies the requirement mediapipe"**
- ⚠️ **Este error indica que estás usando Python 3.12 o superior**
- MediaPipe solo soporta Python 3.8-3.11
- Solución: Instala Python 3.11 y crea un nuevo entorno virtual con esa versión:
  ```bash
  # macOS con Homebrew
  brew install python@3.11
  # Navegar a la carpeta del proyecto
  cd ruta/a/tu/proyecto
  rm -rf venv  # Eliminar entorno virtual anterior si existe
  python3.11 -m venv venv
  source venv/bin/activate
  cd "Entrega 3"
  pip install -r requirements.txt
  
  # Linux
  sudo apt-get install python3.11 python3.11-venv
  cd ruta/a/tu/proyecto
  rm -rf venv
  python3.11 -m venv venv
  source venv/bin/activate
  cd "Entrega 3"
  pip install -r requirements.txt
  
  # Windows
  # Descarga Python 3.11 desde python.org
  # Luego en la terminal:
  cd ruta\a\tu\proyecto
  rmdir /s venv  # Eliminar entorno virtual anterior si existe
  py -3.11 -m venv venv
  venv\Scripts\activate
  cd "Entrega 3"
  pip install -r requirements.txt
  ```

### 🔄 Desactivar el Entorno Virtual

Cuando termines de usar el programa:

```bash
deactivate
```

---

## 📁 Estructura del Proyecto

```
ProyectoFinalIA/
├── Entrega 1/          # Análisis exploratorio y etiquetado inicial
│   ├── videos/         # Videos originales
│   ├── labels/          # Etiquetas temporales
│   └── eda_outputs/     # Resultados del EDA
├── Entrega 2/          # Entrenamiento del modelo y artefactos
│   ├── artifacts/       # Modelos entrenados (.pkl)
│   ├── labels/          # Etiquetas refinadas
│   └── videos/          # Videos procesados
├── Entrega 3/          # Clasificador en tiempo real
│   ├── clasificador_final.py
│   ├── balanced_rf_model.pkl
│   ├── balanced_label_encoder.pkl
│   ├── requirements.txt
│   └── README.md
└── README.md           # Este archivo
```
# Sistema de Análisis de Movimiento Humano

-- Estado del Proyecto: Activo

## Miembros Colaboradores

*   Damy Villegas
*   Cristian Molina
*   Carlos Sanchez
*   Juan Esteban Eraso

## Introducción/Objetivo del Proyecto

Este proyecto busca crear un sistema automático para analizar videos de movimiento humano y detectar signos de movilidad reducida. Utilizaremos MediaPipe Pose para rastrear las articulaciones y modelos de aprendizaje automático para identificar patrones anormales en la amplitud, simetría y velocidad del movimiento. El objetivo es ofrecer una herramienta complementaria para el monitoreo y apoyo preventivo en la detección temprana de limitaciones motoras.

## Métodos Utilizados

*   MediaPipe Pose
*   Aprendizaje Automático Supervisado
*   EDA

## Tecnologías

*   Python
*   MediaPipe
*   scikit-learn
*   OpenCV
*   Streamlit
*   Pandas
*   Matplotlib

## Descripción del Proyecto

El proyecto aborda la clasificación de actividades y la detección de movilidad reducida a partir de videos. Se sigue la metodología CRISP-DM, cubriendo desde la recolección y preparación de datos (extracción de coordenadas con MediaPipe, generación de características biomecánicas) hasta el modelado con algoritmos supervisados (Random Forest, SVM, XGBoost) y la evaluación. Se consideran principios éticos para garantizar un uso seguro y responsable de la tecnología. Los datos consisten en videos de actividades básicas (sentarse, levantarse, caminar, girar) con etiquetas de segmento.

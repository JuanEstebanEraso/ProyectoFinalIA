#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLASIFICADOR DE ACTIVIDADES HUMANAS EN TIEMPO REAL
Proyecto Final APO III - Versión Mejorada

Modelo: Random Forest Optimizado con class_weight balanceado
F1-Score: 0.7067 ± 0.0507
Accuracy: 0.6937 ± 0.0364

Actividades:
- caminar hacia al frente (F1: 0.50)
- caminar hacia atras (F1: 0.86)
- giro 180 (F1: 0.73)
- ponerse de pie (F1: 0.71)
- sentarse (F1: 0.67)
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import json
import os
import time
from collections import deque

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Parámetros de procesamiento
MIN_HISTORY = 15          # Frames mínimos para calibración
WINDOW_SIZE = 10          # Ventana de suavizado temporal
CONFIDENCE_THRESHOLD = 0.35  # Umbral de confianza (reducido para mejor detección)
FEATURES_WINDOW = 30      # Ventana para calcular features temporales

# ============================================================================
# FUNCIONES DE NORMALIZACIÓN Y EXTRACCIÓN
# ============================================================================

def normalize_landmarks(landmarks_array):
    """
    Normaliza landmarks por la distancia entre hombros.
    Hace el modelo invariante al tamaño corporal y distancia a la cámara.
    
    Args:
        landmarks_array: Array de 99 valores (33 landmarks × 3 coords)
    
    Returns:
        Array normalizado o None si falla
    """
    try:
        # Hombros: landmark 11 (izq) y 12 (der)
        left_shoulder = landmarks_array[11*3:11*3+3]
        right_shoulder = landmarks_array[12*3:12*3+3]
        
        # Calcular distancia entre hombros
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        
        if shoulder_dist < 0.01:  # Evitar división por cero
            return None
        
        # Normalizar todo el array
        normalized = landmarks_array / shoulder_dist
        return normalized
        
    except Exception as e:
        return None


def extract_features_from_landmarks(landmarks_norm):
    """
    Extrae 13 características biomecánicas estáticas de un frame.
    
    Features extraídas:
    1. trunk_angle_deg: Ángulo del tronco (postura)
    2. head_y_mean: Altura de la cabeza
    3. head_y_min: Altura mínima de la cabeza
    4. head_y_range: Rango de movimiento vertical
    5. hip_y_mean: Altura de la cadera
    6. hip_x_disp: Desplazamiento horizontal neto
    7. hip_x_path: Camino total recorrido en X
    8. seg_len: Longitud del segmento (temporal)
    9. normalized_head_hip_distance_mean: Distancia cabeza-cadera
    """
    try:
        # Índices de landmarks clave (después de normalización)
        HEAD_IDX = 0         # Nariz
        L_SHOULDER_IDX = 11  # Hombro izquierdo
        R_SHOULDER_IDX = 12  # Hombro derecho
        L_HIP_IDX = 23       # Cadera izquierda
        R_HIP_IDX = 24       # Cadera derecha
        
        # Extraer coordenadas
        head = landmarks_norm[HEAD_IDX*3:HEAD_IDX*3+3]
        l_shoulder = landmarks_norm[L_SHOULDER_IDX*3:L_SHOULDER_IDX*3+3]
        r_shoulder = landmarks_norm[R_SHOULDER_IDX*3:R_SHOULDER_IDX*3+3]
        l_hip = landmarks_norm[L_HIP_IDX*3:L_HIP_IDX*3+3]
        r_hip = landmarks_norm[R_HIP_IDX*3:R_HIP_IDX*3+3]
        
        # Puntos medios
        mid_shoulder = (l_shoulder + r_shoulder) / 2
        mid_hip = (l_hip + r_hip) / 2
        
        # 1. Ángulo del tronco (hombros a caderas)
        trunk_vec = mid_hip - mid_shoulder
        vertical = np.array([0, 1, 0])
        
        dot_prod = np.dot(trunk_vec, vertical)
        trunk_norm = np.linalg.norm(trunk_vec)
        
        if trunk_norm > 0:
            trunk_angle_rad = np.arccos(np.clip(dot_prod / trunk_norm, -1.0, 1.0))
            trunk_angle_deg = float(np.degrees(trunk_angle_rad))
        else:
            trunk_angle_deg = 0.0
        
        # 2-5. Métricas de altura
        head_y = head[1]
        hip_y = mid_hip[1]
        
        head_y_mean = float(head_y)
        head_y_min = float(head_y)  # En un frame, es el mismo
        head_y_range = 0.0  # Se calcula con histórico
        hip_y_mean = float(hip_y)
        
        # 6-7. Desplazamiento horizontal
        hip_x = mid_hip[0]
        hip_x_disp = 0.0  # Se calcula con histórico
        hip_x_path = 0.0  # Se calcula con histórico
        
        # 8. Longitud del segmento
        seg_len = 1.0  # Se actualiza con histórico
        
        # 9. Distancia cabeza-cadera normalizada
        head_hip_dist = float(np.linalg.norm(head - mid_hip))
        
        # Features temporales (se actualizan luego)
        features = {
            "trunk_angle_deg": trunk_angle_deg,
            "head_y_mean": head_y_mean,
            "head_y_min": head_y_min,
            "head_y_range": head_y_range,
            "hip_y_mean": hip_y_mean,
            "hip_x_disp": hip_x_disp,
            "hip_x_path": hip_x_path,
            "seg_len": seg_len,
            "normalized_head_hip_distance_mean": head_hip_dist,
            # Features adicionales para actualización temporal
            "_head_y": head_y,
            "_hip_x": hip_x,
            "_hip_y": hip_y,
        }
        
        return features
        
    except Exception as e:
        return None


def update_temporal_features(history):
    """
    Actualiza features temporales basadas en el histórico de frames.
    
    Features calculadas:
    1-4. mean_speed, std_speed, p75_speed, p90_speed: Velocidades
    5. trunk_angle_var: Varianza del ángulo del tronco
    6. hip_x_disp: Desplazamiento neto en X
    7. hip_x_path: Camino total en X
    8. seg_len: Longitud del segmento
    9. avg_vertical_speed_hip: Velocidad vertical cadera
    10. avg_horizontal_speed_hip: Velocidad horizontal cadera
    11. head_y_range: Rango de movimiento vertical
    """
    if len(history) < 2:
        return {
            "mean_speed": 0.0, "std_speed": 0.0,
            "p75_speed": 0.0, "p90_speed": 0.0,
            "trunk_angle_var": 0.0,
            "hip_x_disp": 0.0, "hip_x_path": 0.0,
            "seg_len": 1.0,
            "avg_vertical_speed_hip": 0.0,
            "avg_horizontal_speed_hip": 0.0,
            "head_y_range": 0.0
        }
    
    # Extraer series temporales
    head_y_series = [h["_head_y"] for h in history]
    hip_x_series = [h["_hip_x"] for h in history]
    hip_y_series = [h["_hip_y"] for h in history]
    trunk_angles = [h["trunk_angle_deg"] for h in history]
    
    # Velocidades verticales (head_y)
    velocities = np.abs(np.diff(head_y_series))
    
    # Velocidades de la cadera
    hip_y_velocities = np.abs(np.diff(hip_y_series))
    hip_x_velocities = np.abs(np.diff(hip_x_series))
    
    # Rango de movimiento vertical
    head_y_range = float(np.max(head_y_series) - np.min(head_y_series))
    
    return {
        "mean_speed": float(np.mean(velocities)) if len(velocities) > 0 else 0.0,
        "std_speed": float(np.std(velocities)) if len(velocities) > 0 else 0.0,
        "p75_speed": float(np.percentile(velocities, 75)) if len(velocities) > 0 else 0.0,
        "p90_speed": float(np.percentile(velocities, 90)) if len(velocities) > 0 else 0.0,
        "trunk_angle_var": float(np.var(trunk_angles)) if len(trunk_angles) > 1 else 0.0,
        "hip_x_disp": float(hip_x_series[-1] - hip_x_series[0]) if len(hip_x_series) > 1 else 0.0,
        "hip_x_path": float(np.sum(np.abs(np.diff(hip_x_series)))) if len(hip_x_series) > 1 else 0.0,
        "seg_len": float(len(history)),
        "avg_vertical_speed_hip": float(np.mean(hip_y_velocities)) if len(hip_y_velocities) > 0 else 0.0,
        "avg_horizontal_speed_hip": float(np.mean(hip_x_velocities)) if len(hip_x_velocities) > 0 else 0.0,
        "head_y_range": head_y_range
    }

# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class FinalActivityClassifier:
    def __init__(self, model_path, encoder_path, meta_path=None):
        print("\n" + "="*70)
        print("  🎥 CLASIFICADOR DE ACTIVIDADES - VERSIÓN FINAL MEJORADA")
        print("  Proyecto Final APO III")
        print("  F1-Score: 0.7067 ± 0.0507 | Accuracy: 69.37%")
        print("="*70)
        print("\nINICIALIZANDO...\n")
        
        # Cargar modelo y encoder
        try:
            self.model = joblib.load(model_path)
            self.encoder = joblib.load(encoder_path)
            print(f"✓ Modelo: {type(self.model).__name__}")
            print(f"✓ Hiperparámetros:")
            print(f"  - n_estimators: {self.model.n_estimators}")
            print(f"  - max_depth: {self.model.max_depth}")
            print(f"  - class_weight: {self.model.class_weight}")
            print(f"✓ Clases: {list(self.encoder.classes_)}")
            
            if meta_path and os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    self.meta = json.load(f)
                print(f"✓ Metadatos cargados")
        except FileNotFoundError as e:
            print(f"✗ ERROR: Archivo no encontrado")
            print(f"  Asegúrate de tener estos archivos en la misma carpeta:")
            print(f"  - {model_path}")
            print(f"  - {encoder_path}")
            raise
        
        # Inicializar MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        print("✓ MediaPipe Pose inicializado")
        
        # Ventanas y historiales
        self.predictions_window = deque(maxlen=WINDOW_SIZE)
        self.confidence_window = deque(maxlen=WINDOW_SIZE)
        self.features_history = deque(maxlen=FEATURES_WINDOW)
        
        print("\n✓ LISTO PARA INICIAR\n")
    
    def process_frame(self, frame):
        """Procesa un frame y retorna la predicción"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None, None, 0.0, False
        
        # Extraer landmarks como array
        landmarks_list = []
        for lm in results.pose_landmarks.landmark:
            landmarks_list.extend([lm.x, lm.y, lm.z])
        landmarks_array = np.array(landmarks_list, dtype=np.float32)
        
        # Normalizar
        landmarks_norm = normalize_landmarks(landmarks_array)
        if landmarks_norm is None:
            return None, None, 0.0, False
        
        # Extraer features
        features = extract_features_from_landmarks(landmarks_norm)
        if features is None:
            return None, None, 0.0, False
        
        # Agregar al histórico
        self.features_history.append(features)
        
        # Actualizar features temporales
        if len(self.features_history) >= 2:
            temporal = update_temporal_features(self.features_history)
            features.update(temporal)
        else:
            # Inicializar features temporales en cero para el primer frame
            features.update({
                "mean_speed": 0.0, "std_speed": 0.0,
                "p75_speed": 0.0, "p90_speed": 0.0,
                "trunk_angle_var": 0.0,
                "avg_vertical_speed_hip": 0.0,
                "avg_horizontal_speed_hip": 0.0,
                "head_y_range": 0.0,
                "seg_len": 1.0
            })
        
        # Preparar para predicción (16 features en el orden correcto)
        feature_names = [
            'mean_speed', 'std_speed', 'p75_speed', 'p90_speed', 
            'trunk_angle_deg', 'trunk_angle_var', 'head_y_mean', 
            'head_y_min', 'head_y_range', 'hip_y_mean', 'hip_x_disp', 
            'hip_x_path', 'seg_len', 'avg_vertical_speed_hip', 
            'avg_horizontal_speed_hip', 'normalized_head_hip_distance_mean'
        ]
        X = pd.DataFrame([features])[feature_names].fillna(0.0)
        
        # Predecir
        pred_encoded = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]
        confidence = float(pred_proba[pred_encoded])
        activity = self.encoder.inverse_transform([pred_encoded])[0]
        
        # Suavizado temporal
        self.predictions_window.append(activity)
        self.confidence_window.append(confidence)
        
        if len(self.predictions_window) > 0:
            smoothed_activity = max(set(self.predictions_window), 
                                   key=list(self.predictions_window).count)
            smoothed_confidence = np.mean(list(self.confidence_window))
        else:
            smoothed_activity = activity
            smoothed_confidence = confidence
        
        return results.pose_landmarks, smoothed_activity, smoothed_confidence, True
    
    def run(self):
        """Ejecuta el clasificador con webcam"""
        print("="*70)
        print("  ABRIENDO WEBCAM...")
        print("="*70 + "\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ ERROR: No se pudo abrir la cámara")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✓ Webcam abierta")
        print("\n" + "="*70)
        print("  CLASIFICADOR ACTIVO")
        print("  Presiona 'q' o ESC para salir")
        print("="*70 + "\n")
        
        fps_history = deque(maxlen=30)
        prev_time = time.time()
        frame_count = 0
        avg_fps = 0.0  # Inicializar para evitar error si hay excepción temprana
        
        # Contadores de detecciones
        activity_counts = {cls: 0 for cls in self.encoder.classes_}
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                current_time = time.time()
                frame_count += 1
                
                # Procesar frame
                landmarks, activity, confidence, detected = self.process_frame(frame)
                
                # Calcular FPS
                fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0
                fps_history.append(fps)
                avg_fps = np.mean(list(fps_history))
                prev_time = current_time
                
                # Dibujar información
                h, w = frame.shape[:2]
                
                # Panel semi-transparente superior
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (w - 10, 140), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                if detected and landmarks:
                    # Dibujar esqueleto
                    self.mp_drawing.draw_landmarks(
                        frame, landmarks, self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    
                    # Estado de calibración
                    if len(self.features_history) < MIN_HISTORY:
                        status = f"CALIBRANDO... {len(self.features_history)}/{MIN_HISTORY}"
                        cv2.putText(frame, status, (20, 45),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                    else:
                        # Actividad detectada
                        if confidence > CONFIDENCE_THRESHOLD:
                            activity_counts[activity] += 1
                            
                            # Color según confianza
                            if confidence > 0.7:
                                color = (0, 255, 0)  # Verde
                            elif confidence > 0.5:
                                color = (0, 200, 255)  # Naranja
                            else:
                                color = (0, 165, 255)  # Amarillo
                            
                            cv2.putText(frame, f"Actividad: {activity}", (20, 45),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            cv2.putText(frame, f"Confianza: {confidence:.2%}", (20, 80),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        else:
                            cv2.putText(frame, "Detectando...", (20, 45),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)
                
                else:
                    cv2.putText(frame, "No se detecta persona", (20, 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # FPS y contador
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (20, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Frames: {frame_count}", (w - 150, 115),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Mostrar frame
                cv2.imshow('Clasificador de Actividades - Modelo Final', frame)
                
                # Control de salida
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q o ESC
                    break
                    
        except KeyboardInterrupt:
            print("\n\n✓ Interrumpido por el usuario")
        
        finally:
            # Estadísticas finales
            print("\n" + "="*70)
            print("  ESTADÍSTICAS DE LA SESIÓN")
            print("="*70)
            print(f"Frames procesados: {frame_count}")
            print(f"FPS promedio: {avg_fps:.1f}")
            print(f"\nDetecciones por actividad:")
            for activity, count in sorted(activity_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / frame_count) * 100
                    print(f"  {activity}: {count} frames ({percentage:.1f}%)")
            print("="*70 + "\n")
            
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Cámara cerrada")

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    # Rutas a los archivos del modelo
    MODEL_PATH = "balanced_rf_model.pkl"
    ENCODER_PATH = "balanced_label_encoder.pkl"
    META_PATH = "train_meta.json"
    
    # Verificar que existen los archivos
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: No se encuentra {MODEL_PATH}")
        print("Asegúrate de tener los archivos del modelo en esta carpeta:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {ENCODER_PATH}")
        print(f"  - {META_PATH} (opcional)")
        exit(1)
    
    if not os.path.exists(ENCODER_PATH):
        print(f"ERROR: No se encuentra {ENCODER_PATH}")
        exit(1)
    
    # Crear y ejecutar clasificador
    try:
        classifier = FinalActivityClassifier(MODEL_PATH, ENCODER_PATH, META_PATH)
        classifier.run()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()

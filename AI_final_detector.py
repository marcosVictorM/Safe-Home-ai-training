import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configurações ---
# Carregue o modelo V2 que você acabou de treinar
MODEL_PATH = "modelo_quedas_seq_60.keras"
FALL_TRIGGER_MESSAGE = "FALL_DETECTED_TRIGGER"

print(f"Carregando modelo '{MODEL_PATH}'...")
try:
    model = load_model(MODEL_PATH)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"ERRO: Não foi possível carregar o modelo. {e}")
    exit()

# Configurações do modelo 
SEQUENCE_LENGTH = 60 
NUM_FEATURES = 135 # 132 (pose) + 3 (extras)

LABELS = ["NORMAL", "QUEDA"]
FALL_CONFIDENCE_THRESHOLD = 0.95 
ALERT_DURATION_SECONDS = 3 
CONFIRMATION_FRAMES_THRESHOLD = 15 

# --- Inicialização ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
pose_sequence = deque(maxlen=SEQUENCE_LENGTH)

# Variáveis de estado
is_alert_active = False
alert_start_time = 0
confirmation_frames_counter = 0

# --- Variáveis para Engenharia de Features em tempo real ---
last_y_hip_mid = 0.5
last_frame_time = time.time()

# --- Função Helper (do extrator_v2) ---
def get_pose_features_realtime(frame_height, frame_width, landmarks):
    """Calcula a Bounding Box e a altura do quadril."""
    x_min, y_min = frame_width, frame_height
    x_max, y_max = 0, 0
    
    for lm in landmarks.landmark:
        if lm.visibility < 0.3: continue
        x, y = int(lm.x * frame_width), int(lm.y * frame_height)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)
            
    body_height, body_width = y_max - y_min, x_max - x_min
    aspect_ratio = body_width / body_height if body_height > 0 else 0
        
    y_hip_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
    y_hip_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    y_hip_mid = (y_hip_left + y_hip_right) / 2
    
    return aspect_ratio, y_hip_mid
# ----------------------------------------

VIDEO_SOURCE = 0 # 0 para webcam, 1 para DroidCam/Câmera USB
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
    exit()

print("Iniciando captura de vídeo... Pressione 'q' para sair.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    current_time = time.time()
    frame_height, frame_width, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    # Começa com 135 zeros
    current_features_flat = np.zeros(NUM_FEATURES)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 1. Calcular as features extras (v2)
        aspect_ratio, y_hip_mid = get_pose_features_realtime(frame_height, frame_width, results.pose_landmarks)
        
        delta_t = current_time - last_frame_time
        if delta_t == 0: delta_t = 1.0 / 30.0 # Evita divisão por zero
        
        velocity_y = (y_hip_mid - last_y_hip_mid) / delta_t
        last_y_hip_mid = y_hip_mid
        last_frame_time = current_time
        
        # 2. Pegar as features de pose (132)
        frame_landmarks_flat = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        ).flatten()
        
        # 3. Combinar tudo (135 features)
        current_features_flat = np.concatenate(
            (frame_landmarks_flat, [aspect_ratio, y_hip_mid, velocity_y])
        )
    else:
        # Se nenhuma pessoa for detectada, reseta a velocidade
        last_y_hip_mid = 0.5 

    # --- Lógica de Inferência da IA ---
    pose_sequence.append(current_features_flat)
    prediction_label = LABELS[0] 
    prediction_confidence = 0.0

    if len(pose_sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(np.array(pose_sequence), axis=0)
        prediction_proba = model.predict(input_data, verbose=0)[0][0]
        prediction_confidence = float(prediction_proba)
        
        if prediction_confidence > FALL_CONFIDENCE_THRESHOLD:
            confirmation_frames_counter += 1
            if confirmation_frames_counter >= CONFIRMATION_FRAMES_THRESHOLD and not is_alert_active:
                print(FALL_TRIGGER_MESSAGE, flush=True) 
                is_alert_active = True
                alert_start_time = time.time()
        else:
            confirmation_frames_counter = 0
        
        if not is_alert_active:
             prediction_label = LABELS[0] if prediction_confidence < FALL_CONFIDENCE_THRESHOLD else LABELS[1]

    # (O resto do código de "Gerenciamento do Alerta Visual" e "Exibição" 
    # é exatamente o mesmo do script anterior, copie e cole aqui)
    # ...
    # --- Gerenciamento do Alerta Visual ---
    if is_alert_active and (time.time() - alert_start_time > ALERT_DURATION_SECONDS):
        is_alert_active = False
        confirmation_frames_counter = 0 

    # --- Lógica de Desenho na Tela ---
    if is_alert_active:
        cv2.rectangle(frame, (0, 0), (frame_width, 60), (0,0,255), -1)
        cv2.putText(frame, f"ALERTA: {LABELS[1]}!", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    else:
        status_text = f"Status: {prediction_label} ({prediction_confidence:.0%})"
        if prediction_confidence > FALL_CONFIDENCE_THRESHOLD:
            color = (0, 165, 255) # Laranja (Confirmando...)
            status_text = f"Status: {LABELS[1]} ({prediction_confidence:.0%}) - Confirmando..."
        else:
            color = (0, 255, 0) # Verde
            
        cv2.rectangle(frame, (0, 0), (frame_width, 60), (0,0,0), -1) 
        cv2.putText(frame, status_text, (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # --- Exibição ---
    WINDOW_NAME = 'Detector de Quedas com IA (v6 - 135 Features)'
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# --- Finalização ---
print("Fechando aplicação...")
cap.release()
cv2.destroyAllWindows()
pose.close()
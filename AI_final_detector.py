import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import sys
import os
import threading 
import requests  
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Configurações da IA ---
MODEL_PATH = "modelo_quedas_seq_60.keras" 
SEQUENCE_LENGTH = 60 
NUM_FEATURES = 135
LABELS = ["NORMAL", "QUEDA"]
FALL_CONFIDENCE_THRESHOLD = 0.9 #0.8
ALERT_DURATION_SECONDS = 4 #5
CONFIRMATION_FRAMES_THRESHOLD = 30 #testar com 15

# --- Configurações da API (INTEGRAÇÃO BACK-END) ---
API_URL = "https://safe-home-backend.onrender.com/api/monitoring"

# DADOS REAIS SEU SISTEMA:
MONITORED_ID = "ID_DO_USUARIO_TESTE_123" 
LOCATION_NAME = "Sala de aula" 

# --- Função para Enviar ao Back-end (Em Segundo Plano) ---
def enviar_alerta_api(confianca):
    """
    Envia o POST para a API sem travar o vídeo.
    """
    try:
        # Montando o pacote de dados (Payload) conforme especificado
        payload = {
            "description": f"Queda detectada pela IA com {confianca:.1%} de confiança.",
            "severity": "high",       # Queda é sempre grave
            "monitored": MONITORED_ID,
            "location": LOCATION_NAME,
            "type": "fall"            # Tipo definido como 'fall'
        }
        
        print(f"Enviando alerta para o Back-end...")
        
        # Envia a requisição
        response = requests.post(API_URL, json=payload)
        
        # Verifica se deu certo
        if response.status_code == 200 or response.status_code == 201:
            print(f"✅ Sucesso! Back-end recebeu o alerta. (Status: {response.status_code})")
        else:
            print(f"Erro no Back-end: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Erro de conexão ao tentar enviar alerta: {e}")

# --- Inicialização ---
print(f"Carregando modelo '{MODEL_PATH}'...")
try:
    model = load_model(MODEL_PATH)
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"ERRO: Modelo não encontrado. {e}")
    exit()

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
pose_sequence = deque(maxlen=SEQUENCE_LENGTH)

is_alert_active = False
alert_start_time = 0
confirmation_frames_counter = 0

# Variáveis para features
last_y_hip_mid = 0.5
last_frame_time = time.time()

# --- Função Helper de Features ---
def get_pose_features_realtime(frame_height, frame_width, landmarks):
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

# --- Captura ---
VIDEO_SOURCE = 0 # Webcam ou DroidCam
cap = cv2.VideoCapture(VIDEO_SOURCE)

print("Sistema Iniciado. Conectado ao Back-end.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue

    current_time = time.time()
    frame_height, frame_width, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    current_features_flat = np.zeros(NUM_FEATURES)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        aspect_ratio, y_hip_mid = get_pose_features_realtime(frame_height, frame_width, results.pose_landmarks)
        delta_t = current_time - last_frame_time
        if delta_t == 0: delta_t = 1.0/30.0
        velocity_y = (y_hip_mid - last_y_hip_mid) / delta_t
        last_y_hip_mid = y_hip_mid
        last_frame_time = current_time
        
        frame_landmarks_flat = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten()
        current_features_flat = np.concatenate((frame_landmarks_flat, [aspect_ratio, y_hip_mid, velocity_y]))
    else:
        last_y_hip_mid = 0.5

    # --- IA ---
    pose_sequence.append(current_features_flat)
    prediction_label = LABELS[0] 
    prediction_confidence = 0.0

    if len(pose_sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(np.array(pose_sequence), axis=0)
        prediction_proba = model.predict(input_data, verbose=0)[0][0]
        prediction_confidence = float(prediction_proba)
        
        if prediction_confidence > FALL_CONFIDENCE_THRESHOLD:
            confirmation_frames_counter += 1
            
            # GATILHO DO ALERTA
            if confirmation_frames_counter >= CONFIRMATION_FRAMES_THRESHOLD and not is_alert_active:
                
                # 1. Ativa o estado visual
                is_alert_active = True
                alert_start_time = time.time()
                
                # 2. DISPARA O POST PARA O BACK-END (USANDO THREAD)
                # O 'target' é a função, 'args' são os argumentos que ela recebe
                threading.Thread(target=enviar_alerta_api, args=(prediction_confidence,)).start()
                
        else:
            confirmation_frames_counter = 0
        
        if not is_alert_active:
             prediction_label = LABELS[0] if prediction_confidence < FALL_CONFIDENCE_THRESHOLD else LABELS[1]

    # --- Visual ---
    if is_alert_active and (time.time() - alert_start_time > ALERT_DURATION_SECONDS):
        is_alert_active = False
        confirmation_frames_counter = 0 

    if is_alert_active:
        cv2.rectangle(frame, (0, 0), (frame_width, 60), (0,0,255), -1)
        cv2.putText(frame, f"ALERTA: {LABELS[1]}!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    else:
        status_text = f"Status: {prediction_label} ({prediction_confidence:.0%})"
        color = (0, 165, 255) if prediction_confidence > FALL_CONFIDENCE_THRESHOLD else (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (frame_width, 60), (0,0,0), -1) 
        cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    cv2.imshow('Detector Safe-Home (Integrado)', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose.close()
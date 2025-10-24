import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from tensorflow.keras.models import load_model
import sys  # Precisamos disso para o flush

# --- Configurações ---
MODEL_PATH = "modelo_quedas.keras"


# Mensagem exata que será enviada para o stdout para o back-end capturar

FALL_TRIGGER_MESSAGE = "FALL_DETECTED_TRIGGER"

print("Carregando modelo...")
try:
    model = load_model(MODEL_PATH)
    print(f"Modelo '{MODEL_PATH}' carregado com sucesso.")
except Exception as e:
    print(f"ERRO: Não foi possível carregar o modelo em '{MODEL_PATH}'.")
    print(f"Verifique se o arquivo existe e se o tensorflow está instalado.")
    exit()

SEQUENCE_LENGTH = 30
LABELS = ["NORMAL", "QUEDA"]
FALL_CONFIDENCE_THRESHOLD = 0.8  # Você pode ajustar isso (ex: 0.7)
ALERT_DURATION_SECONDS = 5

# --- Inicialização ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
pose_sequence = deque(maxlen=SEQUENCE_LENGTH)
is_alert_active = False
alert_start_time = 0

# --- Captura de Vídeo ---
VIDEO_SOURCE = 0
# VIDEO_SOURCE = "http://192.168.141.91:8080/video"
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
    exit()

print("Iniciando captura de vídeo... Pressione 'q' para sair.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        ### Silenciar prints de "ruído" ###
        # print("Ignorando frame vazio.") # Silenciado para o back-end
        continue

    frame_height, frame_width, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    current_pose_features = np.zeros((33, 4))

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66),
                                   thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230),
                                   thickness=2, circle_radius=2)
        )

        frame_landmarks = []
        for lm in results.pose_landmarks.landmark:
            frame_landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
        current_pose_features = np.array(frame_landmarks)

    # --- Lógica de Inferência da IA ---
    flat_pose_features = current_pose_features.flatten()
    pose_sequence.append(flat_pose_features)
    prediction_label = LABELS[0]
    prediction_confidence = 0.0

    if len(pose_sequence) == SEQUENCE_LENGTH:
        input_data = np.expand_dims(np.array(pose_sequence), axis=0)
        prediction_proba = model.predict(input_data)[0][0]
        prediction_confidence = float(prediction_proba)

        if prediction_confidence > FALL_CONFIDENCE_THRESHOLD:
            prediction_label = LABELS[1]
            if not is_alert_active:

                # Esta é a única mensagem que seu back-end precisa ouvir.
                # flush=True força o Python a enviar a mensagem IMEDIATAMENTE.
                print(FALL_TRIGGER_MESSAGE, flush=True)

                is_alert_active = True
                alert_start_time = time.time()

    # --- Gerenciamento do Alerta Visual ---
    if is_alert_active and (time.time() - alert_start_time > ALERT_DURATION_SECONDS):
        is_alert_active = False
        ### Silenciar prints de "ruído" ###
        # print("Alerta resetado.") # Silenciado para o back-end

    # A parte VISUAL (a janela)
    if is_alert_active:
        cv2.rectangle(frame, (0, 0), (frame_width, 60), (0, 0, 255), -1)
        cv2.putText(frame, f"ALERTA: {prediction_label}!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    else:
        # Você pode comentar as 4 linhas abaixo se não quiser nem o
        # status "NORMAL" na tela, apenas o alerta vermelho.
        # ({prediction_confidence:.0%})"
        status_text = f"Status: {LABELS[0]} "
        color = (0, 255, 0)
        cv2.rectangle(frame, (0, 0), (frame_width, 60), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # --- Exibição ---
    WINDOW_NAME = 'Detector de Quedas com IA (v4)'
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# --- Finalização ---
print("Fechando aplicação...")
cap.release()
cv2.destroyAllWindows()
pose.close()

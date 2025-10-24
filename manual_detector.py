import cv2
import mediapipe as mp
import time
import math

# --- Inicialização ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Configurações da Lógica de Queda v3 ---

# ESTADO "PERMANÊNCIA"
FALL_TIME_THRESHOLD = 3.0  # Segundos deitado *após impacto* para considerar queda

# ESTADO "IMPACTO" (Velocidade Vertical)
VELOCITY_THRESHOLD = 0.5  # Limite de velocidade de queda (unidades norm/seg)

# ESTADO "DEITADO" (Aspect Ratio)
# Se largura > altura * threshold, considera deitado.
# 1.0 = largura > altura
# 1.2 = largura > 120% da altura (mais robusto para agachamentos)
ASPECT_RATIO_THRESHOLD = 1.2

# --- Variáveis de Rastreamento de Estado ---
is_lying = False
time_lying_started = 0
potential_impact_detected = False

# Variáveis para cálculo da velocidade
last_y_hip_mid = 0
last_frame_time = 0

# --- Captura de Vídeo ---
VIDEO_SOURCE = 0  # 0 para webcam
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignorando frame vazio.")
        continue

    frame_height, frame_width, _ = frame.shape
    current_time = time.time()

    # --- Processamento de IA (MediaPipe) ---
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

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

        # --- Lógica de Detecção de Queda v3 ---
        try:
            landmarks = results.pose_landmarks.landmark

            # --- 1. Cálculo de Velocidade (Detecção de Impacto) ---
            # (Igual à V2)
            y_hip_mid = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y +
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2

            delta_t = current_time - last_frame_time
            if last_frame_time > 0 and delta_t > 0.01:
                delta_y = y_hip_mid - last_y_hip_mid
                velocity_y = delta_y / delta_t

                if velocity_y > VELOCITY_THRESHOLD:
                    if not potential_impact_detected:
                        print(
                            f"LOG: Impacto potencial detectado! Vel_Y: {velocity_y:.2f}")
                    potential_impact_detected = True

            last_y_hip_mid = y_hip_mid
            last_frame_time = current_time

            # --- 2. Verificação de Estado (Aspect Ratio) ---
            # Calcular a "caixa" (bounding box) da pose
            x_min, y_min = frame_width, frame_height
            x_max, y_max = 0, 0

            for landmark in landmarks:
                # Usa apenas landmarks visíveis
                if landmark.visibility < 0.3:
                    continue

                # Converte coordenadas normalizadas para pixels
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Adiciona uma pequena margem
            x_min = max(0, x_min - 10)
            y_min = max(0, y_min - 10)
            x_max = min(frame_width, x_max + 10)
            y_max = min(frame_height, y_max + 10)

            # Calcular altura e largura do corpo
            body_height = y_max - y_min
            body_width = x_max - x_min

            # Desenha a caixa no frame (para debug)
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)

            # Lógica: "Está deitado?"
            is_currently_lying = (
                body_width > body_height * ASPECT_RATIO_THRESHOLD) and (body_height > 0)

            if is_currently_lying:
                # Pessoa está na posição "deitada"

                # --- 3. Verificação de Permanência (Pós-Impacto) ---
                if potential_impact_detected:
                    # SÓ inicia o timer se houve um impacto ANTES
                    if not is_lying:
                        is_lying = True
                        time_lying_started = time.time()
                        print("LOG: Posição deitada PÓS-IMPACTO. Iniciando timer...")
                    else:
                        time_elapsed = time.time() - time_lying_started
                        cv2.putText(frame, f"Deitado por: {time_elapsed:.1f}s", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        if time_elapsed > FALL_TIME_THRESHOLD:
                            # --- ALERTA DE QUEDA ---
                            print(
                                "ALERTA: QUEDA DETECTADA (IMPACTO + FORMA + PERMANÊNCIA)!")
                            cv2.rectangle(
                                frame, (0, 0), (frame_width, 60), (0, 0, 255), -1)
                            cv2.putText(frame, "ALERTA DE QUEDA!", (10, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            else:
                # Pessoa NÃO está deitada (levantou ou está em pé/sentada)
                if is_lying or potential_impact_detected:
                    print("LOG: Pessoa levantou ou não está deitada. Resetando estado.")

                # --- O RESET QUE FALTAVA ---
                is_lying = False
                time_lying_started = 0
                potential_impact_detected = False

        except Exception as e:
            # Em caso de erro (ex: pessoa saiu do frame), reseta tudo
            # print(f"Erro: {e}") # Descomente para debug
            is_lying = False
            time_lying_started = 0
            potential_impact_detected = False

    # --- Exibição ---
    WINDOW_NAME = 'Monitoramento de Quedas - Protótipo v3 (com Aspect Ratio)'
    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(5) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        break

# --- Finalização ---
cap.release()
cv2.destroyAllWindows()
pose.close()

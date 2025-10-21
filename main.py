import cv2
import mediapipe as mp
import time

# --- Inicialização ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Configurações da Lógica de Queda v2 ---

# ESTADO "DEITADO"
# Se os quadris estiverem abaixo de 70% da altura do frame
LYING_POSTURE_THRESHOLD = 0.7

# ESTADO "PERMANÊNCIA"
FALL_TIME_THRESHOLD = 2.0  # Segundos deitado *após impacto* para considerar queda

# ESTADO "IMPACTO" (Velocidade Vertical)
# Este valor é em "unidades de frame normalizadas por segundo"
# 0.5 significa mover-se metade da altura do frame em 1 segundo.
# Pode precisar de ajuste (aumente se disparar muito fácil, diminua se não pegar quedas)
VELOCITY_THRESHOLD = 0.6

# --- Variáveis de Rastreamento de Estado ---
is_lying = False
time_lying_started = 0
potential_impact_detected = False

# Variáveis para cálculo da velocidade
last_y_hip_mid = 0
last_frame_time = 0

# --- Captura de Vídeo ---
# 0 para webcam, ou "caminho/para/video.mp4"
VIDEO_SOURCE = 0
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"Erro: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
    exit()

while cap.isOpened():
    is_video_file = isinstance(VIDEO_SOURCE, str)

    success, frame = cap.read()
    if not success:
        if is_video_file:
            print("Fim do vídeo. Reiniciando...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        else:
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

        # --- Lógica de Detecção de Queda v2 ---
        try:
            landmarks = results.pose_landmarks.landmark
            y_hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            y_hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            y_hip_mid = (y_hip_left + y_hip_right) / 2

            # --- 1. Cálculo de Velocidade (Detecção de Impacto) ---
            delta_t = current_time - last_frame_time

            # Evitar divisão por zero no primeiro frame ou se o tempo for muito curto
            if last_frame_time > 0 and delta_t > 0.01:
                # delta_y > 0 significa movimento para BAIXO
                delta_y = y_hip_mid - last_y_hip_mid
                # Velocidade em unidades normalizadas por segundo
                velocity_y = delta_y / delta_t

                # Exibir velocidade para debug
                cv2.putText(frame, f"Vel_Y: {velocity_y:.2f}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if velocity_y > VELOCITY_THRESHOLD:
                    if not potential_impact_detected:
                        print(
                            f"LOG: Impacto potencial detectado! Vel_Y: {velocity_y:.2f}")
                    potential_impact_detected = True

            # Atualizar valores para o próximo frame
            last_y_hip_mid = y_hip_mid
            last_frame_time = current_time

            # --- 2. Verificação de Estado (Deitado) ---
            current_lying_threshold = frame_height * LYING_POSTURE_THRESHOLD
            y_hip_pixel = y_hip_mid * frame_height

            # Desenha a linha limite
            cv2.line(frame, (0, int(current_lying_threshold)),
                     (frame_width, int(current_lying_threshold)),
                     (0, 255, 255), 2)

            if y_hip_pixel > current_lying_threshold:
                # Pessoa está na posição "deitada"

                # --- 3. Verificação de Permanência (Pós-Impacto) ---
                if potential_impact_detected:
                    # SÓ inicia o timer se houve um impacto ANTES
                    if not is_lying:
                        is_lying = True
                        time_lying_started = time.time()
                        print("LOG: Posição deitada PÓS-IMPACTO. Iniciando timer...")
                    else:
                        # Se continua deitado, verificar o tempo
                        time_elapsed = time.time() - time_lying_started

                        cv2.putText(frame, f"Deitado por: {time_elapsed:.1f}s", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                        if time_elapsed > FALL_TIME_THRESHOLD:
                            # --- ALERTA DE QUEDA ---
                            print("ALERTA: QUEDA DETECTADA (IMPACTO + PERMANÊNCIA)!")
                            cv2.rectangle(
                                frame, (0, 0), (frame_width, 60), (0, 0, 255), -1)
                            cv2.putText(frame, "ALERTA DE QUEDA!", (10, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            else:
                # Pessoa não está deitada (levantou)
                if is_lying or potential_impact_detected:
                    print("LOG: Pessoa levantou ou não está deitada. Resetando estado.")
                # Reseta todos os estados
                is_lying = False
                time_lying_started = 0
                potential_impact_detected = False

        except Exception as e:
            print(f"Erro ao processar landmarks: {e}")
            is_lying = False
            time_lying_started = 0
            potential_impact_detected = False

    # --- Exibição ---
    cv2.imshow('Monitoramento de Quedas - Protótipo v2 (com Velocidade)', frame)

    delay = 1 if is_video_file else 5
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# --- Finalização ---
cap.release()
cv2.destroyAllWindows()
pose.close()

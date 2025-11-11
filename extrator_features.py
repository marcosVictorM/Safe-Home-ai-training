import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm # Para a barra de progresso

# --- Configuração ---
SOURCE_DIR = "dados_treinamento"
FEATURES_DIR = "features"
LABELS = ["quedas", "nao_quedas"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, 
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# --- Novas Funções de Engenharia de Features ---

def get_pose_features(frame_height, frame_width, landmarks):
    """Calcula a Bounding Box e a altura do quadril."""
    
    # 1. Bounding Box (para aspect ratio)
    x_min, y_min = frame_width, frame_height
    x_max, y_max = 0, 0
    
    for lm in landmarks.landmark:
        if lm.visibility < 0.3: # Ignora pontos invisíveis
            continue
        
        # Converte normalizado para pixels
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
            
    body_height = y_max - y_min
    body_width = x_max - x_min
    
    # Evita divisão por zero
    if body_height == 0:
        aspect_ratio = 0
    else:
        aspect_ratio = body_width / body_height # vertical=pequeno, horizontal=grande
        
    # 2. Altura do Quadril (para proximidade do chão)
    y_hip_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y
    y_hip_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y
    y_hip_mid = (y_hip_left + y_hip_right) / 2 # 0.0 = topo, 1.0 = chão
    
    return aspect_ratio, y_hip_mid

# --- Função Principal Modificada ---

def extrair_features():
    for label in LABELS:
        output_folder = os.path.join(FEATURES_DIR, label)
        os.makedirs(output_folder, exist_ok=True)

    print(f"Pastas de saída criadas em: '{FEATURES_DIR}'")
    
    for label in LABELS:
        input_folder = os.path.join(SOURCE_DIR, label)
        output_folder = os.path.join(FEATURES_DIR, label)
        
        try:
            video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            print(f"\nEncontrados {len(video_files)} vídeos em '{input_folder}'")
        except FileNotFoundError:
            print(f"ERRO: Pasta '{input_folder}' não encontrada.")
            return

        for video_name in tqdm(video_files, desc=f"Processando '{label}'"):
            video_path = os.path.join(input_folder, video_name)
            output_filename = os.path.splitext(video_name)[0] + ".npy"
            output_path = os.path.join(output_folder, output_filename)
            
            if os.path.exists(output_path):
                continue
                
            video_landmarks = []
            
            # --- Variáveis para features de tempo (velocidade) ---
            last_y_hip_mid = 0.5 # Começa no meio
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            # Evita divisão por zero se o FPS não for encontrado
            if fps == 0: 
                fps = 30.0 
            delta_t = 1.0 / fps # Tempo (em seg) entre frames

            if not cap.isOpened():
                print(f"Aviso: Não foi possível abrir o vídeo {video_path}")
                continue
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break 
                
                frame_height, frame_width, _ = frame.shape
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                if results.pose_landmarks:
                    # 1. Pegar os 33 pontos de pose (132 features)
                    frame_landmarks_flat = np.array(
                        [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                    ).flatten() # Achatado para (132,)
                    
                    # 2. Calcular as features de pose (2 features)
                    aspect_ratio, y_hip_mid = get_pose_features(frame_height, frame_width, results.pose_landmarks)
                    
                    # 3. Calcular a feature de velocidade (1 feature)
                    # (delta_y / delta_t) -> (y_atual - y_anterior) / (1/fps)
                    # y > 0 = movimento para baixo
                    velocity_y = (y_hip_mid - last_y_hip_mid) / delta_t
                    last_y_hip_mid = y_hip_mid # Atualiza para o próximo frame
                    
                    # 4. Combinar tudo em um único array de 135 features
                    all_features = np.concatenate(
                        (frame_landmarks_flat, [aspect_ratio, y_hip_mid, velocity_y])
                    )
                    video_landmarks.append(all_features)
                    
                else:
                    # Se não achou pessoa, salvamos 135 zeros
                    video_landmarks.append(np.zeros(132 + 3)) # 132 (pose) + 3 (extras)
            
            cap.release()
            
            if len(video_landmarks) > 0:
                np_video_data = np.array(video_landmarks)
                np.save(output_path, np_video_data)

    print("\n--- Extração de Características (v2) Concluída! ---")

# --- Execução ---
if __name__ == "__main__":
    # IMPORTANTE: Delete a pasta 'features' antiga antes de rodar!
    print("ATENÇÃO: Se você não deletou a pasta 'features' antiga,")
    print("os arquivos antigos podem permanecer.")
    print("Pressione Enter para continuar ou Ctrl+C para cancelar...")
    # input() # Descomente se quiser uma pausa de segurança
    
    extrair_features()
    pose.close()
import cv2
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm  # Para a barra de progresso

# --- Configuração ---

# Pastas de entrada (onde estão seus vídeos)
SOURCE_DIR = "dados_treinamento"
# Pastas de saída (onde salvaremos os dados da pose)
FEATURES_DIR = "features"

# Nossas duas classes
LABELS = ["quedas", "nao_quedas"]

# Configuração do MediaPipe
mp_pose = mp.solutions.pose
# Usamos static_image_mode=True para processar cada frame
# individualmente (melhor para vídeos, mais lento)
pose = mp_pose.Pose(static_image_mode=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

# --- Função Principal ---


def extrair_features():
    """
    Varre as pastas de labels, processa cada vídeo e salva
    as 'features' (pontos de pose) em um arquivo .npy.
    """

    # 1. Criar as pastas de saída se elas não existirem
    for label in LABELS:
        output_folder = os.path.join(FEATURES_DIR, label)
        os.makedirs(output_folder, exist_ok=True)

    print(f"Pastas de saída criadas em: '{FEATURES_DIR}'")

    # 2. Processar cada classe (label)
    for label in LABELS:
        input_folder = os.path.join(SOURCE_DIR, label)
        output_folder = os.path.join(FEATURES_DIR, label)

        # Pegar a lista de todos os vídeos na pasta
        try:
            video_files = [f for f in os.listdir(input_folder) if f.endswith(
                ('.mp4', '.avi', '.mov', '.mkv'))]
            print(
                f"\nEncontrados {len(video_files)} vídeos em '{input_folder}'")
        except FileNotFoundError:
            print(
                f"ERRO: Pasta '{input_folder}' não encontrada. Você criou as pastas 'dados_treinamento/quedas' e 'dados_treinamento/nao_quedas'?")
            return

        # 3. Loop principal (com barra de progresso - tqdm)
        for video_name in tqdm(video_files, desc=f"Processando '{label}'"):
            video_path = os.path.join(input_folder, video_name)

            # Nome do arquivo de saída (ex: video_01.mp4 -> video_01.npy)
            output_filename = os.path.splitext(video_name)[0] + ".npy"
            output_path = os.path.join(output_folder, output_filename)

            # Pular se já processamos este vídeo antes
            if os.path.exists(output_path):
                continue

            # Lista para guardar os dados de todos os frames DESTE vídeo
            video_landmarks = []

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Aviso: Não foi possível abrir o vídeo {video_path}")
                continue

            # 4. Ler cada frame do vídeo
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break  # Fim do vídeo

                # Converter para RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Processar com o MediaPipe
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    # Se achou uma pessoa, extrair os 33 pontos
                    frame_landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        # Salvamos (x, y, z, visibilidade)
                        frame_landmarks.append(
                            [lm.x, lm.y, lm.z, lm.visibility])
                    video_landmarks.append(frame_landmarks)
                else:
                    # Se NÃO achou pessoa no frame, salvamos 33 "zeros"
                    # Isso é CRUCIAL para a IA entender que a pessoa sumiu
                    # A forma é (33 pontos, 4 coordenadas)
                    video_landmarks.append(np.zeros((33, 4)))

            cap.release()

            # 5. Salvar os dados DESTE vídeo em um arquivo .npy
            if len(video_landmarks) > 0:
                # Converter a lista de frames para um array NumPy
                # Forma final: (num_frames, 33, 4)
                np_video_data = np.array(video_landmarks)

                # Salvar o arquivo
                np.save(output_path, np_video_data)

    print("\n--- Extração de Características Concluída! ---")
    print(
        f"Todos os dados foram processados e salvos na pasta '{FEATURES_DIR}'.")

# --- Execução ---


if __name__ == "__main__":
    extrair_features()
    pose.close()

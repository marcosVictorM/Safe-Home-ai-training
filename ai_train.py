import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Configurações ---
FEATURES_DIR = "features"
# IMPORTANTE: A ordem importa!
# "nao_quedas" será o RÓTULO 0 (classe negativa)
# "quedas" será o RÓTULO 1 (classe positiva)
LABELS = ["nao_quedas", "quedas"]

# Quantos frames vamos analisar por vez (ex: 30 frames = 1 segundo de vídeo a 30fps)
SEQUENCE_LENGTH = 30
# Modelo final que será salvo
MODEL_NAME = "modelo_quedas.keras"


def load_data():
    """
    Carrega os arquivos .npy da pasta 'features', os fatia em sequências
    e aplica os rótulos corretos (0 ou 1).
    """
    X = []  # Lista para as "sequências" (features)
    y = []  # Lista para os "rótulos" (labels)

    print("Carregando e processando dados...")

    # Loop sobre "nao_quedas" (label 0) e "quedas" (label 1)
    for label_index, label in enumerate(LABELS):
        folder_path = os.path.join(FEATURES_DIR, label)

        # Pega a lista de arquivos .npy
        file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

        # tqdm é para a barra de progresso
        for filename in tqdm(file_list, desc=f"Lendo '{label}'"):
            file_path = os.path.join(folder_path, filename)

            # Carrega os dados do vídeo (ex: forma (250, 33, 4))
            video_data = np.load(file_path)

            # Achata os dados de pose para cada frame
            # (33, 4) -> (132)
            num_frames = video_data.shape[0]
            # Nova forma será (num_frames, 132)
            video_data_flat = video_data.reshape(
                num_frames, -1)  # -1 infere 33*4=132

            # Aqui está o "fatiamento" (sliding window)
            # Vamos criar múltiplas amostras de 30 frames a partir de um único vídeo
            # Usamos um passo de 15 (metade da sequência) para criar sobreposição
            step = SEQUENCE_LENGTH // 2

            for i in range(0, num_frames - SEQUENCE_LENGTH, step):
                # Pega uma fatia de 30 frames
                chunk = video_data_flat[i: i + SEQUENCE_LENGTH]

                # Garante que a fatia tenha o tamanho exato (descarta sobras)
                if chunk.shape[0] == SEQUENCE_LENGTH:
                    X.append(chunk)
                    # 0 para "nao_quedas", 1 para "quedas"
                    y.append(label_index)

    print("Carregamento concluído.")

    # Converte as listas para arrays NumPy, que é o que o TensorFlow espera
    X_np = np.array(X)
    y_np = np.array(y)

    return X_np, y_np


def plot_history(history):
    """Gera gráficos de acurácia e perda do treinamento."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Gráfico de Acurácia
    ax1.plot(history.history['accuracy'], label='Acurácia (Treino)')
    ax1.plot(history.history['val_accuracy'], label='Acurácia (Validação)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia')
    ax1.legend()
    ax1.set_title('Histórico de Acurácia')

    # Gráfico de Perda
    ax2.plot(history.history['loss'], label='Perda (Treino)')
    ax2.plot(history.history['val_loss'], label='Perda (Validação)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda (Loss)')
    ax2.legend()
    ax2.set_title('Histórico de Perda')

    plt.savefig('grafico_treinamento.png')
    print("\nGráfico do histórico salvo como 'grafico_treinamento.png'")


def main():
    # 1. Carregar os Dados
    X, y = load_data()

    if X.shape[0] == 0:
        print("\n--- ERRO ---")
        print(
            f"Nenhum dado foi carregado. Verifique se a pasta '{FEATURES_DIR}' contém as subpastas '{LABELS[0]}' e '{LABELS[1]}', e se elas contêm os arquivos .npy.")
        return

    # Ex: (5210, 30, 132)
    print(f"\nForma final dos dados (Amostras, Frames, Features): {X.shape}")
    print(f"Forma final dos rótulos (Amostras,): {y.shape}")  # Ex: (5210,)

    # 2. Dividir em Treino e Teste (80% treino, 20% teste)
    # stratify=y garante que a proporção de quedas/nao_quedas seja a mesma
    # tanto no treino quanto no teste.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nAmostras de treino: {X_train.shape[0]}")
    print(f"Amostras de teste: {X_test.shape[0]}")

    # 3. Construir o Modelo (O "Cérebro")

    # Forma da nossa entrada: (30 frames, 132 features por frame)
    input_shape = (SEQUENCE_LENGTH, X.shape[2])  # (30, 132)

    model = Sequential([
        # Camada LSTM para processar a sequência de poses
        # input_shape é necessário na primeira camada
        LSTM(64, return_sequences=True, input_shape=input_shape),
        # Dropout "desliga" neurônios para evitar superespecialização
        Dropout(0.4),

        # Segunda camada LSTM
        LSTM(64),
        Dropout(0.4),

        # Camada densa (normal) para classificação
        Dense(32, activation='relu'),

        # Camada de Saída: 1 neurônio com 'sigmoid'
        # Sigmoid nos dá uma probabilidade entre 0.0 (não-queda) e 1.0 (queda)
        Dense(1, activation='sigmoid')
    ])

    # 4. Compilar o Modelo
    model.compile(
        optimizer='adam',                # Otimizador padrão e eficiente
        # Perda ideal para classificação binária (0 ou 1)
        loss='binary_crossentropy',
        metrics=['accuracy']             # Métrica que queremos observar
    )

    model.summary()  # Imprime um resumo da arquitetura do modelo

    # 5. Treinar o Modelo!
    print("\n--- Iniciando o Treinamento ---")
    # batch_size=32 -> processa 32 amostras por vez
    # epochs=20 -> vai rodar 20 "voltas" sobre todo o dataset de treino
    # validation_data -> usa os dados de teste para ver se o modelo está generalizando
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=20,
        validation_data=(X_test, y_test)
    )
    print("--- Treinamento Concluído ---")

    # 6. Avaliar o Modelo (Parte mais importante)
    print("\n--- Avaliação do Modelo ---")
    # Pega as previsões do modelo para os dados de teste
    y_pred_proba = model.predict(X_test)
    # Converte probabilidades (ex: 0.95) em classes (1)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Imprime o Relatório de Classificação
    # "recall" para "quedas" é o mais importante!
    # Diz: "De todas as quedas reais, quantas o modelo acertou?"
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    # Imprime a Matriz de Confusão
    # [ [Verdadeiro Negativo, Falso Positivo],
    #   [Falso Negativo   , Verdadeiro Positivo] ]
    print("\nMatriz de Confusão (Linhas=Real, Colunas=Previsão):")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 7. Salvar o gráfico e o modelo
    plot_history(history)
    model.save(MODEL_NAME)
    print(f"\nModelo salvo com sucesso como '{MODEL_NAME}'!")


if __name__ == "__main__":
    main()

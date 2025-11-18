import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# --- Configurações ---
FEATURES_DIR = "features"
LABELS = ["nao_quedas", "quedas"]
# Nossas "cobaias" - as janelas de tempo que vamos testar
SEQUENCIAS_PARA_TESTAR = [30, 45, 60]
NUM_FEATURES = 135 # Nosso dado V2 (132 pose + 3 extras)
EPOCHS = 30 # Manter 30 épocas para um teste justo (ou 40 se preferir)

# --- Função de Carregar Dados (Modificada para aceitar seq_len) ---
def load_data(seq_len):
    X = []
    y = []
    print(f"\nCarregando dados para SEQUÊNCIA DE {seq_len} FRAMES...")
    
    for label_index, label in enumerate(LABELS):
        folder_path = os.path.join(FEATURES_DIR, label)
        file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        
        for filename in tqdm(file_list, desc=f"Lendo '{label}' (seq={seq_len})"):
            file_path = os.path.join(folder_path, filename)
            video_data = np.load(file_path) # Forma: (num_frames, 135)
            
            if video_data.shape[1] != NUM_FEATURES:
                print(f"Erro! Dados inválidos. Rode o 'extrator_features.py' v2.")
                return None, None

            # Fatiamento (sliding window)
            step = seq_len // 2
            
            for i in range(0, video_data.shape[0] - seq_len, step):
                chunk = video_data[i : i + seq_len]
                
                if chunk.shape[0] == seq_len:
                    X.append(chunk)
                    y.append(label_index)

    print("Carregamento concluído.")
    X_np, y_np = np.array(X), np.array(y)
    return X_np, y_np

# --- Função para Construir o Modelo (aceita input_shape) ---
def build_model(input_shape):
    model = Sequential([
        # Esta é a nossa arquitetura V2 "campeã"
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.4),
        LSTM(64),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# --- Função de Plotar (salva com nome único) ---
def plot_history(history, seq_len):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Acurácia (Treino)')
    ax1.plot(history.history['val_accuracy'], label='Acurácia (Validação)')
    ax1.set_xlabel('Época'); ax1.set_ylabel('Acurácia'); ax1.legend()
    ax1.set_title(f'Histórico de Acurácia (Seq={seq_len})')
    ax2.plot(history.history['loss'], label='Perda (Treino)')
    ax2.plot(history.history['val_loss'], label='Perda (Validação)')
    ax2.set_xlabel('Época'); ax2.set_ylabel('Perda (Loss)'); ax2.legend()
    ax2.set_title(f'Histórico de Perda (Seq={seq_len})')
    
    # Salva o gráfico com o nome da sequência
    filename = f'grafico_treinamento_seq_{seq_len}.png'
    plt.savefig(filename)
    print(f"\nGráfico do histórico salvo como '{filename}'")

# --- Loop Principal de Teste ---
def main():
    print("--- INICIANDO TESTE DE HIPERPARÂMETROS (SEQUENCE_LENGTH) ---")
    start_time_total = time.time()
    
    # Dicionário para guardar os resultados de cada teste
    resultados_finais = {}

    for seq_len in SEQUENCIAS_PARA_TESTAR:
        print("\n" + "="*70)
        print(f"INICIANDO TESTE COM SEQUENCE_LENGTH = {seq_len}")
        print("="*70 + "\n")
        
        start_time_teste = time.time()
        
        # 1. Carregar os Dados (com a nova seq_len)
        X, y = load_data(seq_len)
        if X is None or X.shape[0] == 0:
            print(f"Falha ao carregar dados para seq={seq_len}. Pulando.")
            continue
            
        print(f"Forma dos dados (seq={seq_len}): {X.shape}")

        # 2. Calcular Pesos de Classe (para esta nova divisão de dados)
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights = dict(zip(np.unique(y), weights))
        print("Pesos de classe calculados:")
        print(f"  Peso 0 (nao_quedas): {class_weights[0]:.2f}")
        print(f"  Peso 1 (quedas): {class_weights[1]:.2f}")

        # 3. Dividir em Treino e Teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        # 4. Construir o Modelo
        input_shape = (seq_len, NUM_FEATURES)
        model = build_model(input_shape)
        # model.summary() # Descomente se quiser ver a arquitetura

        # 5. Treinar
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=EPOCHS,
            validation_data=(X_test, y_test),
            class_weight=class_weights,
            verbose=2 # (verbose=2 é mais limpo para um loop)
        )

        # 6. Avaliar
        print(f"\n--- AVALIAÇÃO (Seq={seq_len}) ---")
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        report = classification_report(y_test, y_pred, target_names=LABELS)
        matrix = confusion_matrix(y_test, y_pred)
        
        print("\nRelatório de Classificação:")
        print(report)
        print("\nMatriz de Confusão:")
        print(matrix)
        
        # 7. Salvar tudo
        plot_history(history, seq_len)
        model_name = f'modelo_quedas_seq_{seq_len}.keras'
        model.save(model_name)
        print(f"Modelo salvo como '{model_name}'")
        
        # Guardar resultados
        resultados_finais[seq_len] = (report, matrix)
        
        end_time_teste = time.time()
        print(f"Tempo do teste (seq={seq_len}): {(end_time_teste - start_time_teste) / 60:.1f} minutos")

    # --- Resumo Final ---
    print("\n" + "="*70)
    print("--- TESTE DE HIPERPARÂMETROS CONCLUÍDO ---")
    print("="*70 + "\n")
    
    for seq_len, (report, matrix) in resultados_finais.items():
        print(f"--- RESULTADO PARA SEQUENCE_LENGTH = {seq_len} ---")
        print(report)
        print(matrix)
        print("\n")
        
    end_time_total = time.time()
    print(f"Tempo total do experimento: {(end_time_total - start_time_total) / 60:.1f} minutos")

if __name__ == "__main__":
    main()
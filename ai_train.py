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

# --- Configurações ---
FEATURES_DIR = "features"
# IMPORTANTE: A ordem importa!
# "nao_quedas" será o RÓTULO 0 (classe negativa)
# "quedas" será o RÓTULO 1 (classe positiva)
LABELS = ["nao_quedas", "quedas"] 

# Usamos 45 para dar mais contexto (ajuda a diferenciar queda de deitar)
SEQUENCE_LENGTH = 45
MODEL_NAME = "modelo_quedas_v2.keras" # Salvar como um novo modelo

# --- Função de Carregar Dados (Modificada) ---
def load_data():
    X = []
    y = []

    print("Carregando e processando dados (v2)...")
    
    for label_index, label in enumerate(LABELS):
        folder_path = os.path.join(FEATURES_DIR, label)
        
        try:
            file_list = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        except FileNotFoundError:
            print(f"\n--- ERRO ---")
            print(f"Pasta '{folder_path}' não encontrada.")
            print("Você rodou o novo 'extrator_features.py'?")
            return None, None
        
        for filename in tqdm(file_list, desc=f"Lendo '{label}'"):
            file_path = os.path.join(folder_path, filename)
            
            video_data = np.load(file_path) # Forma: (num_frames, 135)
            
            # Verificar se os dados têm as 135 features
            if video_data.shape[1] != 135:
                print(f"Erro! Arquivo {filename} tem forma {video_data.shape}!")
                print("Delete a pasta 'features' e rode o 'extrator_features.py' v2.")
                return None, None

            # Fatiamento (sliding window)
            step = SEQUENCE_LENGTH // 2 # 22
            
            for i in range(0, video_data.shape[0] - SEQUENCE_LENGTH, step):
                chunk = video_data[i : i + SEQUENCE_LENGTH]
                
                if chunk.shape[0] == SEQUENCE_LENGTH:
                    X.append(chunk)
                    y.append(label_index) # 0 ou 1

    print("Carregamento concluído.")
    
    X_np = np.array(X)
    y_np = np.array(y)
    
    return X_np, y_np

# (função plot_history não muda, pode copiar do seu script antigo)
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Acurácia (Treino)')
    ax1.plot(history.history['val_accuracy'], label='Acurácia (Validação)')
    ax1.set_xlabel('Época'); ax1.set_ylabel('Acurácia'); ax1.legend()
    ax1.set_title('Histórico de Acurácia')
    ax2.plot(history.history['loss'], label='Perda (Treino)')
    ax2.plot(history.history['val_loss'], label='Perda (Validação)')
    ax2.set_xlabel('Época'); ax2.set_ylabel('Perda (Loss)'); ax2.legend()
    ax2.set_title('Histórico de Perda')
    plt.savefig('grafico_treinamento_v2.png')
    print("\nGráfico do histórico salvo como 'grafico_treinamento_v2.png'")

def main():
    # 1. Carregar os Dados
    X, y = load_data()
    
    if X is None or X.shape[0] == 0:
        print("Nenhum dado foi carregado. Saindo.")
        return

    # Nosso X agora tem 135 features (pose + 3 extras)
    print(f"\nForma final dos dados (Amostras, Frames, Features): {X.shape}") 
    print(f"Forma final dos rótulos (Amostras,): {y.shape}")

    # 2. Calcular Pesos de Classe (Class Weights) ### MUDANÇA IMPORTANTE ###
    # Isso corrige o desbalanceamento (12.900 vs 6.650)
    # 'balanced' faz o sklearn calcular os pesos ideais para nós.
    weights = class_weight.compute_class_weight('balanced',
                                                classes=np.unique(y),
                                                y=y)
    # Converte para um dicionário que o Keras entende: {0: peso_0, 1: peso_1}
    class_weights = dict(zip(np.unique(y), weights))
    
    print("\nDataset desbalanceado. Aplicando pesos de classe:")
    print(f"Peso para Classe 0 (nao_quedas): {class_weights[0]:.2f}")
    print(f"Peso para Classe 1 (quedas): {class_weights[1]:.2f}")

    # 3. Dividir em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"\nAmostras de treino: {X_train.shape[0]}")
    print(f"Amostras de teste: {X_test.shape[0]}")

    # 4. Construir o Modelo (com input_shape=135)
    
    # Forma da entrada agora é (45, 135)
    input_shape = (X.shape[1], X.shape[2]) # (45, 135)
    
    model = Sequential([
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
    
    model.summary() 

    # 5. Treinar o Modelo (com class_weight)
    print("\n--- Iniciando o Treinamento (v2) ---")
    
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=80, # 30 épocas (aumentamos de 20)
        validation_data=(X_test, y_test),
        class_weight=class_weights  # <<<--- A MÁGICA ACONTECE AQUI
    )
    print("--- Treinamento Concluído ---")

    # 6. Avaliar
    print("\n--- Avaliação do Modelo (v2) ---")
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=LABELS))
    
    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    
    # 7. Salvar
    plot_history(history)
    model.save(MODEL_NAME)
    print(f"\nModelo salvo com sucesso como '{MODEL_NAME}'!")


if __name__ == "__main__":
    main()
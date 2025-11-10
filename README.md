# Safe-Home-ai-training

Este é um projeto de Inteligência Artificial para detecção de quedas em tempo real usando Python, MediaPipe e TensorFlow/Keras.

## 🚀 Como Executar o Projeto

Siga estes passos para configurar seu ambiente local e rodar a aplicação.

### 1. Pré-requisitos

* **Python 3.10**
    * (O projeto foi desenvolvido e testado com Python 3.10.x. Versões diferentes, como 3.12+, podem causar conflitos de biblioteca.)
* [Git](https://git-scm.com/downloads)

### 2. Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/marcosVictorM/Safe-Home-ai-training.git](https://github.com/marcosVictorM/Safe-Home-ai-training.git)
    cd Safe-Home-ai-training
    ```

2.  **Crie e ative um Ambiente Virtual:**
    ```bash
    # Crie o ambiente (usando Python 3.10)
    python -m venv venv

    # Ative o ambiente
    # No Windows (PowerShell/CMD):
    .\venv\Scripts\activate
    # No Linux/Mac:
    # source venv/bin/activate
    ```

3.  **Instale todas as dependências:**
    (Este comando lê o arquivo `requirements.txt` e instala tudo automaticamente)
    ```bash
    pip install -r requirements.txt
    ```

### 3. Execução

Após a instalação, você pode rodar os scripts principais:

* **Para treinar a IA (se você tiver os dados):**
    ```bash
    python ai_train.py
    ```
* **Para rodar o detector final com a webcam:**
    ```bash
    python AI_final_detector.py
    ```
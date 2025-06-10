# -*- coding: utf-8 -*-
"""
Script para treinar e avaliar um modelo de classificação de imagens (CNN)
para distinguir entre cachorros, gatos e pandas.
"""

# 1. Imports
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from typing import List, Tuple
from numpy.typing import NDArray

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras.utils import to_categorical

# 2. Constantes e Configurações
# Usar letras maiúsculas para variáveis que não devem ser alteradas durante a execução.
DATA_DIR = Path('./animals/')
CATEGORIES = ['cachorro', 'panda', 'gato']
NUM_CLASSES = len(CATEGORIES)

# Configurações de pré-processamento da imagem
IMG_HEIGHT = 50
IMG_WIDTH = 50
N_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)

# Configurações de Treinamento
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 20

# 3. Funções Auxiliares

def setup_gpu():
    """Verifica a disponibilidade da GPU e imprime o status."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Sucesso! {len(gpus)} GPU(s) encontradas:")
        for i, gpu in enumerate(gpus):
            print(f"  - GPU {i}: {gpu.name}")
    else:
        print("⚠️ Aviso: Nenhuma GPU encontrada. O TensorFlow usará a CPU.")
        print("   Para treinos de imagem, o uso da CPU pode ser muito lento.")

def load_and_preprocess_data(data_dir: Path, categories: List[str]) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Carrega as imagens, pré-processa e as divide em conjuntos de treino e teste.

    Args:
        data_dir (Path): O diretório raiz contendo as subpastas das categorias.
        categories (List[str]): Uma lista com os nomes das categorias.

    Returns:
        Tuple: Contendo (trainX, trainY, testX, testY).
    """
    print("\nIniciando carregamento e pré-processamento dos dados...")
    
    image_paths = []
    labels = []

    for i, category in enumerate(categories):
        category_path = data_dir / category
        for f in category_path.iterdir():
            image_paths.append(str(f))
            labels.append(i)

    # Combinar e embaralhar para garantir a distribuição aleatória
    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    data = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        data.append(image_resized)

    # Normalizar pixels para o intervalo [0, 1] e converter para NumPy arrays
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)

    # Dividir os dados em treino e teste
    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )

    # Converter labels de treino para o formato one-hot encoding
    trainY_categorical = to_categorical(trainY, NUM_CLASSES)

    print("✅ Dados carregados e pré-processados com sucesso.")
    print(f"   - Formato do lote de treino (X): {trainX.shape}")
    print(f"   - Formato do lote de treino (Y): {trainY_categorical.shape}")
    print(f"   - Formato do lote de teste (X):  {testX.shape}")
    print(f"   - Formato do lote de teste (Y):  {testY.shape}")
    
    return trainX, trainY_categorical, testX, testY

def build_model(input_shape: Tuple[int, int, int], num_classes: int) -> Sequential:
    """
    Constrói e compila o modelo de Rede Neural Convolucional (CNN).

    Args:
        input_shape (Tuple): A forma dos dados de entrada (altura, largura, canais).
        num_classes (int): O número de classes de saída.

    Returns:
        Sequential: O modelo Keras compilado.
    """
    print("\nConstruindo o modelo CNN...")
    
    model = Sequential([
        Input(shape=input_shape),

        Conv2D(256, (3, 3)),
        BatchNormalization(), # <--- ADICIONAR
        Activation('relu'),   # <--- MUDAR
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(256, (3, 3)),
        BatchNormalization(), # <--- ADICIONAR
        Activation('relu'),   # <--- MUDAR
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        
        Dense(512),
        BatchNormalization(), # <--- ADICIONAR
        Activation('relu'),   # <--- MUDAR
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print("✅ Modelo construído e compilado.")
    model.summary()
    
    return model

def train_model(model: Sequential, trainX: NDArray, trainY: NDArray) -> Sequential:
    """
    Treina o modelo com os dados fornecidos.

    Args:
        model (Sequential): O modelo Keras a ser treinado.
        trainX (np_ndarray): Os dados de imagem de treino.
        trainY (np_ndarray): Os rótulos de treino.

    Returns:
        Sequential: O modelo treinado.
    """
    print("\nIniciando o treinamento do modelo...")
    
    model.fit(trainX, trainY, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    
    print("✅ Treinamento concluído.")
    return model

def evaluate_model(model: Sequential, testX: NDArray, testY: NDArray, categories: List[str]):
    """
    Avalia o desempenho do modelo no conjunto de teste e plota a matriz de confusão.

    Args:
        model (Sequential): O modelo treinado.
        testX (np_ndarray): As imagens de teste.
        testY (np_ndarray): Os rótulos verdadeiros de teste.
        categories (List[str]): Nomes das classes para os eixos do gráfico.
    """
    print("\nIniciando avaliação do modelo...")
    
    # Fazer predições e converter de volta para rótulos de classe (de one-hot para inteiro)
    pred_probabilities = model.predict(testX)
    predictions = np.argmax(pred_probabilities, axis=1)

    # Calcular e exibir a acurácia
    accuracy = accuracy_score(testY, predictions)
    print(f"Acurácia no conjunto de teste: {accuracy:.2%}")

    # Gerar e plotar a matriz de confusão
    cm = confusion_matrix(testY, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro (True)')
    plt.xlabel('Predito (Predicted)')
    plt.show()

def classify_image(model: Sequential, image_path: str, categories: List[str]):
    """
    Carrega, pré-processa e classifica uma única imagem.

    Args:
        model (Sequential): O modelo treinado.
        image_path (str): O caminho para a imagem a ser classificada.
        categories (List[str]): A lista de nomes de categorias.
    """
    print(f"\nClassificando a imagem: {image_path}")
    
    # Pré-processamento da imagem
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image_normalized = np.array(image_resized, dtype="float32") / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0) # Adiciona dimensão de lote

    # Predição
    score = model.predict(image_batch, verbose=0)
    label_index = np.argmax(score)
    confidence = np.max(score)
    animal_name = categories[label_index]

    print(f"   - Animal previsto: '{animal_name.capitalize()}'")
    print(f"   - Confiança: {confidence:.2%}")


# 4. Bloco de Execução Principal
def main():
    """Função principal que orquestra todo o fluxo do script."""
    setup_gpu()

    # Verificar se o diretório de dados existe
    if not DATA_DIR.exists():
        print(f"❌ Erro: O diretório de dados '{DATA_DIR}' não foi encontrado.")
        return

    # Etapas do pipeline de Machine Learning
    trainX, trainY, testX, testY = load_and_preprocess_data(DATA_DIR, CATEGORIES)
    model = build_model(INPUT_SHAPE, NUM_CLASSES)
    model = train_model(model, trainX, trainY)
    evaluate_model(model, testX, testY, CATEGORIES)

    # Exemplo de classificação de uma imagem de teste
    test_image_path = str(DATA_DIR / 'gato' / 'cats_00009.jpg')
    if Path(test_image_path).exists():
        classify_image(model, test_image_path, CATEGORIES)
    else:
        print(f"⚠️ Aviso: Imagem de teste '{test_image_path}' não encontrada.")


if __name__ == "__main__":
    main()
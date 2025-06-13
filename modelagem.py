# -*- coding: utf-8 -*-
"""
Script para treinar e avaliar um modelo de classificação de imagens (ResNet-50)
para distinguir entre cachorros, gatos e pandas, usando Transfer Learning.
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

# --- Linhas Alteradas/Novas ---
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50 # <--- NOVO: Importa o ResNet-50
from tensorflow.keras.applications.resnet50 import preprocess_input # <--- NOVO: Função de pré-processamento do ResNet

# 2. Constantes e Configurações
DATA_DIR = Path('./animals/')
CATEGORIES = ['cachorro', 'panda', 'gato']
NUM_CLASSES = len(CATEGORIES)

# --- ALTERAÇÃO: Ajustar o tamanho da imagem para o padrão do ResNet-50 ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
N_CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, N_CHANNELS)

# Configurações de Treinamento
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 128 # Com ResNet-50 e imagens maiores, talvez precise diminuir o batch size se tiver pouca VRAM se a gpu estiver com folga aumente esse valor - 256/12gb, 128/6gb, 64/3gb, ...
EPOCHS = 10 # Transfer learning costuma ser mais rápido, 10 épocas pode ser um bom começo

# 3. Funções Auxiliares

def setup_gpu():
    """Verifica a disponibilidade da GPU e imprime o status."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Sucesso! {len(gpus)} GPU(s) encontradas:")
        for i, gpu in enumerate(gpus):
            print(f"   - GPU {i}: {gpu.name}")
    else:
        print("⚠️ Aviso: Nenhuma GPU encontrada. O TensorFlow usará a CPU.")
        print("   Para treinos de imagem, o uso da CPU pode ser muito lento.")

# --- ALTERAÇÃO: Usar o pré-processamento específico do ResNet-50 ---
def load_and_preprocess_data(data_dir: Path, categories: List[str]) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Carrega as imagens, redimensiona para 224x224 e pré-processa para o ResNet-50.
    """
    print("\nIniciando carregamento e pré-processamento dos dados...")
    
    image_paths = []
    labels = []

    for i, category in enumerate(categories):
        category_path = data_dir / category
        for f in category_path.iterdir():
            image_paths.append(str(f))
            labels.append(i)

    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    data = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        data.append(image_resized)

    # Converter para array e aplicar o pré-processamento do ResNet
    data = np.array(data, dtype="float32")
    data = preprocess_input(data) # <--- ALTERAÇÃO IMPORTANTE

    labels = np.array(labels)

    (trainX, testX, trainY, testY) = train_test_split(
        data, labels, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE
    )

    trainY_categorical = to_categorical(trainY, NUM_CLASSES)

    print("✅ Dados carregados e pré-processados com sucesso.")
    print(f"   - Formato do lote de treino (X): {trainX.shape}")
    print(f"   - Formato do lote de treino (Y): {trainY_categorical.shape}")
    print(f"   - Formato do lote de teste (X):  {testX.shape}")
    print(f"   - Formato do lote de teste (Y):  {testY.shape}")
    
    return trainX, trainY_categorical, testX, testY

# --- ALTERAÇÃO: Nova função para construir o modelo com ResNet-50 ---
def build_resnet_model(input_shape: Tuple[int, int, int], num_classes: int) -> Model:
    """
    Constrói um modelo usando a base do ResNet-50 com Transfer Learning.
    """
    print("\nConstruindo o modelo com base ResNet-50...")
    
    # 1. Carregar o modelo base (ResNet-50) sem a camada de classificação final ('include_top=False')
    # Usaremos os pesos pré-treinados na base de dados ImageNet
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # 2. Congelar as camadas do modelo base. Não queremos treiná-las novamente.
    base_model.trainable = False

    # 3. Adicionar nossas próprias camadas de classificação no topo
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Converte as features em um vetor
    x = Dense(1024, activation='relu')(x) # Uma camada densa para aprender combinações
    x = Dropout(0.5)(x) # Dropout para regularização
    predictions = Dense(num_classes, activation='softmax')(x) # A camada de saída final

    # 4. Criar o modelo final
    model = Model(inputs=base_model.input, outputs=predictions)

    # 5. Compilar o modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("✅ Modelo construído e compilado.")
    model.summary()
    
    return model

# As outras funções (train_model_with_augmentation, evaluate_model, classify_image) podem ser mantidas como estão!

def train_model_with_augmentation(model: Model, trainX: NDArray, trainY: NDArray, testX: NDArray, testY: NDArray) -> Model:
    """Treina o modelo usando data augmentation em tempo real."""
    print("\nIniciando o treinamento do modelo com Data Augmentation...")
    imgAug = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    train_generator = imgAug.flow(trainX, trainY, batch_size=BATCH_SIZE)
    model.fit(train_generator, steps_per_epoch=len(trainX) // BATCH_SIZE, epochs=EPOCHS, validation_data=(testX, testY), verbose=1)
    print("✅ Treinamento concluído.")
    return model

def evaluate_model(model: Model, testX: NDArray, testY: NDArray, categories: List[str]):
    """Avalia o desempenho do modelo no conjunto de teste e plota a matriz de confusão."""
    print("\nIniciando avaliação do modelo...")
    pred_probabilities = model.predict(testX)
    predictions = np.argmax(pred_probabilities, axis=1)
    accuracy = accuracy_score(testY, predictions)
    print(f"Acurácia no conjunto de teste: {accuracy:.2%}")
    cm = confusion_matrix(testY, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.title('Matriz de Confusão')
    plt.ylabel('Verdadeiro (True)')
    plt.xlabel('Predito (Predicted)')
    plt.show()

def classify_image(model: Model, image_path: str, categories: List[str]):
    """Carrega, pré-processa e classifica uma única imagem."""
    print(f"\nClassificando a imagem: {image_path}")
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image_for_processing = np.expand_dims(image_resized, axis=0)
    image_preprocessed = preprocess_input(np.array(image_for_processing, dtype="float32"))
    score = model.predict(image_preprocessed, verbose=0)
    label_index = np.argmax(score)
    confidence = np.max(score)
    animal_name = categories[label_index]
    print(f"   - Animal previsto: '{animal_name.capitalize()}'")
    print(f"   - Confiança: {confidence:.2%}")

# 4. Bloco de Execução Principal
def main():
    """Função principal que orquestra todo o fluxo do script."""
    setup_gpu()

    if not DATA_DIR.exists():
        print(f"❌ Erro: O diretório de dados '{DATA_DIR}' não foi encontrado.")
        return

    trainX, trainY_cat, testX, testY = load_and_preprocess_data(DATA_DIR, CATEGORIES)
    
    # --- ALTERAÇÃO: Chamar a nova função de construção do modelo ---
    model = build_resnet_model(INPUT_SHAPE, NUM_CLASSES)
    
    testY_cat = to_categorical(testY, NUM_CLASSES)
    model = train_model_with_augmentation(model, trainX, trainY_cat, testX, testY_cat)
    
    evaluate_model(model, testX, testY, CATEGORIES)

    test_image_path = str(DATA_DIR / 'gato' / 'cats_00009.jpg')
    if Path(test_image_path).exists():
        classify_image(model, test_image_path, CATEGORIES)
    else:
        print(f"⚠️ Aviso: Imagem de teste '{test_image_path}' não encontrada.")

if __name__ == "__main__":
    main()
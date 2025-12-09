import cv2
import os
import numpy as np

# --- CONFIGURAÇÃO ---
DATASET_DIR = "dataset"
# Tamanho padrão para redimensionar todas as imagens. Essencial para evitar o erro.
STANDARD_SIZE = (100, 100) 
# --------------------

model = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_label = 0

print("Iniciando carregamento e pré-processamento de imagens...")

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)

    # Ignora arquivos que não são diretórios (pastas de pessoas)
    if not os.path.isdir(person_path):
        continue

    # Mapeia o nome da pessoa ao ID numérico
    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        # Ignora arquivos ocultos ou que não são imagens JPG/PNG
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(person_path, img_name)
        # Carrega a imagem em escala de cinza
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"ATENÇÃO: Pulando arquivo inválido/corrompido: {img_path}")
            continue

        try:

            img_resized = cv2.resize(img, STANDARD_SIZE)
        except cv2.error as e:
            # Captura erros de redimensionamento (se a imagem tiver dimensões inválidas)
            print(f"ERRO DE DIMENSÃO: Pulando {img_path} devido a: {e}")
            continue

        faces.append(img_resized)
        labels.append(current_label)

    current_label += 1

# Garante que temos imagens para treinar
if not faces:
    print("ERRO: Nenhuma imagem válida encontrada no diretório 'dataset'. O treinamento foi abortado.")
    exit()


# Esta linha agora deve funcionar, pois todos os elementos de 'faces' têm o mesmo tamanho
print(f"Total de {len(faces)} imagens carregadas e redimensionadas.")
faces = np.array(faces)
labels = np.array(labels)

print("\nTreinando, aguenta aí...")

# Treina o modelo LBPH
model.train(faces, labels)

# Salva o modelo treinado
model.save("model.yml")

# Salva o mapa de rótulos (nomes)
with open("labels.txt", "w") as f:
    f.write(str(label_map))

print(" Treinamento concluído!")
print(f"Modelo salvo em 'model.yml' e rótulos em 'labels.txt'.")
print("Pessoas reconhecidas:", label_map)

import cv2
import os
import time
import numpy as np

# =============================
# 1. CAPTURA DE IMAGENS
# =============================

person_name = input("Digite o nome da pessoa: ").strip()

SAVE_FOLDER = os.path.join("dataset", person_name)

# Se a pasta existir, não cria outra
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)
    print(f"Pasta criada: {SAVE_FOLDER}")
else:
    print(f"Pasta já existe: {SAVE_FOLDER}\nAs novas imagens serão guardadas nela.")

# Contar quantas imagens já existem na pasta
existing_images = len(os.listdir(SAVE_FOLDER))
start_index = existing_images + 1  # evita sobrescrever

cap = cv2.VideoCapture(0)

total_images = 200
delay = 0.05

print(f"Iniciando captura para {person_name}...")

for i in range(start_index, start_index + total_images):
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem.")
        continue

    filename = f"{person_name}_{i}.jpg"
    path = os.path.join(SAVE_FOLDER, filename)

    cv2.imwrite(path, frame)
    print(f"Imagem {i - start_index + 1}/{total_images} salva.")

    time.sleep(delay)
    cv2.imshow("Capturando imagens...", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✔ Captura concluída!")

# =============================
# 2. TREINAMENTO AUTOMÁTICO
# =============================

print("\n🔧 Iniciando treinamento...")

DATASET_DIR = "dataset"
model = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_label = 0

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

model.train(faces, labels)
model.save("model.yml")

with open("labels.txt", "w") as f:
    f.write(str(label_map))

print("✔ Treinamento concluído!")
print("Pessoas reconhecidas:", label_map)

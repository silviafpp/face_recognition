import cv2
import os
import numpy as np

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

        faces.append(img)
        labels.append(current_label)

    current_label += 1

faces = np.array(faces)
labels = np.array(labels)

print("Treinando, aguenta aí...")

model.train(faces, labels)
model.save("model.yml")

with open("labels.txt", "w") as f:
    f.write(str(label_map))

print("✔ Treinamento concluído!")
print("Pessoas reconhecidas:", label_map)

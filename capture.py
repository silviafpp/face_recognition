import cv2
import os
import time

# Pergunta o nome ao utilizador
person_name = input("Digite o nome da pessoa: ").strip()

# Cria pasta dentro de dataset
SAVE_FOLDER = os.path.join("dataset", person_name)
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Inicializa webcam
cap = cv2.VideoCapture(0)

total_images = 200  # pode alterar
delay = 0.05        # pequeno atraso

print(f"Iniciando captura para {person_name}...")

for i in range(1, total_images + 1):
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem.")
        continue

    # Nome da imagem e caminho de destino
    filename = f"{person_name}_{i}.jpg"
    path = os.path.join(SAVE_FOLDER, filename)

    # Salva a foto
    cv2.imwrite(path, frame)
    print(f"Imagem {i}/{total_images} salva em {SAVE_FOLDER}")

    # Delay curto
    time.sleep(delay)

    # Previsualização em tempo real
    cv2.imshow("Capturando imagens...", frame)

    # Aperte 'q' para parar antes de terminar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Captura concluída.")
cap.release()
cv2.destroyAllWindows()
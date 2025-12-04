import cv2, os, numpy as np

# === CAPTURA ===
nome = input("Nome: ")
path = f"dataset/{nome}"
os.makedirs(path, exist_ok=True)

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
count = 0

while count < 30:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{path}/{count}.jpg", face)
        count += 1
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)

    cv2.imshow("Captura", frame)
    if cv2.waitKey(1)==ord('q'): break

cap.release()
cv2.destroyAllWindows()

print("✔ Captura concluída!")

# === TREINO ===
faces, labels = [], []
label_dict = {}  
current_id = 0

for pessoa in os.listdir("dataset"):
    pasta = f"dataset/{pessoa}"
    if not os.path.isdir(pasta): continue
    
    label_dict[current_id] = pessoa
    
    for img in os.listdir(pasta):
        img_path = f"{pasta}/{img}"
        face_img = cv2.imread(img_path, 0)
        faces.append(face_img)
        labels.append(current_id)
    
    current_id += 1

faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)
recognizer.save("modelo.yml")

print("✔ Treino concluído! Modelo salvo em modelo.yml")

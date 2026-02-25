import cv2
import numpy as np
import joblib
from deepface import DeepFace

MODEL_NAME = "ArcFace"


DETECTOR_BACKEND = "opencv"   
ENFORCE_DET = True           

clf = joblib.load("face_clf.pkl")
le = joblib.load("label_encoder.pkl")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def get_embedding(face_bgr):
    
    rep = DeepFace.represent(
        img_path=face_bgr,
        model_name=MODEL_NAME,
        detector_backend="skip",   
        enforce_detection=False
    )
    return np.array(rep[0]["embedding"], dtype=np.float32)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not found or cannot be opened.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    
    if len(faces) > 0:
        faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
        x, y, w, h = faces[0]

        
        pad = int(0.25 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face = frame[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))

        try:
            emb = get_embedding(face)
            pred_id = clf.predict([emb])[0]
            name = le.inverse_transform([pred_id])[0]

            
            probs = clf.predict_proba([emb])[0]
            conf = float(np.max(probs))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        except Exception as e:
            cv2.putText(frame, "Face detected, embedding error", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
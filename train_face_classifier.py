import os
import numpy as np
from deepface import DeepFace
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_ROOT = "Face_dataset_cropped"  
MODEL_NAME = "ArcFace"              
DETECTOR = "skip"                   

def load_embeddings(split):
    X, y = [], []
    split_dir = os.path.join(DATA_ROOT, split)

    for person in sorted(os.listdir(split_dir)):
        person_dir = os.path.join(split_dir, person)
        if not os.path.isdir(person_dir):
            continue

        for fn in os.listdir(person_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(person_dir, fn)

            try:
                emb = DeepFace.represent(
                    img_path=path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR,
                    enforce_detection=False
                )
                
                vec = np.array(emb[0]["embedding"], dtype=np.float32)
                X.append(vec)
                y.append(person)
            except Exception as e:
                print("Error on:", path, "->", e)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X_train, y_train = load_embeddings("Train")
    X_val, y_val = load_embeddings("Validation")

    print("Train samples:", len(y_train), "Val samples:", len(y_val))

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)

    clf = LogisticRegression(max_iter=3000)
    clf.fit(X_train, y_train_enc)

    pred = clf.predict(X_val)
    acc = accuracy_score(y_val_enc, pred)
    print("Validation accuracy:", acc)

    print("\nReport:\n", classification_report(y_val_enc, pred, target_names=le.classes_))

    joblib.dump(clf, "face_clf.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("\nSaved: face_clf.pkl, label_encoder.pkl")
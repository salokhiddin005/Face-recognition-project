import os
import cv2
from mtcnn import MTCNN

INPUT_ROOT = "Face_dataset"
OUTPUT_ROOT = "Face_dataset_cropped"

detector = MTCNN()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def crop_one_image(img_path, out_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return False

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    if len(faces) == 0:
        return False

    
    faces = sorted(faces, key=lambda f: f["box"][2] * f["box"][3], reverse=True)
    x, y, w, h = faces[0]["box"]

    
    pad = int(0.25 * max(w, h))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_bgr.shape[1], x + w + pad)
    y2 = min(img_bgr.shape[0], y + h + pad)

    face = img_bgr[y1:y2, x1:x2]
    face = cv2.resize(face, (160, 160))  
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, face)
    return True

def process_split(split_name):
    split_in = os.path.join(INPUT_ROOT, split_name)
    split_out = os.path.join(OUTPUT_ROOT, split_name)

    for person in os.listdir(split_in):
        person_in = os.path.join(split_in, person)
        if not os.path.isdir(person_in):
            continue

        person_out = os.path.join(split_out, person)
        ensure_dir(person_out)

        for fn in os.listdir(person_in):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            inp = os.path.join(person_in, fn)
            outp = os.path.join(person_out, fn)
            ok = crop_one_image(inp, outp)
            if not ok:
                print(f"[NO FACE] {inp}")

if __name__ == "__main__":
    process_split("Train")
    process_split("Validation")
    print("Done. Cropped dataset saved to:", OUTPUT_ROOT)
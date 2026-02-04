import os
import json
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Where models are expected (top-level of project)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

EFF_MODEL = os.path.join(MODELS_DIR, "efficientnet_b0_model.h5")
RES_MODEL = os.path.join(MODELS_DIR, "resnet50_model.h5")

_models = []
for p in (EFF_MODEL, RES_MODEL):
    if os.path.exists(p):
        try:
            _models.append(load_model(p))
            print("Loaded model:", os.path.basename(p))
        except Exception as e:
            print("Failed to load", p, ":", e)

# class mapping (optional)
CLASS_MAP = {}
class_map_file = os.path.join(MODELS_DIR, "class_indices.json")
if os.path.exists(class_map_file):
    try:
        with open(class_map_file, "r") as f:
            CLASS_MAP = json.load(f)
            # ensure values are ints
            CLASS_MAP = {k: int(v) for k, v in CLASS_MAP.items()}
    except Exception as e:
        print("Failed to load class_indices.json:", e)

# --- Prediction function ---
def predict_image(img_path):
    """
    Predict using ensemble of loaded models.
    Returns (label, confidence) where label is class name if class indices provided else index str.
    """
    if not _models:
        raise RuntimeError("No models loaded in 'models/' directory. Add your .h5 files.")

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)

    preds = []
    for m in _models:
        try:
            p = m.predict(x)
            preds.append(p[0])
        except Exception as e:
            print("Model prediction error:", e)
    if not preds:
        raise RuntimeError("No predictions produced by models.")

    avg = np.mean(preds, axis=0)
    idx = int(np.argmax(avg))
    confidence = float(np.max(avg))

    # map to class name if CLASS_MAP exists
    if CLASS_MAP:
        inv = {v: k for k, v in CLASS_MAP.items()}
        label = inv.get(idx, str(idx))
    else:
        label = str(idx)
    return label, confidence

# --- Optional preprocessing helper (segmentation) ---
def preprocess_image_if_available(img_path):
    """
    Try to remove background if rembg installed, else do a simple contour crop via OpenCV.
    This modifies the image file in-place (overwrites).
    """
    try:
        # try rembg first
        from rembg import remove
        with open(img_path, "rb") as i:
            input_bytes = i.read()
        output = remove(input_bytes)
        with open(img_path, "wb") as o:
            o.write(output)
        return True
    except Exception:
        # fallback: OpenCV contour crop
        try:
            import cv2
            import numpy as np
            img = cv2.imread(img_path)
            if img is None:
                return False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return False
            c = max(contours, key=cv2.contourArea)
            x,y,w,h = cv2.boundingRect(c)
            crop = img[y:y+h, x:x+w]
            crop = cv2.resize(crop, (224,224))
            cv2.imwrite(img_path, crop)
            return True
        except Exception:
            return False

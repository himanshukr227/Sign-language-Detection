# improved_sign_language_live.py
import os
import cv2
import numpy as np

# Use tensorflow.keras for compatibility
from tensorflow.keras.models import model_from_json

# --- load model ---
MODEL_JSON = "signlanguagedetectionmodel48x48.json"
MODEL_WEIGHTS = "signlanguagedetectionmodel48x48.h5"

if not os.path.exists(MODEL_JSON) or not os.path.exists(MODEL_WEIGHTS):
    raise FileNotFoundError(f"Model files not found. Expected {MODEL_JSON} and {MODEL_WEIGHTS} in cwd: {os.getcwd()}")

with open(MODEL_JSON, "r") as f:
    model_json = f.read()

model = model_from_json(model_json)
model.load_weights(MODEL_WEIGHTS)
print("Model loaded.")

# --- helper ---
def extract_features(image):
    """
    image: single-channel 48x48 numpy array (uint8 or float)
    returns normalized array shaped (1,48,48,1), dtype=float32
    """
    arr = np.array(image, dtype=np.float32)
    arr = arr.reshape(1, 48, 48, 1)
    return arr / 255.0

# label list: make sure this order matches your model's training label order
label = ['A', 'M', 'N', 'S', 'T', 'blank']

# --- camera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    # try different indices if your camera is not at 0
    cap.release()
    raise SystemExit("ERROR: Cannot open camera. Try changing the index (0,1,2) or check camera permissions.")

print("Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read frame from camera.")
        break

    # draw ROI rectangle
    cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
    roi = frame[40:300, 0:300]  # note: ensure your frame is at least this large

    # if ROI size is not expected (camera smaller), skip
    if roi.size == 0 or roi.shape[0] < 48 or roi.shape[1] < 48:
        cv2.putText(frame, "ROI too small", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))

    # prepare for model
    x = extract_features(resized)  # shape (1,48,48,1), dtype float32

    # predict
    pred = model.predict(x, verbose=0)   # shape (1, num_classes)
    prob = float(np.max(pred))           # highest probability
    idx = int(np.argmax(pred, axis=1)[0])
    prediction_label = label[idx] if idx < len(label) else "?"

    # draw top bar background
    cv2.rectangle(frame, (0, 0), (400, 40), (0, 165, 255), -1)

    if prediction_label == 'blank':
        text = ""  # hide blank
    else:
        accu = "{:.2f}".format(prob * 100)
        text = f'{prediction_label}  {accu}%'

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # show windows
    cv2.imshow("output", frame)
    cv2.imshow("ROI", resized)  # show the grayscale 48x48 for debugging

    # key handling: press 'q' to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
print("Exited cleanly.")

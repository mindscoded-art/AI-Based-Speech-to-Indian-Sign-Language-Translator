import os
import cv2
import numpy as np
import tensorflow as tf

# -------- ABSOLUTE MODEL PATH (SAFE) --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_cnn_model.h5")

print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
# -------------------------------------------

# Labels MUST match your dataset folder names (order matters)
import json

# Load class indices
with open(os.path.join(BASE_DIR, "models", "class_indices.json"), "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index → label
labels = {v: k for k, v in class_indices.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.reshape(img, (1, 64, 64, 3))

    # Predict
    prediction = model.predict(img, verbose=0)
    gesture = labels[np.argmax(prediction)]

    # Display result
    cv2.putText(frame, f"Gesture: {gesture}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("ISL Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
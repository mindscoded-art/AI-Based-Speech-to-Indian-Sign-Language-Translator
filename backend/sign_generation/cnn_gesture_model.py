import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "ISL_Hand_Gesture_Dataset", "train")
TEST_DIR  = os.path.join(BASE_DIR, "dataset", "ISL_Hand_Gesture_Dataset", "test")

print("Train path:", TRAIN_DIR)
print("Test path:", TEST_DIR)

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(64, 64),
    class_mode="categorical"
)

test_data = datagen.flow_from_directory(
    TEST_DIR,
    target_size=(64, 64),
    class_mode="categorical"
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(train_data.num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, epochs=15, validation_data=test_data)

model.save("models/isl_cnn_model.h5")

print("CNN Model trained & saved")

# Save class indices for prediction
import json

class_indices = train_data.class_indices
with open("models/class_indices.json", "w") as f:
    json.dump(class_indices, f)

print("Class indices saved:", class_indices)
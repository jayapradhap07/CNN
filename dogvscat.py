
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras import layers, models

# 1. Configuration
DATA_DIR = Path("dataset/cats_and_dogs_filtered")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

tf.random.set_seed(SEED)
np.random.seed(SEED)

# 2. Check dataset
if not TRAIN_DIR.is_dir() or not VAL_DIR.is_dir():
    raise FileNotFoundError(
        "Dataset not found. Check dataset/cats_and_dogs_filtered/"
    )

for folder in [TRAIN_DIR / "cats", TRAIN_DIR / "dogs",
               VAL_DIR / "cats", VAL_DIR / "dogs"]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing folder: {folder}")

# 3. Load training and validation images
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["cats", "dogs"],
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    class_names=["cats", "dogs"],
    shuffle=False
)

print("Classes: cats = 0, dogs = 1")

# 4. Normalize images and optimize loading
normalization = layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(
    lambda images, labels: (normalization(images), labels)
).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds.map(
    lambda images, labels: (normalization(images), labels)
).prefetch(tf.data.AUTOTUNE)

# 5. Build CNN model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 6. Train model
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# 7. Evaluate model
val_loss, val_accuracy = model.evaluate(val_ds, verbose=1)

print(f"\nValidation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# 8. Plot accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# 9. Classification report and confusion matrix
y_true = np.concatenate([
    labels.numpy().ravel() for _, labels in val_ds
]).astype(int)

y_prob = model.predict(val_ds, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=[0, 1],
    target_names=["Cat", "Dog"],
    zero_division=0
))

print("Accuracy:", round(accuracy_score(y_true, y_pred) * 100, 2), "%")

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["Cat", "Dog"])
plt.yticks([0, 1], ["Cat", "Dog"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.show()

# 10. Predict a new image
def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        print("Image not found:", image_path)
        return

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    array = tf.keras.utils.img_to_array(image) / 255.0
    probability = float(
        model.predict(array[np.newaxis, ...], verbose=0)[0][0]
    )

    predicted = "Dog" if probability >= 0.5 else "Cat"
    confidence = probability if probability >= 0.5 else 1 - probability

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"{predicted}: {confidence * 100:.2f}% confidence")
    plt.show()

    print("Prediction:", predicted)
    print(f"Confidence: {confidence * 100:.2f}%")

# Example: uncomment and provide your image path
# predict_image("my_cat.jpg")

# 11. Save trained model
model.save("dog_cat_cnn.keras")
print("\nModel saved as dog_cat_cnn.keras")

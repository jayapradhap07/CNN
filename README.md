# Cats vs. Dogs Image Classification using CNN 🐱🐶

A deep learning project implementing a Convolutional Neural Network (CNN) in **TensorFlow / Keras** for binary image classification to distinguish between cats and dogs.

---

## 📌 Project Overview

This repository provides an end-to-end pipeline for:
- Loading and preprocessing image data using `tf.keras.utils.image_dataset_from_directory`.
- Rescaling pixel values and leveraging asynchronous prefetching with `tf.data.AUTOTUNE`.
- Building and training a deep Convolutional Neural Network (CNN).
- Preventing overfitting with Dropout regularization and Early Stopping.
- Evaluating model performance with accuracy, loss plots, classification reports, and a confusion matrix.
- Running inference on new custom images with confidence scores.
- Saving and exporting the trained model in Keras format (`.keras`).

---

## 🏗️ Model Architecture

The CNN model takes an RGB image of size `128 x 128 x 3` and processes it through feature extraction and classification layers:

| Layer (Type) | Output Shape | Details / Activation |
|---|---|---|
| **InputLayer** | `(None, 128, 128, 3)` | RGB Input Image |
| **Conv2D** | `(None, 126, 126, 32)` | 32 filters, 3x3 kernel, ReLU |
| **MaxPooling2D** | `(None, 63, 63, 32)` | 2x2 pooling window |
| **Conv2D** | `(None, 61, 61, 64)` | 64 filters, 3x3 kernel, ReLU |
| **MaxPooling2D** | `(None, 30, 30, 64)` | 2x2 pooling window |
| **Conv2D** | `(None, 28, 28, 128)` | 128 filters, 3x3 kernel, ReLU |
| **MaxPooling2D** | `(None, 14, 14, 128)` | 2x2 pooling window |
| **Flatten** | `(None, 25088)` | Flattens feature maps |
| **Dense** | `(None, 128)` | 128 units, ReLU |
| **Dropout** | `(None, 128)` | Dropout rate: 0.5 |
| **Dense (Output)** | `(None, 1)` | 1 unit, Sigmoid |

- **Loss Function:** `binary_crossentropy`
- **Optimizer:** `adam`
- **Metric:** `accuracy`
- **Callback:** `EarlyStopping` (monitors `val_loss`, patience = 3, restores best weights)

---

## 📁 Dataset Structure

The dataset is expected in the following directory layout under `dataset/cats_and_dogs_filtered`:

```text
dataset/
└── cats_and_dogs_filtered/
    ├── train/
    │   ├── cats/    # Training images of cats
    │   └── dogs/    # Training images of dogs
    └── validation/
        ├── cats/    # Validation images of cats
        └── dogs/    # Validation images of dogs
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/jayapradhap07/CNN.git
cd CNN
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Train the Model
Run the main script to train, evaluate, and save the model:
```bash
python dogvscat.py
```

### Inference on New Images
You can use the built-in `predict_image()` function in `dogvscat.py` to predict any custom image:

```python
from dogvscat import predict_image

# Predict an image
predict_image("path/to/your/image.jpg")
```

The script will display the image with the predicted label (`Cat` or `Dog`) and the prediction confidence percentage.

---

## 📊 Evaluation & Visualizations

During training and evaluation, the script outputs:
1. **Training & Validation Accuracy/Loss Curves:** Visualized across all epochs.
2. **Classification Report:** Detailed Precision, Recall, and F1-score for both classes.
3. **Confusion Matrix:** Heatmap showing True Positives, False Positives, True Negatives, and False Negatives.

---

## 💾 Saved Model

Once training finishes, the best model weights are automatically saved to:
```text
dog_cat_cnn.keras
```
You can reload it anytime via:
```python
import tensorflow as tf
model = tf.keras.models.load_model("dog_cat_cnn.keras")
```

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

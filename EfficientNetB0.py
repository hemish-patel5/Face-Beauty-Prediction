# ============================================================
# SCUT-FBP5500 Facial Beauty Prediction
# Optimised for Kaggle T4
#
# Changes:
# 1. EfficientNetB0
# 2. Mixed Precision
# 3. Huber Loss
# 4. Normalisation inside tf.data
# 5. Removed VGG16
# ============================================================

# ── CELL 1: Install + Imports ───────────────────────────────
!pip install -q mtcnn lz4

import os
import cv2
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from mtcnn import MTCNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Mixed precision (huge VRAM saver)
mixed_precision.set_global_policy('mixed_float16')

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
print("Mixed precision:", mixed_precision.global_policy())

# ── CELL 2: Config ──────────────────────────────────────────
class Config:
    DATA_DIR   = "/kaggle/input/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores"
    OUTPUT_DIR = "/kaggle/working"

    IMG_SIZE   = 224
    CHANNELS   = 3

    # Smaller batch for EfficientNet
    BATCH_SIZE = 16

    EPOCHS_HEAD = 10
    EPOCHS_FINE = 25

    LR_HEAD = 3e-4
    LR_FINE = 1e-5

    DROPOUT = 0.4
    DENSE   = 256

    VAL_RATIO  = 0.10
    TEST_RATIO = 0.10

    MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    STD  = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

cfg = Config()

# ── CELL 3: Load Dataset ────────────────────────────────────
data = np.load(
    f"{cfg.DATA_DIR}/scut_fbp5500-cmprsd.npz",
    allow_pickle=True
)

X = data['X'].astype(np.uint8)
y = data['y'].astype(np.float32)

print("Images:", X.shape)
print("Scores:", y.shape)

# ── CELL 4: Score Distribution ──────────────────────────────
plt.figure(figsize=(8,4))
sns.histplot(y, bins=30, kde=True)
plt.title("Beauty Score Distribution")
plt.show()

# ── CELL 5: Face Alignment ──────────────────────────────────
detector = MTCNN()

CACHE_ALIGNED = f"{cfg.OUTPUT_DIR}/X_aligned.npy"

def align_face(img, target_size=224):

    results = detector.detect_faces(img)

    if results:
        r = results[0]

        x, yb, w, h = r['box']
        kps = r['keypoints']

        left_eye = np.array(kps['left_eye'], dtype=np.float32)
        right_eye = np.array(kps['right_eye'], dtype=np.float32)

        dX = right_eye[0] - left_eye[0]
        dY = right_eye[1] - left_eye[1]

        angle = np.degrees(np.arctan2(dY, dX))

        center = (
            float((left_eye[0] + right_eye[0]) / 2),
            float((left_eye[1] + right_eye[1]) / 2)
        )

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        aligned = cv2.warpAffine(
            img,
            M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_LINEAR
        )

        margin = int(0.20 * max(w, h))

        x1 = max(0, x - margin)
        y1 = max(0, yb - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], yb + h + margin)

        face = aligned[y1:y2, x1:x2]

        if face.size == 0:
            face = img

    else:
        h0, w0 = img.shape[:2]
        s = min(h0, w0)

        y1 = (h0 - s) // 2
        x1 = (w0 - s) // 2

        face = img[y1:y1+s, x1:x1+s]

    return cv2.resize(face, (target_size, target_size))

if os.path.exists(CACHE_ALIGNED):

    print("Loading aligned faces...")
    X_aligned = np.load(CACHE_ALIGNED)

else:

    print("Running face alignment...")

    X_aligned = np.zeros(
        (len(X), cfg.IMG_SIZE, cfg.IMG_SIZE, 3),
        dtype=np.uint8
    )

    for i in range(len(X)):

        try:
            X_aligned[i] = align_face(X[i])

        except:
            X_aligned[i] = cv2.resize(
                X[i],
                (cfg.IMG_SIZE, cfg.IMG_SIZE)
            )

        if (i + 1) % 500 == 0:
            print(f"{i+1}/{len(X)}")

    np.save(CACHE_ALIGNED, X_aligned)

print("Aligned:", X_aligned.shape)

# ── CELL 6: Train/Val/Test Split ────────────────────────────
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_aligned,
    y,
    test_size=cfg.TEST_RATIO,
    random_state=SEED
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval,
    y_trainval,
    test_size=cfg.VAL_RATIO / (1 - cfg.TEST_RATIO),
    random_state=SEED
)

print("Train:", len(X_train))
print("Val:", len(X_val))
print("Test:", len(X_test))

# ── CELL 7: tf.data Pipeline ────────────────────────────────

# Normalisation now happens HERE
def preprocess(image, label):

    image = tf.cast(image, tf.float32) / 255.0

    image = (image - cfg.MEAN) / cfg.STD

    return image, label

rotation = layers.RandomRotation(0.03)
zoom     = layers.RandomZoom(0.10)

def augment(image, label):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, 0.15)

    image = tf.image.random_contrast(image, 0.85, 1.15)

    image = tf.image.random_saturation(image, 0.85, 1.15)

    image = rotation(tf.expand_dims(image, 0))[0]

    image = zoom(tf.expand_dims(image, 0))[0]

    return image, label

def make_dataset(Xd, yd, training=False):

    ds = tf.data.Dataset.from_tensor_slices((Xd, yd))

    if training:
        ds = ds.shuffle(1024, seed=SEED)

    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(cfg.BATCH_SIZE)

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

train_ds = make_dataset(X_train, y_train, training=True)
val_ds   = make_dataset(X_val, y_val)
test_ds  = make_dataset(X_test, y_test)

print("Datasets ready.")

# ── CELL 8: Build EfficientNetB0 Model ──────────────────────
def build_model():

    inp = tf.keras.Input(
        shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, 3)
    )

    base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_tensor=inp
    )

    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)

    x = layers.Dropout(cfg.DROPOUT)(x)

    x = layers.Dense(
        cfg.DENSE,
        activation='relu'
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.25)(x)

    # IMPORTANT:
    # output MUST be float32 with mixed precision
    out = layers.Dense(
        1,
        activation='linear',
        dtype='float32'
    )(x)

    model = models.Model(inp, out)

    return model, base

model, base = build_model()

model.summary()

# ── CELL 9: Compile + Callbacks ─────────────────────────────
def compile_model(model, lr):

    optimizer = Adam(
        learning_rate=lr,
        clipnorm=1.0
    )

    # Huber loss better for noisy labels
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae']
    )

def get_callbacks(name):

    return [

        callbacks.ModelCheckpoint(
            f"{cfg.OUTPUT_DIR}/{name}.keras",
            monitor='val_mae',
            save_best_only=True,
            mode='min',
            verbose=1
        ),

        callbacks.ReduceLROnPlateau(
            monitor='val_mae',
            factor=0.5,
            patience=3,
            mode='min',
            verbose=1
        ),

        callbacks.EarlyStopping(
            monitor='val_mae',
            patience=7,
            mode='min',
            restore_best_weights=True,
            verbose=1
        )
    ]

# ── CELL 10: Stage 1 Training ───────────────────────────────
compile_model(model, cfg.LR_HEAD)

print("\nStage 1: Head training")

hist1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg.EPOCHS_HEAD,
    callbacks=get_callbacks("efficientnet_s1"),
    verbose=1
)

# ── CELL 11: Fine-Tuning ────────────────────────────────────
print("\nStage 2: Fine-tuning")

# Unfreeze last 120 layers
for layer in base.layers[-120:]:
    layer.trainable = True

compile_model(model, cfg.LR_FINE)

hist2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg.EPOCHS_FINE,
    callbacks=get_callbacks("efficientnet_s2"),
    verbose=1
)

# ── CELL 12: Evaluation ─────────────────────────────────────
preds = model.predict(test_ds, verbose=0).flatten()

mae  = mean_absolute_error(y_test, preds)
rmse = np.sqrt(np.mean((y_test - preds) ** 2))

pcc, _  = pearsonr(y_test, preds)
srcc, _ = spearmanr(y_test, preds)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

print(f"PCC  : {pcc:.4f}")
print(f"SRCC : {srcc:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")

print("="*50)

# ── CELL 13: Scatter Plot ───────────────────────────────────
plt.figure(figsize=(6,6))

plt.scatter(
    y_test,
    preds,
    alpha=0.35,
    s=12
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    'r--'
)

plt.xlabel("Ground Truth")
plt.ylabel("Prediction")

plt.title(
    f"PCC={pcc:.3f} | MAE={mae:.3f}"
)

plt.tight_layout()

plt.show()

# ── CELL 14: Save Model ─────────────────────────────────────
model.save(
    f"{cfg.OUTPUT_DIR}/efficientnet_beauty_final.keras"
)

print("\nSaved model:")
print(f"{cfg.OUTPUT_DIR}/efficientnet_beauty_final.keras")

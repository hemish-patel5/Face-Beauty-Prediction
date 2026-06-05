# ── Imports ─────────────────────────────────────
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

# ── Config ──────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS_HEAD = 15
EPOCHS_FINE = 80
LR_HEAD     = 1e-3
LR_FINE     = 1e-5

# ── Load Dataset ────────────────────────────────
data = np.load("/kaggle/input/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores/scut_fbp5500-cmprsd.npz")

X = data['X']
y = data['y'].astype(np.float32)

print(X.shape, y.shape)

# ── Resize, store as uint8 ───────────────────────
X_resized = np.zeros((len(X), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

for i in range(len(X)):
    X_resized[i] = tf.image.resize(X[i], (IMG_SIZE, IMG_SIZE)).numpy().astype(np.uint8)

del data, X

# ── Split ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_resized, y, test_size=0.2, random_state=42
)

del X_resized

# ── Dataset factory ──────────────────────────────
def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    return img, label

def make_dataset(X, y, shuffle=False, augment_data=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(
        lambda img, label: (eff_preprocess(tf.cast(img, tf.float32)), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train, shuffle=True, augment_data=True)
test_ds  = make_dataset(X_test,  y_test)

# ── Build Model ──────────────────────────────────
base = EfficientNetB3(weights='imagenet', include_top=False,
                      input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

model = tf.keras.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1)
], name='efficientnetb3')

# ── Stage 1: Head only ───────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(LR_HEAD),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=['mae']
)

print(f"\n── Stage 1: Head only ({EPOCHS_HEAD} epochs) ──")
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae', patience=5,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae', factor=0.5,
            patience=3, verbose=1)
    ]
)

# ── Stage 2: Fine-tune top 2/3 of base ───────────
total = len(base.layers)
freeze_until = total // 3

for layer in base.layers:
    layer.trainable = True  # unfreeze everything

# cosine decay over the full fine-tuning run
steps_per_epoch = len(X_train) // BATCH_SIZE
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR_FINE,
    decay_steps=steps_per_epoch * EPOCHS_FINE
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule),
    loss=tf.keras.losses.Huber(delta=1.0),
    metrics=['mae']
)

print(f"\n── Stage 2: Fine-tuning top {total - freeze_until} layers ({EPOCHS_FINE} epochs) ──")
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_FINE,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_mae', patience=10,
            restore_best_weights=True, verbose=1),
    ]
    # NOTE: removed ReduceLROnPlateau — cosine decay handles LR scheduling
)

# ── Evaluate ─────────────────────────────────────
preds  = model.predict(test_ds).flatten()
mae    = mean_absolute_error(y_test, preds)
pcc, _ = pearsonr(y_test, preds)

print(f"\n── EfficientNetB3 (fine-tuned) ──")
print(f"MAE : {mae:.4f}")
print(f"PCC : {pcc:.4f}")
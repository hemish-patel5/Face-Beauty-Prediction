# ── CELL 1: Imports ──────────────────────────────────────────
!pip install -q lz4
!pip install -q mtcnn

import os
import random
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, VGG16
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
 
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
 
 # ── CELL 2: Configuration ────────────────────────────────────
class Config:
    DATA_DIR    = "/kaggle/input/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores"
    OUTPUT_DIR  = "/kaggle/working"
    IMG_SIZE    = 224
    CHANNELS    = 3
    BATCH_SIZE  = 32
    EPOCHS_HEAD = 20
    EPOCHS_FINE = 50
    LR_HEAD     = 3e-4
    LR_FINE     = 3e-6
    DROPOUT_1   = 0.5
    DROPOUT_2   = 0.3
    DENSE_UNITS = 256
    VAL_RATIO   = 0.10
    TEST_RATIO  = 0.10
    MEAN        = [0.485, 0.456, 0.406]
    STD         = [0.229, 0.224, 0.225]
 
cfg = Config()
print("Config ready.")
 # ── CELL 3: Load Data from .npz ──────────────────────────────
data = np.load(f"{cfg.DATA_DIR}/scut_fbp5500-cmprsd.npz")
 
X = data['X']                      # (5500, 350, 350, 3)  uint8
y = data['y'].astype(np.float32)   # (5500,)  beauty scores as float
 
print(f"Images : {X.shape}  dtype: {X.dtype}")
print(f"Scores : {y.shape}  dtype: {y.dtype}")
print(f"Score range : {y.min():.4f} – {y.max():.4f}")
print(f"Score mean  : {y.mean():.4f}")
print(f"Score std   : {y.std():.4f}")
 
# Plot score distribution
plt.figure(figsize=(8, 4))
sns.histplot(y, bins=30, kde=True, color='steelblue')
plt.title("Beauty Score Distribution – SCUT-FBP5500")
plt.xlabel("Beauty Score (1–5)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{cfg.OUTPUT_DIR}/score_distribution.png", dpi=150)
plt.show()
print("Plot saved.")
 # ── CELL 3B: Face Alignment ───────────────────────────────────

!pip install -q lz4
!pip install -q mtcnn

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mtcnn import MTCNN

# Reload data with lz4 support
data = np.load(f"{cfg.DATA_DIR}/scut_fbp5500-cmprsd.npz", allow_pickle=True)
X = data['X']
y = data['y'].astype(np.float32)
print(f"Reloaded: {X.shape}  scores: {y.shape}")

detector = MTCNN()
# ... rest of the cell stays exactly the same

def align_face(img, target_size=224):
    """
    Detect face, align by eye angle, crop with margin, resize.
    Falls back to centre crop if detection fails.
    """
    # MTCNN needs RGB uint8
    results = detector.detect_faces(img)

    if results:
        r = results[0]
        kps = r['keypoints']
        x, y, w, h = r['box']

        # Calculate eye angle for alignment
        left_eye  = np.array(kps['left_eye'],  dtype=np.float32)
        right_eye = np.array(kps['right_eye'], dtype=np.float32)
        dX = right_eye[0] - left_eye[0]
        dY = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dY, dX))

        # Rotate image to level the eyes
        eyes_center = (
            float((left_eye[0] + right_eye[0]) / 2),
            float((left_eye[1] + right_eye[1]) / 2))
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                 flags=cv2.INTER_LINEAR)

        # Crop with 20% margin
        margin = int(0.20 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img.shape[1], x + w + margin)
        y2 = min(img.shape[0], y + h + margin)
        face = aligned[y1:y2, x1:x2]

        # Fallback if crop is empty
        if face.size == 0:
            face = img

    else:
        # No face detected — use centre crop as fallback
        h, w = img.shape[:2]
        s  = min(h, w)
        y1 = (h - s) // 2
        x1 = (w - s) // 2
        face = img[y1:y1+s, x1:x1+s]

    return cv2.resize(face, (target_size, target_size))


# Run alignment on all 5500 images
# Takes ~20–30 minutes — saves to cache so only runs once
CACHE_ALIGNED = f"{cfg.OUTPUT_DIR}/X_aligned.npy"

if os.path.exists(CACHE_ALIGNED):
    print("Loading aligned faces from cache...")
    X_aligned = np.load(CACHE_ALIGNED)
    print(f"Loaded: {X_aligned.shape}")
else:
    print(f"Aligning {len(X)} faces with MTCNN...")
    X_aligned = np.zeros((len(X), 224, 224, 3), dtype=np.uint8)
    failed = 0

    for i in range(len(X)):
        try:
            X_aligned[i] = align_face(X[i])
        except Exception:
            # If anything goes wrong just resize without alignment
            X_aligned[i] = cv2.resize(X[i], (224, 224))
            failed += 1

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(X)}  (failed detections: {failed})")

    np.save(CACHE_ALIGNED, X_aligned)
    print(f"Done. Aligned: {len(X)-failed}  Failed: {failed}")
    print(f"Saved to cache → X_aligned.npy")

# Show a sample to verify alignment worked
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Sample aligned faces", fontsize=13)
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_aligned[i * 100])
    ax.set_title(f"Score: {y[i*100]:.2f}", fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig(f"{cfg.OUTPUT_DIR}/aligned_samples.png", dpi=150)
plt.show()

# ── CELL 4: Normalise aligned images ─────────────────────────
CACHE_X = f"{cfg.OUTPUT_DIR}/X_processed.npy"
CACHE_Y = f"{cfg.OUTPUT_DIR}/y_scores.npy"

if os.path.exists(CACHE_X):
    print("Loading from cache...")
    X_processed = np.load(CACHE_X)
    y_scores    = np.load(CACHE_Y)
    print(f"Loaded: {X_processed.shape}")
else:
    mean = np.array(cfg.MEAN, dtype=np.float32)
    std  = np.array(cfg.STD,  dtype=np.float32)

    # Use X_aligned instead of raw X
    print(f"Normalising {len(X_aligned)} aligned images...")
    X_processed = np.zeros(
        (len(X_aligned), cfg.IMG_SIZE, cfg.IMG_SIZE, 3), dtype=np.float32)

    for i in range(len(X_aligned)):
        img = X_aligned[i].astype(np.float32) / 255.0
        img = (img - mean) / std
        X_processed[i] = img
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(X_aligned)} done")

    y_scores = y.copy()
    np.save(CACHE_X, X_processed)
    np.save(CACHE_Y, y_scores)
    print(f"Done. Shape: {X_processed.shape}")

# ── CELL 5: Train / Val / Test Split ─────────────────────────
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_processed, y_scores,
    test_size=cfg.TEST_RATIO, random_state=42)
 
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=cfg.VAL_RATIO / (1 - cfg.TEST_RATIO),
    random_state=42)
 
print(f"Train : {len(X_train)}")
print(f"Val   : {len(X_val)}")
print(f"Test  : {len(X_test)}")
 
# ── CELL 6: Augmentation & Data Pipelines ────────────────────
rotation_layer = tf.keras.layers.RandomRotation(10/360)

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.20)
    image = tf.image.random_contrast(image, lower=0.80, upper=1.20)
    noise = tf.random.normal(shape=tf.shape(image), stddev=0.01)
    image = tf.clip_by_value(image + noise, -3.0, 3.0)
    image = rotation_layer(tf.expand_dims(image, 0))[0]
    return image, label

def make_dataset(X, y, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X.astype('float32'), y.astype('float32')))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train, augment_data=True,  shuffle=True)
val_ds   = make_dataset(X_val,   y_val,   augment_data=False, shuffle=False)
test_ds  = make_dataset(X_test,  y_test,  augment_data=False, shuffle=False)

print("Data pipelines ready.")

# ── CELL 7: Model Builder ────────────────────────────────────
def build_model(arch='resnet50', trainable_base=False):
    inp = tf.keras.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS))

    if arch == 'resnet50':
        base = ResNet50(weights='imagenet', include_top=False, input_tensor=inp)
    else:
        base = VGG16(weights='imagenet', include_top=False, input_tensor=inp)

    # Freeze base layers
    base.trainable = trainable_base

    # Always keep batch norm layers trainable
    # Frozen BN uses ImageNet stats which breaks regression on new data
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(cfg.DROPOUT_1)(x)
    x = layers.Dense(cfg.DENSE_UNITS, activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg.DROPOUT_2)(x)
    out = layers.Dense(
        1,
        activation='linear',
        dtype='float32',
        kernel_initializer='glorot_uniform',
        name='score')(x)

    return models.Model(inputs=inp, outputs=out), base

print("Model builder ready.")

# ── CELL 8: Training Helpers ─────────────────────────────────
def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae'])

def get_callbacks(name):
    return [
        callbacks.ModelCheckpoint(
            f"{cfg.OUTPUT_DIR}/{name}_best.keras",
            monitor='val_mae', save_best_only=True, mode='min', verbose=1),
        callbacks.ReduceLROnPlateau(
            monitor='val_mae', factor=0.5, patience=3, mode='min', verbose=1),
        callbacks.EarlyStopping(
            monitor='val_mae', patience=10, mode='min', 
            restore_best_weights=True, verbose=1),
    ]

def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['loss'],     label='train')
    axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_title(f'{name} – Loss (MSE)')
    axes[0].legend()
    axes[1].plot(history.history['mae'],     label='train')
    axes[1].plot(history.history['val_mae'], label='val')
    axes[1].set_title(f'{name} – MAE')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/{name.replace(' ','_')}_history.png", dpi=150)
    plt.show()

def sanity_check(model, X_sample, y_sample):
    preds = model.predict(X_sample[:10], verbose=0).flatten()
    print("\nSanity check – first 10 predictions vs ground truth:")
    for p, t in zip(preds, y_sample[:10]):
        print(f"  Predicted: {p:.3f}   True: {t:.3f}")
    print(f"  Prediction std: {preds.std():.4f}  (should be > 0.1 if learning)")

print("Helpers ready.")

# ── CELL 9: Evaluate Helper ──────────────────────────────────
def evaluate(model, test_ds, y_test, name="Model"):
    y_pred = model.predict(test_ds, verbose=0).flatten()
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = np.sqrt(np.mean((y_test - y_pred) ** 2))
    pcc, _ = pearsonr(y_test, y_pred)
    srcc,_ = spearmanr(y_test, y_pred)
 
    print(f"\n{'='*48}")
    print(f"  {name}")
    print(f"{'='*48}")
    print(f"  PCC  : {pcc:.4f}")
    print(f"  SRCC : {srcc:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"{'='*48}\n")
 
    # Scatter plot
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, alpha=0.3, s=10, color='steelblue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1)
    plt.xlabel("Ground Truth Score")
    plt.ylabel("Predicted Score")
    plt.title(f"{name}\nPCC={pcc:.3f}  MAE={mae:.3f}")
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/{name.replace(' ','_')}_scatter.png", dpi=150)
    plt.show()
 
    return {'pcc': pcc, 'srcc': srcc, 'mae': mae, 'rmse': rmse, 'preds': y_pred}
 
print("Evaluate helper ready.")
 

 # ── CELL 10: Train ResNet50 ──────────────────────────────────
print("="*50)
print("TRAINING ResNet50")
print("="*50)

resnet_model, resnet_base = build_model('resnet50', trainable_base=False)
compile_model(resnet_model, lr=cfg.LR_HEAD)

print("\n── Stage 1: Head only (BN layers unfrozen) ──")
hist1 = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg.EPOCHS_HEAD,
    callbacks=get_callbacks("resnet50_s1"),
    verbose=1)

# Sanity check after Stage 1
sanity_check(resnet_model, X_val, y_val)

# Stage 2 — unfreeze top 50% of layers
total = len(resnet_base.layers)
freeze_until = total // 4
for layer in resnet_base.layers[:freeze_until]:
    layer.trainable = False
for layer in resnet_base.layers[freeze_until:]:
    layer.trainable = True

# Keep all BN layers trainable
for layer in resnet_base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

compile_model(resnet_model, lr=cfg.LR_FINE)
print(f"\n── Stage 2: Fine-tuning top {total - freeze_until} layers ──")
hist2 = resnet_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg.EPOCHS_FINE,
    callbacks=get_callbacks("resnet50_s2"),
    verbose=1)

plot_history(hist2, "ResNet50 Fine-tune")

# Sanity check after Stage 2
sanity_check(resnet_model, X_val, y_val)
print("ResNet50 training complete.")

# ── CELL 11: Train VGG16 ─────────────────────────────────────
print("="*50)
print("TRAINING VGG16")
print("="*50)
 
vgg_model, vgg_base = build_model('vgg16', trainable_base=False)
compile_model(vgg_model, lr=cfg.LR_HEAD)
 
print("\n── Stage 1: Head only (base frozen) ──")
vgg_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg.EPOCHS_HEAD,
    callbacks=get_callbacks("vgg16_s1"),
    verbose=1)
 
total_vgg = len(vgg_base.layers)
for layer in vgg_base.layers[:total_vgg // 2]:
    layer.trainable = False
for layer in vgg_base.layers[total_vgg // 2:]:
    layer.trainable = True
 
compile_model(vgg_model, lr=cfg.LR_FINE)
print(f"\n── Stage 2: Fine-tuning top {total_vgg // 2} layers ──")
hist_vgg = vgg_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=cfg.EPOCHS_FINE,
    callbacks=get_callbacks("vgg16_s2"),
    verbose=1)
 
plot_history(hist_vgg, "VGG16 Fine-tune")
print("VGG16 training complete.")
 
 # ── CELL 12: Evaluate Both Models ────────────────────────────
results = {}
results['ResNet50'] = evaluate(resnet_model, test_ds, y_test, "ResNet50")
results['VGG16']    = evaluate(vgg_model,    test_ds, y_test, "VGG16")
 
# ── CELL 13: Ensemble (ResNet50 × 3) ─────────────────────────
print("Training 2 more ResNet50 models for ensemble...")
ensemble_preds = [results['ResNet50']['preds']]
 
for seed in [1, 2]:
    tf.random.set_seed(seed)
    np.random.seed(seed)
 
    m, b = build_model('resnet50', trainable_base=False)
    compile_model(m, lr=cfg.LR_HEAD)
    m.fit(train_ds, validation_data=val_ds,
          epochs=cfg.EPOCHS_HEAD,
          callbacks=get_callbacks(f"resnet_seed{seed}_s1"),
          verbose=0)
 
    for layer in b.layers[:len(b.layers)//2]:
        layer.trainable = False
    for layer in b.layers[len(b.layers)//2:]:
        layer.trainable = True
 
    compile_model(m, lr=cfg.LR_FINE)
    m.fit(train_ds, validation_data=val_ds,
          epochs=cfg.EPOCHS_FINE,
          callbacks=get_callbacks(f"resnet_seed{seed}_s2"),
          verbose=0)
 
    preds = m.predict(test_ds, verbose=0).flatten()
    ensemble_preds.append(preds)
    print(f"  Seed {seed} done.")
 
ensemble_avg   = np.mean(ensemble_preds, axis=0)
mae_e          = mean_absolute_error(y_test, ensemble_avg)
rmse_e         = np.sqrt(np.mean((y_test - ensemble_avg) ** 2))
pcc_e, _       = pearsonr(y_test, ensemble_avg)
srcc_e, _      = spearmanr(y_test, ensemble_avg)
 
results['Ensemble_R3'] = {
    'pcc': pcc_e, 'srcc': srcc_e,
    'mae': mae_e, 'rmse': rmse_e,
    'preds': ensemble_avg}
 
print(f"\n── Ensemble (ResNet50×3) ──")
print(f"  PCC={pcc_e:.4f}  SRCC={srcc_e:.4f}  MAE={mae_e:.4f}  RMSE={rmse_e:.4f}")
 
 # ── CELL 14: Final Results Table ─────────────────────────────
import pandas as pd
 
print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)
print(f"{'Model':<25} {'PCC':>6} {'SRCC':>6} {'MAE':>6} {'RMSE':>6}")
print("-"*60)
for name, r in results.items():
    print(f"{name:<25} {r['pcc']:>6.4f} {r['srcc']:>6.4f} "
          f"{r['mae']:>6.4f} {r['rmse']:>6.4f}")
print("="*60)
 
# Save to CSV  →  copy these numbers into your report
df_results = pd.DataFrame({
    k: {m: v for m, v in r.items() if m != 'preds'}
    for k, r in results.items()
}).T
df_results.to_csv(f"{cfg.OUTPUT_DIR}/results_summary.csv")
print(f"\nSaved → {cfg.OUTPUT_DIR}/results_summary.csv")
 
# ── CELL 15: Ablation Study ───────────────────────────────────

# ── Paste your final results ──────────────────────────────────
pcc = 0.8856
mae = 0.2422

# ── Config ───────────────────────────────────────────────────
IMG_SIZE    = 224
BATCH_SIZE  = 16
LR_HEAD     = 1e-3
LR_FINE     = 1e-5

# ── Reload data and rebuild splits ───────────────────────────
data = np.load("/kaggle/input/datasets/pranavchandane/scut-fbp5500-v2-facial-beauty-scores/scut_fbp5500-cmprsd.npz")
X    = data['X']
y    = data['y'].astype(np.float32)
del data

X_resized = np.zeros((len(X), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i in range(len(X)):
    X_resized[i] = tf.image.resize(X[i], (IMG_SIZE, IMG_SIZE)).numpy().astype(np.uint8)
del X

X_train, X_test, y_train, y_test = train_test_split(
    X_resized, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.10, random_state=42)
del X_resized

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# ── Dataset factory ───────────────────────────────────────────
rotation_layer = tf.keras.layers.RandomRotation(10/360)

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    return img, label

def make_dataset(X, y, shuffle=False, augment_data=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(len(X))
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(
            lambda img, label: (eff_preprocess(
                tf.cast(img, tf.float32)), label),
            num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ── Ablation helpers ──────────────────────────────────────────
print("\nRunning ablation study...")
ablation = {}

def cleanup():
    gc.collect()
    tf.keras.backend.clear_session()

def make_ablation_datasets(augment=True):
    tr = make_dataset(X_train, y_train, augment_data=augment, shuffle=True)
    vl = make_dataset(X_val,   y_val,   augment_data=False)
    te = make_dataset(X_test,  y_test,  augment_data=False)
    return tr, vl, te

def run_abl(name, augment=True, finetune=True, dropout=True):
    cleanup()
    tr, vl, te = make_ablation_datasets(augment=augment)

    base_a = EfficientNetB3(weights='imagenet', include_top=False,
                            input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_a.trainable = False

    if dropout:
        m = tf.keras.Sequential([
            base_a,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, dtype='float32')
        ])
    else:
        m = tf.keras.Sequential([
            base_a,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, dtype='float32')
        ])

    m.compile(
        optimizer=tf.keras.optimizers.Adam(LR_HEAD, clipnorm=1.0),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae'])
    m.fit(tr, validation_data=vl, epochs=10, verbose=0)

    if finetune:
        total_a = len(base_a.layers)
        for layer in base_a.layers[:total_a // 3]:
            layer.trainable = False
        for layer in base_a.layers[total_a // 3:]:
            layer.trainable = True
        m.compile(
            optimizer=tf.keras.optimizers.Adam(LR_FINE, clipnorm=1.0),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mae'])
        m.fit(tr, validation_data=vl, epochs=20, verbose=0)

    pred    = m.predict(te, verbose=0).flatten()
    pcc_a,_ = pearsonr(y_test, pred)
    mae_a   = mean_absolute_error(y_test, pred)
    print(f"  {name:<35} PCC={pcc_a:.3f}  MAE={mae_a:.3f}")

    del m, base_a, pred, tr, vl, te
    cleanup()
    return {'pcc': pcc_a, 'mae': mae_a}

# ── Run ablations ─────────────────────────────────────────────
ablation['Full Model'] = {'pcc': pcc, 'mae': mae}
print(f"  {'Full Model':<35} PCC={pcc:.3f}  MAE={mae:.3f}  ← full model")

ablation['w/o Augmentation']  = run_abl('w/o Augmentation',  augment=False)
ablation['w/o Fine-tuning']   = run_abl('w/o Fine-tuning',   finetune=False)
ablation['w/o Dropout']       = run_abl('w/o Dropout',       dropout=False)

# ── Print summary ─────────────────────────────────────────────
print("\n── Ablation Summary ──")
print(f"{'Configuration':<35} {'PCC':>6} {'MAE':>6}")
print("-" * 50)
for name, r in ablation.items():
    marker = "  ← full model" if name == "Full Model" else ""
    print(f"{name:<35} {r['pcc']:>6.3f} {r['mae']:>6.3f}{marker}")

# ─────────────────────────── CELL 16: Grad-CAM  ──────────────────────────────────
import os
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
# Match your original config
class Config:
    OUTPUT_DIR = "/kaggle/working"
    IMG_SIZE = 224
    CHANNELS = 3
    TEST_RATIO = 0.10
    VAL_RATIO = 0.10
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
cfg = Config()
# Reload best ResNet checkpoint
model_path = f"{cfg.OUTPUT_DIR}/resnet50_s2_best.keras"
resnet_model = tf.keras.models.load_model(model_path)
print("Loaded:", model_path)
# Reload processed dataset and recreate same test split
X_processed = np.load(f"{cfg.OUTPUT_DIR}/X_processed.npy")
y_scores    = np.load(f"{cfg.OUTPUT_DIR}/y_scores.npy")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_processed,
    y_scores,
    test_size=cfg.TEST_RATIO,
    random_state=42
)
del X_processed, y_scores, X_trainval, y_trainval
gc.collect()
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
# Grad-CAM functions
def make_gradcam(img_array, model, last_conv_layer):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        score = preds[:, 0]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                     # aim for (H, W)

    # ── FIX: guard against 0-d or 1-d output ──────────────────────
    heatmap = heatmap.numpy()
    if heatmap.ndim < 2:
        heatmap = np.ones((7, 7), dtype=np.float32)  # fallback blank
    heatmap = heatmap.astype(np.float32)
    # ──────────────────────────────────────────────────────────────

    heatmap = np.maximum(heatmap, 0)
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val
    return heatmap  # guaranteed 2-D float32


def overlay_gradcam(img_norm, model, conv_layer="conv5_block3_out"):
    inp = img_norm[np.newaxis].astype("float32")
    heatmap = make_gradcam(inp, model, conv_layer)

    # ── FIX: ensure correct shape before resize ───────────────────
    heatmap = np.squeeze(heatmap).astype(np.float32)
    if heatmap.ndim != 2:
        heatmap = np.ones((7, 7), dtype=np.float32)
    # ─────────────────────────────────────────────────────────────

    heatmap = cv2.resize(heatmap, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    mean_arr = np.array(cfg.MEAN, dtype=np.float32)
    std_arr  = np.array(cfg.STD,  dtype=np.float32)
    img_disp = np.clip((img_norm * std_arr + mean_arr) * 255, 0, 255).astype(np.uint8)

    overlay = np.uint8(0.55 * img_disp + 0.45 * heatmap)
    score = model.predict(inp, verbose=0)[0, 0]
    return overlay, score
# Pick 6 samples
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), 6, replace=False)
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle(
    "Grad-CAM – Regions Influencing Beauty Prediction",
    fontsize=14,
    fontweight="bold"
)
for ax, idx in zip(axes.flatten(), sample_idx):
    overlay, pred_score = overlay_gradcam(
        X_test[idx],
        resnet_model,
        conv_layer="conv5_block3_out"
    )
    ax.imshow(overlay)
    ax.set_title(
        f"True: {y_test[idx]:.2f} | Pred: {pred_score:.2f}",
        fontsize=10
    )
    ax.axis("off")
plt.tight_layout()
save_path = f"{cfg.OUTPUT_DIR}/gradcam.png"
plt.savefig(save_path, dpi=150)
plt.show()
print(f"Grad-CAM saved → {save_path}")

# ── CELL 17: Save & List ─────────────────────────────────────
# Reload models if cleared by ablation
if 'resnet_model' not in dir():
    resnet_model = tf.keras.models.load_model(
        f"{cfg.OUTPUT_DIR}/resnet50_final.keras")
if 'vgg_model' not in dir():
    vgg_model = tf.keras.models.load_model(
        f"{cfg.OUTPUT_DIR}/vgg16_final.keras")

resnet_model.save(f"{cfg.OUTPUT_DIR}/resnet50_beauty_final.keras")
vgg_model.save(f"{cfg.OUTPUT_DIR}/vgg16_beauty_final.keras")

print("\n" + "="*50)
print("ALL DONE. Files in /kaggle/working/:")
print("="*50)
for fname in sorted(os.listdir(cfg.OUTPUT_DIR)):
    fpath = os.path.join(cfg.OUTPUT_DIR, fname)
    size  = os.path.getsize(fpath) / 1e6
    print(f"  {fname:<45} {size:.1f} MB")
 
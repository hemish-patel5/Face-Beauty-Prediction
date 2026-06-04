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
from tensorflow.keras.applications import EfficientNetB3, VGG16
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
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
    BATCH_SIZE  = 16
    EPOCHS_HEAD = 15
    EPOCHS_FINE = 40
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

# ── CELL 4: Preprocess aligned images ────────────────────────
CACHE_EFF = f"{cfg.OUTPUT_DIR}/X_eff.npy"
CACHE_VGG = f"{cfg.OUTPUT_DIR}/X_vgg.npy"
CACHE_Y   = f"{cfg.OUTPUT_DIR}/y_scores.npy"

if os.path.exists(CACHE_EFF):
    print("Loading from cache...")
    X_eff    = np.load(CACHE_EFF)
    X_vgg    = np.load(CACHE_VGG)
    y_scores = np.load(CACHE_Y)
    print(f"Loaded: {X_eff.shape}")
else:
    print("Preprocessing images...")
    X_f32 = X_aligned.astype(np.float32)
    X_eff = eff_preprocess(X_f32.copy())
    X_vgg = vgg_preprocess(X_f32.copy())
    del X_f32
    y_scores = y.copy()
    np.save(CACHE_EFF, X_eff)
    np.save(CACHE_VGG, X_vgg)
    np.save(CACHE_Y,   y_scores)
    print("Done.")

# ── CELL 5: Train / Val / Test Split ─────────────────────────
from sklearn.model_selection import train_test_split

idx = np.arange(len(X_eff))
idx_trainval, idx_test = train_test_split(idx, test_size=cfg.TEST_RATIO, random_state=42)
idx_train, idx_val     = train_test_split(idx_trainval,
                                           test_size=cfg.VAL_RATIO / (1 - cfg.TEST_RATIO),
                                           random_state=42)

X_eff_train, X_eff_val, X_eff_test = X_eff[idx_train], X_eff[idx_val], X_eff[idx_test]
X_vgg_train, X_vgg_val, X_vgg_test = X_vgg[idx_train], X_vgg[idx_val], X_vgg[idx_test]
y_train, y_val, y_test = y_scores[idx_train], y_scores[idx_val], y_scores[idx_test]

# Free the full arrays — only keep the splits
del X_eff, X_vgg

print(f"Train : {len(X_eff_train)}")
print(f"Val   : {len(X_eff_val)}")
print(f"Test  : {len(X_eff_test)}")
 
 # ── CELL 6: Augmentation & Data Pipelines ────────────────────
rotation_layer = tf.keras.layers.RandomRotation(10/360)

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    noise = tf.random.normal(shape=tf.shape(image), stddev=0.01)
    image = image + noise
    image = rotation_layer(tf.expand_dims(image, 0))[0]
    return image, label

# NOTE: no brightness/contrast augmentation — those are invalid on preprocessed inputs

def make_dataset(X, y, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X.astype('float32'), y.astype('float32')))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), seed=42)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

eff_train_ds = make_dataset(X_eff_train, y_train, augment_data=True,  shuffle=True)
eff_val_ds   = make_dataset(X_eff_val,   y_val)
eff_test_ds  = make_dataset(X_eff_test,  y_test)

vgg_train_ds = make_dataset(X_vgg_train, y_train, augment_data=True,  shuffle=True)
vgg_val_ds   = make_dataset(X_vgg_val,   y_val)
vgg_test_ds  = make_dataset(X_vgg_test,  y_test)

# ── CELL 7: Model Builder ────────────────────────────────────
def build_model(arch='efficientnetb3', trainable_base=False):
    inp = tf.keras.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS))

    if arch == 'efficientnetb3':
        base = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=inp)
    else:
        base = VGG16(weights='imagenet', include_top=False, input_tensor=inp)

    base.trainable = trainable_base

    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(cfg.DROPOUT_1)(x)
    x = layers.Dense(cfg.DENSE_UNITS, activation='relu',
                     kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(cfg.DROPOUT_2)(x)
    out = layers.Dense(1, activation='linear', dtype='float32',
                       kernel_initializer='glorot_uniform', name='score')(x)

    return models.Model(inputs=inp, outputs=out), base

print("Model builder ready.")

# ── CELL 8: Training Helpers ─────────────────────────────────
def compile_model(model, lr):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss='mse',
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
print("TRAINING EfficientNetB3")
print("="*50)

eff_model, eff_base = build_model('efficientnetb3', trainable_base=False)
compile_model(eff_model, lr=cfg.LR_HEAD)

print("\n── Stage 1: Head only ──")
eff_model.fit(eff_train_ds, validation_data=eff_val_ds,
              epochs=cfg.EPOCHS_HEAD,
              callbacks=get_callbacks("efficientnetb3_s1"), verbose=1)

sanity_check(eff_model, X_eff_val, y_val)

total = len(eff_base.layers)
freeze_until = total // 3
for layer in eff_base.layers[:freeze_until]:
    layer.trainable = False
for layer in eff_base.layers[freeze_until:]:
    layer.trainable = True
for layer in eff_base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

compile_model(eff_model, lr=cfg.LR_FINE)
print(f"\n── Stage 2: Fine-tuning top {total - freeze_until} layers ──")
eff_model.fit(eff_train_ds, validation_data=eff_val_ds,
              epochs=cfg.EPOCHS_FINE,
              callbacks=get_callbacks("efficientnetb3_s2"), verbose=1)

plot_history(eff_model.history, "EfficientNetB3 Fine-tune")
sanity_check(eff_model, X_eff_val, y_val)
print("EfficientNetB3 training complete.")

# ── CELL 11: Train VGG16 ─────────────────────────────────────
print("="*50)
print("TRAINING VGG16")
print("="*50)
 
vgg_model, vgg_base = build_model('vgg16', trainable_base=False)
compile_model(vgg_model, lr=cfg.LR_HEAD)
 
print("\n── Stage 1: Head only (base frozen) ──")
vgg_model.fit(
    vgg_train_ds,
    validation_data=vgg_val_ds,
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
results['EfficientNetB3'] = evaluate(eff_model, eff_test_ds, y_test, "EfficientNetB3")
results['VGG16']          = evaluate(vgg_model, vgg_test_ds, y_test, "VGG16")
 
# ── CELL 13: Ensemble (EfficientNetB3 × 3) ───────────────────
print("Training 2 more EfficientNetB3 models for ensemble...")
ensemble_preds = [results['EfficientNetB3']['preds']]

for seed in [1, 2]:
    tf.random.set_seed(seed)
    np.random.seed(seed)

    m, b = build_model('efficientnetb3', trainable_base=False)
    compile_model(m, lr=cfg.LR_HEAD)
    m.fit(eff_train_ds, validation_data=eff_val_ds,
          epochs=cfg.EPOCHS_HEAD,
          callbacks=get_callbacks(f"efficientnetb3_seed{seed}_s1"),
          verbose=0)

    total = len(b.layers)
    freeze_until = total // 3
    for layer in b.layers[:freeze_until]:
        layer.trainable = False
    for layer in b.layers[freeze_until:]:
        layer.trainable = True
    for layer in b.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    compile_model(m, lr=cfg.LR_FINE)
    m.fit(eff_train_ds, validation_data=eff_val_ds,
          epochs=cfg.EPOCHS_FINE,
          callbacks=get_callbacks(f"efficientnetb3_seed{seed}_s2"),
          verbose=0)

    preds = m.predict(eff_test_ds, verbose=0).flatten()
    ensemble_preds.append(preds)
    del m, b  # free each model before training the next
    print(f"  Seed {seed} done.")

ensemble_avg = np.mean(ensemble_preds, axis=0)
mae_e        = mean_absolute_error(y_test, ensemble_avg)
rmse_e       = np.sqrt(np.mean((y_test - ensemble_avg) ** 2))
pcc_e, _     = pearsonr(y_test, ensemble_avg)
srcc_e, _    = spearmanr(y_test, ensemble_avg)

results['Ensemble_EfficientNetB3x3'] = {
    'pcc': pcc_e, 'srcc': srcc_e,
    'mae': mae_e, 'rmse': rmse_e,
    'preds': ensemble_avg}

print(f"\n── Ensemble (EfficientNetB3×3) ──")
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
 
 # ── CELL 15: Ablation Study ──────────────────────────────────
# Tests what happens when we remove each component one at a time.
# Uses fewer epochs for speed (~30 min total).
 
print("\nRunning ablation study...")
ablation = {}
 
def quick_eval(X_in, y_in, name, epochs_h=5, epochs_f=10):
    Xtr, Xte, ytr, yte = train_test_split(X_in, y_in,
                                           test_size=0.10, random_state=42)
    Xtr, Xv, ytr, yv   = train_test_split(Xtr, ytr,
                                           test_size=0.10, random_state=42)
    dtr = make_dataset(Xtr, ytr, augment_data=False)
    dv  = make_dataset(Xv, yv)
    dte = make_dataset(Xte, yte)
 
    m, b = build_model('resnet50', trainable_base=False)
    compile_model(m, lr=cfg.LR_HEAD)
    m.fit(dtr, validation_data=dv, epochs=epochs_h, verbose=0)
 
    for layer in b.layers[len(b.layers)//2:]:
        layer.trainable = True
    compile_model(m, lr=cfg.LR_FINE)
    m.fit(dtr, validation_data=dv, epochs=epochs_f, verbose=0)
 
    pred = m.predict(dte, verbose=0).flatten()
    pcc, _ = pearsonr(yte, pred)
    mae    = mean_absolute_error(yte, pred)
    print(f"  {name:<42} PCC={pcc:.3f}  MAE={mae:.3f}")
    return {'pcc': pcc, 'mae': mae}
 
# Full model result (already trained – use real test result)
ablation['Full ResNet50 (all components)'] = {
    'pcc': results['ResNet50']['pcc'],
    'mae': results['ResNet50']['mae']}
 
# Without alignment: use raw resized images without normalisation
mean_arr = np.array(cfg.MEAN, dtype=np.float32)
std_arr  = np.array(cfg.STD,  dtype=np.float32)
X_raw = np.zeros_like(X_processed)
for i in range(len(X)):
    img = cv2.resize(X[i], (cfg.IMG_SIZE, cfg.IMG_SIZE)).astype(np.float32) / 255.0
    X_raw[i] = img   # no normalisation, no alignment
ablation['w/o Face Alignment'] = quick_eval(X_raw, y_scores, 'w/o Face Alignment')
 
# Without augmentation: train without augment_data=True
def make_no_aug_dataset(Xd, yd):
    ds = tf.data.Dataset.from_tensor_slices((Xd, yd))
    return ds.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
 
Xtr2, Xte2, ytr2, yte2 = train_test_split(X_processed, y_scores,
                                            test_size=0.10, random_state=42)
Xtr2, Xv2, ytr2, yv2   = train_test_split(Xtr2, ytr2,
                                            test_size=0.10, random_state=42)
dtr2 = make_no_aug_dataset(Xtr2, ytr2)
dv2  = make_no_aug_dataset(Xv2, yv2)
dte2 = make_no_aug_dataset(Xte2, yte2)
 
m2, b2 = build_model('resnet50', trainable_base=False)
compile_model(m2, lr=cfg.LR_HEAD)
m2.fit(dtr2, validation_data=dv2, epochs=5, verbose=0)
for layer in b2.layers[len(b2.layers)//2:]:
    layer.trainable = True
compile_model(m2, lr=cfg.LR_FINE)
m2.fit(dtr2, validation_data=dv2, epochs=10, verbose=0)
pred2    = m2.predict(dte2, verbose=0).flatten()
pcc2, _  = pearsonr(yte2, pred2)
mae2     = mean_absolute_error(yte2, pred2)
print(f"  {'w/o Data Augmentation':<42} PCC={pcc2:.3f}  MAE={mae2:.3f}")
ablation['w/o Data Augmentation'] = {'pcc': pcc2, 'mae': mae2}
 
# Without fine-tuning: head-only training only
m3, _ = build_model('resnet50', trainable_base=False)
compile_model(m3, lr=cfg.LR_HEAD)
dtr3 = make_dataset(X_train, y_train, augment_data=True, shuffle=True)
m3.fit(dtr3, validation_data=val_ds, epochs=10, verbose=0)
pred3    = m3.predict(test_ds, verbose=0).flatten()
pcc3, _  = pearsonr(y_test, pred3)
mae3     = mean_absolute_error(y_test, pred3)
print(f"  {'w/o Fine-tuning (head only)':<42} PCC={pcc3:.3f}  MAE={mae3:.3f}")
ablation['w/o Fine-tuning (head only)'] = {'pcc': pcc3, 'mae': mae3}
 
# Without dropout: build model without dropout layers
def build_no_dropout(arch='resnet50'):
    inp  = tf.keras.Input(shape=(cfg.IMG_SIZE, cfg.IMG_SIZE, cfg.CHANNELS))
    base = ResNet50(weights='imagenet', include_top=False, input_tensor=inp)
    base.trainable = False
    x    = layers.GlobalAveragePooling2D()(base.output)
    x    = layers.Dense(cfg.DENSE_UNITS, activation='relu')(x)
    out  = layers.Dense(1, activation='linear')(x)
    return models.Model(inputs=inp, outputs=out), base
 
m4, b4 = build_no_dropout()
compile_model(m4, lr=cfg.LR_HEAD)
m4.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
for layer in b4.layers[len(b4.layers)//2:]:
    layer.trainable = True
compile_model(m4, lr=cfg.LR_FINE)
m4.fit(train_ds, validation_data=val_ds, epochs=10, verbose=0)
pred4    = m4.predict(test_ds, verbose=0).flatten()
pcc4, _  = pearsonr(y_test, pred4)
mae4     = mean_absolute_error(y_test, pred4)
print(f"  {'w/o Dropout Regularisation':<42} PCC={pcc4:.3f}  MAE={mae4:.3f}")
ablation['w/o Dropout Regularisation'] = {'pcc': pcc4, 'mae': mae4}
 
# Print ablation summary
print("\n── Ablation Summary ──")
print(f"{'Configuration':<42} {'PCC':>6} {'MAE':>6}")
print("-"*57)
for name, r in ablation.items():
    marker = " ← full model" if "Full" in name else ""
    print(f"{name:<42} {r['pcc']:>6.3f} {r['mae']:>6.3f}{marker}")
 
# Save ablation results
df_abl = pd.DataFrame(ablation).T
df_abl.to_csv(f"{cfg.OUTPUT_DIR}/ablation_results.csv")
print(f"\nSaved → {cfg.OUTPUT_DIR}/ablation_results.csv")
 
 # ── CELL 16: Grad-CAM Visualisation ──────────────────────────
# Shows which face regions the model focuses on.
# Save gradcam.png and use it in your demo video.
 
def make_gradcam(img_array, model, last_conv_layer):
    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        score = preds[:, 0]
    grads      = tape.gradient(score, conv_out)
    pooled     = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out   = conv_out[0]
    heatmap    = conv_out @ pooled[..., tf.newaxis]
    heatmap    = tf.squeeze(heatmap)
    heatmap    = tf.maximum(heatmap, 0)
    heatmap    = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()
 
def overlay_gradcam(img_norm, model, conv_layer='top_conv'):  # was 'conv5_block3_out'
    inp     = img_norm[np.newaxis]
    heatmap = make_gradcam(inp, model, conv_layer)
    heatmap = cv2.resize(heatmap, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
 
    # Denormalise image back to [0,255] for display
    mean_arr = np.array(cfg.MEAN, dtype=np.float32)
    std_arr  = np.array(cfg.STD,  dtype=np.float32)
    img_disp = (img_norm * std_arr + mean_arr)
    img_disp = np.clip(img_disp * 255, 0, 255).astype(np.uint8)
 
    overlay  = np.uint8(0.55 * img_disp + 0.45 * heatmap)
    score    = model.predict(inp, verbose=0)[0, 0]
    return overlay, score
 
# Pick 6 random test samples and visualise
sample_idx = np.random.choice(len(X_test), 6, replace=False)
fig, axes  = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("Grad-CAM – Regions Influencing Beauty Prediction",
             fontsize=14, fontweight='bold')
 
for ax, idx in zip(axes.flatten(), sample_idx):
    overlay, pred_score = overlay_gradcam(X_test[idx], resnet_model)
    true_score = y_test[idx]
    ax.imshow(overlay)
    ax.set_title(f"True: {true_score:.2f}  |  Pred: {pred_score:.2f}",
                 fontsize=10)
    ax.axis('off')
 
plt.tight_layout()
plt.savefig(f"{cfg.OUTPUT_DIR}/gradcam.png", dpi=150)
plt.show()
print("Grad-CAM saved → gradcam.png")
 
 # ── CELL 17: Save Final Models ───────────────────────────────
eff_model.save(f"{cfg.OUTPUT_DIR}/efficientnetb3_beauty_final.keras")
vgg_model.save(f"{cfg.OUTPUT_DIR}/vgg16_beauty_final.keras")
 
print("\n" + "="*50)
print("ALL DONE. Files saved to /kaggle/working/:")
print("="*50)
for fname in sorted(os.listdir(cfg.OUTPUT_DIR)):
    fpath = os.path.join(cfg.OUTPUT_DIR, fname)
    size  = os.path.getsize(fpath) / 1e6
    print(f"  {fname:<45} {size:.1f} MB")
 
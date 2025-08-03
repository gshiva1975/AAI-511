import os
import sys
import json
import logging
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from fastmcp import FastMCP
from io import StringIO

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_FREEZE = 5
EPOCHS_FINETUNE = 5
MODEL_PATH = "composer_etl.h5"
LABELS_PATH = "composer_labels.json"

# Logging - redirect to stderr to avoid JSON parsing issues
logger = logging.getLogger("composer_pdf_classifier")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# Convert PDF to image
def pdf_to_image(pdf_path: str, img_size: int):
    try:
        pages = convert_from_path(pdf_path, first_page=1, last_page=1)
        if not pages:
            return None
        img = pages[0].convert("RGB").resize((img_size, img_size), Image.BICUBIC)
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        logger.error(f"PDF error for {pdf_path}: {e}")
        return None

# Load dataset from directory
def load_dataset(data_dir: str, img_size: int):
    X, y = [], []
    composers = sorted(d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)))
    if not composers:
        raise RuntimeError(f"No composer folders in {data_dir}")

    logger.info(f"Found {len(composers)} composers")
    
    for comp in composers:
        comp_count = 0
        for root, _, files in os.walk(os.path.join(data_dir, comp)):
            for fn in files:
                if fn.lower().endswith(".pdf"):
                    img = pdf_to_image(os.path.join(root, fn), img_size)
                    if img is not None:
                        X.append(img)
                        y.append(comp)
                        comp_count += 1
        logger.info(f"{comp}: {comp_count} PDFs loaded")

    if not X:
        raise RuntimeError(f"No valid PDF images found under {data_dir}")

    X = np.stack(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    logger.info(f"Total dataset: {len(X)} images, {len(le.classes_)} classes")
    return X, y_enc, list(le.classes_)

# Build model with EfficientNetB0
def build_transfer_model(input_shape, n_classes, lr=1e-4):
    try:
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
        logger.info("Using ImageNet pretrained weights")
    except:
        base = EfficientNetB0(weights=None, include_top=False, input_shape=input_shape)
        logger.info("Using random weights (ImageNet not available)")
    
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation="softmax")(x)
    model = Model(base.input, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base

# MCP server
mcp = FastMCP("composer_pdf_classifier")

@mcp.tool()
async def train_model(
    data_dir: str,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    epochs_freeze: int = EPOCHS_FREEZE,
    epochs_finetune: int = EPOCHS_FINETUNE,
    model_out: str = MODEL_PATH,
    labels_out: str = LABELS_PATH
) -> dict:
    """
    Trains EfficientNetB0 on sheet music PDFs and saves model + labels.
    """
    try:
        logger.info(f"Starting training with data_dir={data_dir}")
        logger.info(f"Parameters: img_size={img_size}, batch_size={batch_size}, epochs_freeze={epochs_freeze}, epochs_finetune={epochs_finetune}")
        
        # Load dataset
        X, y, classes = load_dataset(data_dir, img_size)
        
        # Train/validation split
        strat = y if len(y) * 0.2 >= 2 * len(classes) else None
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=strat, random_state=42)
        logger.info(f"Train set: {len(X_tr)}, Validation set: {len(X_va)}")

        # Create datasets
        tr_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(len(X_tr)).batch(batch_size)
        va_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va)).batch(batch_size)

        # Build model
        model, base = build_transfer_model((img_size, img_size, 3), len(classes))
        logger.info(f"Model built with {len(classes)} output classes")

        # Phase 1: Freeze base and train classifier
        logger.info("Phase 1: Training frozen base model")
        history1 = model.fit(tr_ds, validation_data=va_ds, epochs=epochs_freeze, verbose=0)
        freeze_acc = max(history1.history['val_accuracy'])
        logger.info(f"Phase 1 completed. Best validation accuracy: {freeze_acc:.4f}")

        # Phase 2: Unfreeze and fine-tune
        logger.info("Phase 2: Fine-tuning with unfrozen layers")
        base.trainable = True
        for layer in base.layers[:-20]:
            layer.trainable = False
        
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        history2 = model.fit(tr_ds, validation_data=va_ds, epochs=epochs_finetune, verbose=0)
        finetune_acc = max(history2.history['val_accuracy'])
        logger.info(f"Phase 2 completed. Best validation accuracy: {finetune_acc:.4f}")

        # Save model and labels
        model.save(model_out)
        with open(labels_out, "w") as f:
            json.dump(classes, f)
        
        logger.info(f"Model saved to {model_out}")
        logger.info(f"Labels saved to {labels_out}")

        return {
            "message": f"✅ Model trained successfully and saved to {model_out}",
            "model_path": model_out,
            "labels_path": labels_out,
            "num_classes": len(classes),
            "total_samples": len(X),
            "train_samples": len(X_tr),
            "val_samples": len(X_va),
            "freeze_phase_val_acc": float(freeze_acc),
            "finetune_phase_val_acc": float(finetune_acc),
            "classes": classes[:10]  # Show first 10 classes
        }
        
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

@mcp.tool()
async def predict_composer(
    pdf_path: str,
    model_path: str = MODEL_PATH,
    labels_path: str = LABELS_PATH,
    img_size: int = IMG_SIZE
) -> dict:
    """
    Predict composer from a sheet music PDF using trained model.
    """
    try:
        model = load_model(model_path)
        classes = json.load(open(labels_path))
        img = pdf_to_image(pdf_path, img_size)
        if img is None:
            return {"error": "Failed to load or convert PDF"}

        prob = model.predict(img[np.newaxis, ...], verbose=0)[0]
        idx = int(np.argmax(prob))
        
        # Get top 3 predictions
        top_indices = np.argsort(prob)[::-1][:3]
        top_predictions = [
            {"composer": classes[i], "confidence": float(prob[i])}
            for i in top_indices
        ]
        
        return {
            "predicted_composer": classes[idx],
            "confidence": float(prob[idx]),
            "top_predictions": top_predictions
        }
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run(transport="stdio")


import tensorflow as tf
import logging
import os
import pandas as pd
import numpy as np
import json
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS_FREEZE = 5
EPOCHS_FINETUNE = 5
MODEL_PATH = "mobilenetv2_music_model.h5"
LABELS_PATH = "mobilenetv2_music_labels.json"
METADATA_CSV_PATH = "/Users/gshiva/AA-511/example/mcp-server-example/mcp-server/sheetmusic/test_image/processed_pdfs/metadata.csv"
PDF_IMAGE_DIR = "/Users/gshiva/AA-511/example/mcp-server-example/mcp-server/sheetmusic/test_image/processed_pdfs"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress TensorFlow logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_dataset_from_metadata(metadata_path: str, image_dir: str, img_size: int):
    """Load dataset from CSV metadata file containing image filenames and composer labels"""
    df = pd.read_csv(metadata_path)
    X, y = [], []

    logger.info(f"Loading dataset from {metadata_path}")
    logger.info(f"Found {len(df)} entries in metadata")

    for _, row in df.iterrows():
        # Handle both .pdf.png and .png extensions
        filename = row['filename']
        if not filename.endswith('.png'):
            filename = filename + '.png'

        img_path = os.path.join(image_dir, filename)

        if not os.path.exists(img_path):
            logger.warning(f"Image not found: {img_path}")
            continue

        try:
            img = load_img(img_path, target_size=(img_size, img_size))
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(row['composer'])
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            continue

    if len(X) == 0:
        raise ValueError("No valid images found in the dataset")

    X = np.stack(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    classes = le.classes_.tolist()

    logger.info(f"Loaded {len(X)} images with {len(classes)} unique composers")
    logger.info(f"Composers: {classes}")

    return X, y_enc, classes

def build_mobilenet_model(input_shape, n_classes, lr=1e-4):
    """Build MobileNetV2 model for composer classification"""
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation="softmax")(x)

    model = Model(base.input, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base

def train_model(
    metadata_path: str = METADATA_CSV_PATH,
    image_dir: str = PDF_IMAGE_DIR,
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    epochs_freeze: int = EPOCHS_FREEZE,
    epochs_finetune: int = EPOCHS_FINETUNE,
    model_out: str = MODEL_PATH,
    labels_out: str = LABELS_PATH
) -> dict:
    """Train MobileNetV2 model on sheet music images for composer classification"""
    try:
        logger.info("Starting model training...")

        # Load dataset
        X, y, classes = load_dataset_from_metadata(metadata_path, image_dir, img_size)

        # Split dataset
        strat = y if len(y) * 0.2 >= 2 * len(classes) else None
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=strat, random_state=42)

        logger.info(f"Training set: {len(X_tr)} samples")
        logger.info(f"Validation set: {len(X_va)} samples")

        # Create datasets
        tr_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).shuffle(len(X_tr)).batch(batch_size)
        va_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va)).batch(batch_size)

        # Build model
        model, base = build_mobilenet_model((img_size, img_size, 3), len(classes))

        # Capture training output to prevent verbose output
        training_output = StringIO()

        logger.info("Phase 1: Training with frozen base layers")
        # Phase 1: Train with frozen base
        with redirect_stdout(training_output), redirect_stderr(training_output):
            history1 = model.fit(tr_ds, validation_data=va_ds, epochs=epochs_freeze, verbose=0)

        logger.info("Phase 2: Fine-tuning with unfrozen layers")
        # Phase 2: Fine-tune
        base.trainable = True
        for layer in base.layers[:-20]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        with redirect_stdout(training_output), redirect_stderr(training_output):
            history2 = model.fit(tr_ds, validation_data=va_ds, epochs=epochs_finetune, verbose=0)

        # Save model and labels
        model.save(model_out)
        with open(labels_out, "w") as f:
            json.dump(classes, f)

        final_acc = history2.history['val_accuracy'][-1] if 'val_accuracy' in history2.history else 0

        logger.info(f"Training completed! Final validation accuracy: {final_acc:.4f}")

        return {
            "message": f"✅ Model trained and saved to {model_out}",
            "model_path": model_out,
            "labels_path": labels_out,
            "num_classes": len(classes),
            "classes": classes,
            "final_validation_accuracy": float(final_acc),
        }

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {"error": f"Training failed: {str(e)}"}

def predict_composer(
    image_path: str,
    model_path: str = MODEL_PATH,
    labels_path: str = LABELS_PATH,
    img_size: int = IMG_SIZE
) -> dict:
    """Predict composer from sheet music image"""
    try:
        # Load model and labels
        if not os.path.exists(model_path):
            return {"error": f"Model not found at {model_path}. Please train the model first."}

        if not os.path.exists(labels_path):
            return {"error": f"Labels not found at {labels_path}. Please train the model first."}

        # Suppress TensorFlow output during model loading and prediction
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            model = load_model(model_path)

        with open(labels_path, "r") as f:
            classes = json.load(f)

        if not os.path.exists(image_path):
            return {"error": f"Image not found at {image_path}"}

        img = load_img(image_path, target_size=(img_size, img_size))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction with suppressed output
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            prob = model.predict(img, verbose=0)[0]

        idx = int(np.argmax(prob))

        # Top-3 predictions
        top_indices = np.argsort(prob)[::-1][:3]
        top_predictions = [
            {"composer": classes[i], "confidence": float(prob[i])}
            for i in top_indices
        ]

        return {
            "predicted_composer": classes[idx],
            "confidence": float(prob[idx]),
            "top_predictions": top_predictions,
            "image_path": image_path
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": f"Prediction failed: {str(e)}"}

def main():
    """Main function to run the music classifier"""
    parser = argparse.ArgumentParser(description="MobileNetV2 Music Composer Classifier")
    parser.add_argument("command", choices=["train", "predict"], help="Command to execute")
    parser.add_argument("--image_path", type=str, help="Path to image for prediction")
    parser.add_argument("--metadata_path", type=str, default=METADATA_CSV_PATH, help="Path to metadata CSV file")
    parser.add_argument("--image_dir", type=str, default=PDF_IMAGE_DIR, help="Directory containing images")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to save/load model")
    parser.add_argument("--labels_path", type=str, default=LABELS_PATH, help="Path to save/load labels")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Image size for processing")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--epochs_freeze", type=int, default=EPOCHS_FREEZE, help="Epochs for frozen training")
    parser.add_argument("--epochs_finetune", type=int, default=EPOCHS_FINETUNE, help="Epochs for fine-tuning")

    args = parser.parse_args()

    if args.command == "train":
        logger.info("Starting training process...")
        result = train_model(
            metadata_path=args.metadata_path,
            image_dir=args.image_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            epochs_freeze=args.epochs_freeze,
            epochs_finetune=args.epochs_finetune,
            model_out=args.model_path,
            labels_out=args.labels_path
        )
        print(json.dumps(result, indent=2))

    elif args.command == "predict":
        if not args.image_path:
            logger.error("--image_path is required for prediction")
            return

        logger.info(f"Predicting composer for: {args.image_path}")
        result = predict_composer(
            image_path=args.image_path,
            model_path=args.model_path,
            labels_path=args.labels_path,
            img_size=args.img_size
        )
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

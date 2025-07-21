import os
import sys
import glob
import json
import logging
import numpy as np
import pretty_midi
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from fastmcp import FastMCP

# Configure logging
logger = logging.getLogger("MidiComposerService")
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stderr)
h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(h)

# Utilities
def extract_notes(path: str, max_len: int) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(path)
    pitches = [note.pitch
               for inst in pm.instruments if not inst.is_drum
               for note in inst.notes]
    seq = pitches[:max_len]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    return np.array(seq, dtype=np.int32)

def prepare_dataset(root: str, max_len: int):
    root = os.path.expanduser(root)
    composers = [d for d in os.listdir(root)
                 if os.path.isdir(os.path.join(root, d))]
    seqs, labels = [], []
    for comp in composers:
        for pat in ("*.mid", "*.midi"):
            for f in glob.glob(os.path.join(root, comp, pat)):
                try:
                    seqs.append(extract_notes(f, max_len))
                    labels.append(comp)
                except Exception as e:
                    logger.error(f"Parse error {f}: {e}")
    if not seqs:
        raise ValueError(f"No MIDI data under {root}")
    X = np.stack(seqs)
    enc = LabelEncoder()
    y_idx = enc.fit_transform(labels)
    y = tf.keras.utils.to_categorical(y_idx, num_classes=len(enc.classes_))
    return X, y, list(enc.classes_)

def build_model(max_len: int, n_classes: int):
    m = Sequential([
        Embedding(128, 64),
        Conv1D(128, 5, activation="relu"), MaxPooling1D(2), Dropout(0.3),
        Conv1D(64, 3, activation="relu"),  MaxPooling1D(2), Dropout(0.3),
        LSTM(128, return_sequences=True), Dropout(0.3),
        LSTM(64),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])
    return m

def predict_file(model, classes, path: str, max_len: int):
    seq = extract_notes(path, max_len).reshape(1, max_len)
    pr = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(pr))
    return classes[idx], float(pr[idx])

# MCP app
mcp = FastMCP("midi_composer_service")

# Tool with hyperparameter tuning
@mcp.tool()
async def get_docs(
    dataset_dir: str = "~/Downloads/music",
    midi_path: str = "~/Downloads/music/Bach/example.mid",
    seq_length: int = 200,
    epochs: int = 5,
    batch_size: int = 8
) -> dict:
    """
    Train CNN-LSTM on MIDI dataset and predict composer with hyperparameter tuning.
    """
    X, y, classes = prepare_dataset(dataset_dir, seq_length)
    logger.info(f"Dataset: {X.shape[0]} samples, composers={classes}")

    learning_rates = [1e-3, 5e-4, 1e-4]
    batch_sizes = [8, 16]

    best_val_acc = 0
    best_model = None
    best_params = {}

    for lr in learning_rates:
        for bs in batch_sizes:
            logger.info(f"Trying lr={lr}, batch_size={bs}")
            model = build_model(seq_length, len(classes))
            model.compile(
                optimizer=Adam(learning_rate=lr),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            hist = model.fit(
                X, y,
                epochs=epochs,
                batch_size=bs,
                validation_split=0.1,
                verbose=0
            )
            val_acc = float(hist.history["val_accuracy"][-1])
            logger.info(f"Validation accuracy={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_params = {"learning_rate": lr, "batch_size": bs}
                logger.info("✅ New best model found!")

    midi_path = os.path.expanduser(midi_path)
    if not os.path.isfile(midi_path):
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")
    composer, confidence = predict_file(best_model, classes, midi_path, seq_length)
    logger.info(f"Predicted {composer} with {confidence:.2%} confidence")

    return {
        "predicted_composer": composer,
        "confidence": confidence,
        "training_accuracy": best_val_acc,
        "best_hyperparameters": best_params
    }

# Run over stdio
if __name__ == "__main__":
    mcp.run(transport="stdio")


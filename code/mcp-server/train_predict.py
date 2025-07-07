
#!/usr/bin/env python3
import sys
import os
import glob
import logging
import numpy as np
import pretty_midi
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from fastmcp import FastMCP

# ─── 1) stderr logging ─────────────────────────────────────────────────────────
logger = logging.getLogger("MidiComposerService")
logger.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stderr)
h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(h)

# ─── 2) Core utilities ─────────────────────────────────────────────────────────
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
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset not found: {root}")

    composers = [d for d in os.listdir(root)
                 if os.path.isdir(os.path.join(root, d))]
    seqs, labels = [], []
    for comp in composers:
        for pat in ("*.mid","*.midi"):
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
    m.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def predict_file(model, classes, path: str, max_len: int):
    seq = extract_notes(path, max_len).reshape(1, max_len)
    probs = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])

# ─── 3) MCP app over stdio ────────────────────────────────────────────────────
mcp = FastMCP("midi_composer_service")

@mcp.tool()
async def train_model(
    dataset_dir: str = "~/Downloads/music",
    seq_length: int = 200,
    epochs: int = 5,
    batch_size: int = 8,
    model_path: str = "midi_model.keras",
    classes_path: str = "midi_classes.npy"
) -> dict:
    """
    Trains a CNN-LSTM on all composers under dataset_dir
    and saves model + class list.
    """
    X, y, classes = prepare_dataset(dataset_dir, seq_length)
    logger.info(f"Training on {len(X)} samples for composers={classes}")
    model = build_model(seq_length, len(classes))
    hist = model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0.1
    )
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
    np.save(classes_path, np.array(classes))
    train_acc = float(hist.history["accuracy"][-1])
    return {
        "message": f"Trained {len(X)} samples; accuracy={train_acc:.2%}",
        "model_path": model_path,
        "classes_path": classes_path,
        "training_accuracy": train_acc
    }

@mcp.tool()
async def predict_composer(
    midi_path: str,
    seq_length: int = 200,
    model_path: str = "midi_model.keras",
    classes_path: str = "midi_classes.npy"
) -> dict:
    """
    Loads a saved model+classes and predicts the composer of midi_path.
    """
    midi_path = os.path.expanduser(midi_path)
    if not os.path.isfile(midi_path):
        raise FileNotFoundError(f"MIDI not found: {midi_path}")
    model = load_model(model_path)
    classes = np.load(classes_path, allow_pickle=True).tolist()
    composer, confidence = predict_file(model, classes, midi_path, seq_length)
    return {
        "composer": composer,
        "confidence": confidence
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")


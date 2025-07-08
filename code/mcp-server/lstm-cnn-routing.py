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
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# ─── 2) Core utilities ─────────────────────────────────────────────────────────
def extract_notes(path: str, max_len: int) -> np.ndarray:
    pm = pretty_midi.PrettyMIDI(path)
    pitches = [
        note.pitch
        for inst in pm.instruments if not inst.is_drum
        for note in inst.notes
    ]
    seq = pitches[:max_len]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    return np.array(seq, dtype=np.int32)

def prepare_dataset(root: str, max_len: int):
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset not found: {root}")

    composers = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ]
    seqs, labels = [], []
    for comp in composers:
        comp_dir = os.path.join(root, comp)
        for pat in ("*.mid","*.midi"):
            for fpath in glob.glob(os.path.join(comp_dir, pat)):
                try:
                    seqs.append(extract_notes(fpath, max_len))
                    labels.append(comp)
                except Exception as e:
                    logger.error(f"Parse error {fpath}: {e}")
    if not seqs:
        raise ValueError(f"No MIDI data under {root}")

    X = np.stack(seqs)  # (N, max_len)
    enc = LabelEncoder()
    y_idx = enc.fit_transform(labels)
    y = tf.keras.utils.to_categorical(y_idx, num_classes=len(enc.classes_))
    return X, y, list(enc.classes_)

def build_lstm_model(max_len: int, n_classes: int):
    m = Sequential([
        Embedding(128, 64, input_length=max_len),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(n_classes, activation="softmax")
    ])
    m.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def build_cnn_lstm_model(max_len: int, n_classes: int):
    m = Sequential([
        Embedding(128, 64, input_length=max_len),
        Conv1D(128, 5, activation="relu"), MaxPooling1D(2), Dropout(0.3),
        Conv1D(64, 3, activation="relu"),  MaxPooling1D(2), Dropout(0.3),
        LSTM(128, return_sequences=True), Dropout(0.3),
        LSTM(64),
        Dense(64, activation="relu"),
        Dense(n_classes, activation="softmax")
    ])
    m.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def predict_with_model(model, classes, sequence: np.ndarray):
    probs = model.predict(sequence, verbose=0)[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])

# ─── 3) MCP app over stdio ────────────────────────────────────────────────────
mcp = FastMCP("midi_composer_service")

# ─── 4) Training tool ─────────────────────────────────────────────────────────
@mcp.tool()
async def train_model(
    dataset_dir:  str = "~/Downloads/music",
    seq_length:   int = 200,
    epochs:       int = 5,
    batch_size:   int = 8,
    model_type:   str = None,               # "lstm", "cnn", or None=both
    lstm_path:    str = "lstm_model.keras",
    cnn_path:     str = "cnn_model.keras",
    classes_path: str = "midi_classes.npy"
) -> dict:
    """
    Trains an LSTM, a CNN-LSTM, or both depending on model_type.
    Returns training accuracies and saved paths.
    """
    X, y, classes = prepare_dataset(dataset_dir, seq_length)
    os.makedirs(os.path.dirname(lstm_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(cnn_path) or ".", exist_ok=True)

    results = {"classes": classes}
    types = [model_type] if model_type in ("lstm","cnn") else ("lstm","cnn")
    for mtype in types:
        if mtype == "lstm":
            model = build_lstm_model(seq_length, len(classes))
            logger.info("Training LSTM model...")
            hist = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                             verbose=1, validation_split=0.1)
            model.save(lstm_path)
            acc = hist.history["accuracy"][-1]
            results["lstm"] = {"path": lstm_path, "accuracy": float(acc)}
        else:  # "cnn"
            model = build_cnn_lstm_model(seq_length, len(classes))
            logger.info("Training CNN-LSTM model...")
            hist = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                             verbose=1, validation_split=0.1)
            model.save(cnn_path)
            acc = hist.history["accuracy"][-1]
            results["cnn"] = {"path": cnn_path, "accuracy": float(acc)}

    # save composer classes
    np.save(classes_path, np.array(classes))
    results["classes_path"] = classes_path
    return results

# ─── 5) Prediction tool with routing ─────────────────────────────────────────
@mcp.tool()
async def predict_composer(
    midi_path:    str,
    seq_length:   int = 200,
    model_type:   str = None,                   # prefer "lstm" or "cnn"
    lstm_path:    str = "lstm_model.keras",
    cnn_path:     str = "cnn_model.keras",
    classes_path: str = "midi_classes.npy",
    cnn_threshold:int = 300
) -> dict:
    """
    Chooses which model to load (via model_type or seq_length heuristic),
    predicts composer of midi_path, and returns result.
    """
    mt = (model_type or "").lower()
    if mt not in ("lstm","cnn"):
        mt = "cnn" if seq_length > cnn_threshold else "lstm"
    logger.info(f"Routing to {mt.upper()} model")

    model_path = cnn_path if mt=="cnn" else lstm_path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = load_model(model_path)
    classes = np.load(classes_path, allow_pickle=True).tolist()

    seq = extract_notes(os.path.expanduser(midi_path), seq_length).reshape(1, seq_length)
    composer, confidence = predict_with_model(model, classes, seq)
    return {
        "model_type": mt,
        "composer": composer,
        "confidence": confidence
    }

# ─── 6) Run over stdio ────────────────────────────────────────────────────────
if __name__ == "__main__":
    mcp.run(transport="stdio")


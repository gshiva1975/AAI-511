
import os
import sys
import json
import logging
import numpy as np
import pretty_midi
import tensorflow as tf
from sklearn.model_selection import train_test_split
from fastmcp import FastMCP
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

# Configure logging to STDERR
logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logger = logging.getLogger("mcp-cnn-midi")

mcp = FastMCP("cnn_midi_classifier")

#  Extract MIDI note features
def extract_midi_features(midi_path):
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append([note.pitch, note.velocity, note.start, note.end])
        return notes
    except Exception as e:
        logger.error(f"Failed to parse {midi_path}: {e}")
        return []

#  Prepare dataset: Walk through train/test/dev and extract MIDI features
def load_dataset(data_dir, segment_duration=5.0, num_pitch_bins=128, num_duration_bins=50):
    X, y = [], []
    composers = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    composer_to_label = {composer: idx for idx, composer in enumerate(composers)}
    logger.info(f"Found composers: {composers}")

    for composer in composers:
        composer_dir = os.path.join(data_dir, composer)
        for midi_file in os.listdir(composer_dir):
            if midi_file.lower().endswith(".mid"):
                midi_path = os.path.join(composer_dir, midi_file)
                notes = extract_midi_features(midi_path)
                if notes:
                    features = segment_features(notes, segment_duration, num_pitch_bins, num_duration_bins)
                    X.extend(features)
                    y.extend([composer_to_label[composer]] * len(features))
    return np.array(X), np.array(y), composer_to_label

#  Segment features from MIDI notes
def segment_features(notes, segment_duration=5.0, num_pitch_bins=128, num_duration_bins=50):
    segmented_features = []
    notes.sort(key=lambda x: x[2])  # Sort by start time
    total_duration = notes[-1][3] if notes else 0

    start_time = 0
    while start_time < total_duration:
        end_time = start_time + segment_duration
        segment_notes = [n for n in notes if start_time <= n[2] < end_time]
        if segment_notes:
            pitch_hist = np.zeros(num_pitch_bins)
            duration_hist = np.zeros(num_duration_bins)
            velocity_sum, note_count = 0, len(segment_notes)

            for pitch, velocity, n_start, n_end in segment_notes:
                pitch_hist[pitch] += 1
                dur_bin = min(int((n_end - n_start) * 10), num_duration_bins - 1)
                duration_hist[dur_bin] += 1
                velocity_sum += velocity

            feature_vector = list(pitch_hist) + list(duration_hist)
            feature_vector.append(note_count)
            feature_vector.append(velocity_sum / note_count if note_count > 0 else 0)
            segmented_features.append(feature_vector)
        start_time += segment_duration
    return segmented_features

#  Build CNN Model
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, 5, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(2),
        Dropout(0.3),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

#  MCP Tool: Train Model
@mcp.tool()
async def train_model(data_root: str = "~/Downloads/Composer_Dataset/NN_midi_files_extended/train", epochs: int = 20, batch_size: int = 32):
    data_root = os.path.expanduser(data_root)
    X, y, composer_map = load_dataset(data_root)

    X = np.expand_dims(X, axis=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_cnn((X.shape[1], 1), len(composer_map))

    class MCPLogging(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch+1}/{epochs}: Loss={logs['loss']:.4f}, Val Acc={logs.get('val_accuracy', 0):.4f}", file=sys.stderr)

    model.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[MCPLogging()])
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    model.save("cnn_midi_model.h5")
    with open("composer_labels.json", "w") as f:
        json.dump(composer_map, f)

    return {"message": "Training complete", "accuracy": float(acc), "loss": float(loss)}

#  MCP Tool: Predict
@mcp.tool()
async def predict_midi(midi_path: str):
    model = tf.keras.models.load_model("cnn_midi_model.h5")
    composer_map = json.load(open("composer_labels.json"))
    inv_map = {v: k for k, v in composer_map.items()}

    notes = extract_midi_features(midi_path)
    features = segment_features(notes)
    features = np.expand_dims(np.array(features), axis=-1)
    probs = model.predict(features, verbose=0)
    avg_prob = np.mean(probs, axis=0)
    pred_idx = int(np.argmax(avg_prob))
    return {"predicted_composer": inv_map[pred_idx], "confidence": float(avg_prob[pred_idx])}

if __name__ == "__main__":
    mcp.run(transport="stdio")


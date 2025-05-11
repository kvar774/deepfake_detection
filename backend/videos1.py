import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, Flatten, GRU, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Enable Mixed Precision for Speed-up
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("mixed_float16")

IMG_SIZE = (128, 128)
FRAME_COUNT = 10
BATCH_SIZE = 16
DATASET_PATH = "./videos_dataset/"

def is_valid_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    cap.release()
    return True

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMG_SIZE)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)
    cap.release()
    frames = np.array(frames, dtype=np.float32)
    pad_size = FRAME_COUNT - frames.shape[0]
    if pad_size > 0:
        frames = np.pad(frames, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode='constant')
    return frames

def load_dataset():
    video_paths, labels = [], []
    for label in ["Real", "Fake"]:
        folder_path = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist!")
            continue
        for video_file in os.listdir(folder_path):
            video_path = os.path.join(folder_path, video_file)
            if is_valid_video(video_path):
                video_paths.append(video_path)
                labels.append(0 if label == "Real" else 1)
    return video_paths, labels

# Define a valid Keras Sequence class
class VideoDataGenerator(Sequence):
    def __init__(self, video_paths, labels, batch_size, shuffle=True):
        self.video_paths = video_paths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.video_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_paths = self.video_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        X = [preprocess_video(path) for path in batch_paths]
        return np.array(X, dtype=np.float32), np.array(batch_labels)
    
    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.video_paths, self.labels))
            np.random.shuffle(temp)
            self.video_paths, self.labels = zip(*temp)

X_paths, y_labels = load_dataset()
X_train_paths, X_test_paths, y_train_labels, y_test_labels = train_test_split(X_paths, y_labels, test_size=0.2, random_state=42)

train_gen = VideoDataGenerator(X_train_paths, y_train_labels, batch_size=BATCH_SIZE)
test_gen = VideoDataGenerator(X_test_paths, y_test_labels, batch_size=BATCH_SIZE)

base_cnn = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
model = Sequential([
    tf.keras.layers.Input(shape=(FRAME_COUNT, 128, 128, 3)),
    TimeDistributed(base_cnn),
    TimeDistributed(Flatten()),
    Bidirectional(GRU(64, return_sequences=True)),
    Bidirectional(GRU(32, return_sequences=False)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid', dtype=tf.float32)
])

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_gen, epochs=5, validation_data=test_gen)

# ✅ Save model in correct format
model.save("deepfake_video_classifier_fast.keras", save_format="keras")
print("✅ Model saved as deepfake_video_classifier_fast.keras")
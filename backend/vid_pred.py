import numpy as np 
import tensorflow as tf
import cv2
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Constants
IMG_SIZE = (128, 128)
FRAME_COUNT = 10
MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_video_classifier_fast.keras")

# ✅ Function to create the model architecture
def create_video_model():
    base_model = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(128, 128, 3)
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAME_COUNT, 128, 128, 3)),
        tf.keras.layers.TimeDistributed(base_model),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, return_sequences=False)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float32)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ✅ Function to preprocess the video
def preprocess_video(video_path):
    print("🎥 Extracting frames from video...")
    sys.stdout.flush()

    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // FRAME_COUNT)
    
    count = 0
    while len(frames) < FRAME_COUNT:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            frame = frame.astype(np.float32) / 255.0  
            frames.append(frame)
        count += 1
    cap.release()
    
    # Pad frames if the video was too short
    while len(frames) < FRAME_COUNT:
        frames.append(frames[-1])
    
    print(f"✅ {len(frames)} frames extracted successfully.")
    sys.stdout.flush()
    return np.array(frames[:FRAME_COUNT], dtype=np.float32)

# ✅ Function to run the prediction
def predict_video(video_path):
    print("🔄 Rebuilding model architecture...")
    sys.stdout.flush()

    # Build the model
    model = create_video_model()
    
    # ✅ Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        sys.stdout.flush()
        return "Error", 0.0
    
    # ✅ Load the model weights
    try:
        print("📂 Loading model weights...")
        sys.stdout.flush()
        model.load_weights(MODEL_PATH)
        print("✅ Model weights loaded successfully.")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.stdout.flush()
        return "Error", 0.0
    
    # ✅ Preprocess the video
    try:
        print(f"🎥 Processing video: {video_path}")
        sys.stdout.flush()
        video_data = preprocess_video(video_path)
        video_data = np.expand_dims(video_data, axis=0)
        print(f"✅ Video preprocessed successfully. Shape: {video_data.shape}")
        sys.stdout.flush()
    except Exception as e:
        print(f"❌ Error processing video: {e}")
        sys.stdout.flush()
        return "Error", 0.0
    
    # ✅ Run the prediction
    try:
        print("🔍 Running prediction...")
        sys.stdout.flush()
        prediction = model.predict(video_data, verbose=0)[0][0]

        # ✅ Calculate confidence
        confidence = float(1 - prediction if prediction < 0.5 else prediction)
        class_label = "Real" if prediction < 0.5 else "Fake"
        print(f"🔢 Raw Prediction Value: {prediction}")
        sys.stdout.flush()

        # ✅ Format result exactly like image prediction output
        print(f"✅ Final result: {class_label} (Confidence: {confidence * 100:.2f}%)")
        sys.stdout.flush()
        return class_label, confidence * 100
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        sys.stdout.flush()
        return "Error", 0.0

# ✅ Main function
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: No video file provided.")
        sys.stdout.flush()
        sys.exit(1)
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        sys.stdout.flush()
        sys.exit(1)
    
    # ✅ Run the prediction
    result, confidence = predict_video(video_path)
    sys.stdout.flush()

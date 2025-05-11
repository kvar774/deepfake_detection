import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import sys
import os
import io

sys.stdout.reconfigure(encoding='utf-8')

# Correct model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "deepfake_image_classifier.keras")

# Updated model architecture (matching training script)
def create_deepfake_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.GlobalAveragePooling2D(),

        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to rebuild model, load weights, and run prediction
def load_model_and_predict(img_path):
    print("🔄 Rebuilding model architecture...")
    sys.stdout.flush()

    model = create_deepfake_model()

    if not os.path.exists(MODEL_PATH):
        error_message = f"❌ Error: Model file not found at {MODEL_PATH}"
        print(error_message)
        sys.stdout.flush()
        return error_message

    try:
        print("📂 Loading model weights...")
        sys.stdout.flush()
        model.load_weights(MODEL_PATH)
        print("✅ Model weights loaded successfully.")
        sys.stdout.flush()
    except Exception as e:
        error_message = f"❌ Error loading model: {e}"
        print(error_message)
        sys.stdout.flush()
        return error_message

    try:
        print(f"📸 Loading image: {img_path}")
        sys.stdout.flush()
        img = image.load_img(img_path, target_size=(128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"✅ Image loaded. Shape: {img_array.shape}")
        sys.stdout.flush()
    except Exception as e:
        error_message = f"❌ Error processing image: {e}"
        print(error_message)
        sys.stdout.flush()
        return error_message

    try:
        print("🔍 Running prediction...")
        sys.stdout.flush()
        prediction = model.predict(img_array, verbose=0)

        if prediction.shape != (1, 1):
            error_message = f"❌ Error: Unexpected prediction shape {prediction.shape}"
            print(error_message)
            sys.stdout.flush()
            return error_message

        prediction_value = prediction[0][0]
        print(f"🔢 Raw Prediction Value: {prediction_value}")  # Debugging line

        confidence = abs(prediction_value - 0.5) * 2 * 100
        class_label = "Fake" if prediction_value >= 0.2 else "Real"
        result = f"{class_label} (Confidence: {confidence:.2f}%)"

        print(f"✅ Final result: {result}")
        sys.stdout.flush()
        return result
    except Exception as e:
        error_message = f"❌ Prediction error: {e}"
        print(error_message)
        sys.stdout.flush()
        return error_message


def main():
    if len(sys.argv) < 2:
        print("❌ Error: No image file provided.")
        sys.stdout.flush()
        return
    
    img_path = sys.argv[1]
    load_model_and_predict(img_path)  # No extra print statement here
    sys.stdout.flush()

if __name__ == "__main__":
    main()

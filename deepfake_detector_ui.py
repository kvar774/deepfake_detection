import streamlit as st 
import os
import subprocess
from PIL import Image

# Ensure temp directory exists
temp_dir = "temp"
os.makedirs(temp_dir, exist_ok=True)

# Custom CSS for enhanced UI
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffdead !important;
        background-size: cover;
        color: #2d3436;
    }
    .title {
        text-align: center;
        color: #dc143c !important;  /* Ensuring the color is applied */
        font-size: 54px !important;
        font-family: "Times New Roman" !important;
        text-shadow: 2px 2px 5px #7cfc00;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #b22222 !important;
        font-size: 24px !important;
        font-family: "Times New Roman" !important;
        margin-bottom: 30px;
    }
    .result-box {
        background-color: #dfe6e9;
        color: #2d3436;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #0984e3;
        margin-top: 20px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="title">Deepfake Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image or video to check if it\'s Real or Fake</p>', unsafe_allow_html=True)

# Dropdown for file type selection
file_type = st.selectbox("Select file type", ["Image", "Video"])

# File upload section
uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "png", "mp4", "avi", "mov", "mkv"], label_visibility='collapsed')

def process_file(uploaded_file, file_type):
    """Save the uploaded file and run the appropriate model for prediction."""
    file_path = os.path.join(temp_dir, uploaded_file.name)

    # Save the uploaded file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"✅ File uploaded: {uploaded_file.name}")

    # Display the uploaded file
    if file_type == "Image":
        image = Image.open(file_path)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    elif file_type == "Video":
        st.video(file_path)

    # Choose the correct prediction script
    command = ["python", f"backend/{'img_pred.py' if file_type == 'Image' else 'vid_pred.py'}", file_path]

    # Run the script
    st.write(f"🚀 Running prediction on {uploaded_file.name} ...")
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")

    if result.returncode != 0:
        st.error(f"❌ Error: {result.stderr}")
    else:
        st.success("✅ Prediction Completed!")

        # Display prediction result in a rectangular box
        st.markdown('<div class="result-box">Prediction Result:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-box">{result.stdout if result.stdout else "⚠️ No valid output received."}</div>', unsafe_allow_html=True)

    # Clean up temp file
    os.remove(file_path)
    st.write("🗑️ Temporary file deleted.")

if uploaded_file is not None:
    process_file(uploaded_file, file_type)

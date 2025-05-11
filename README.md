# 🧠 DeepFake Detection using CNN + LSTM

A Deep Learning project to detect whether an image or video is **real** or **fake** using the power of **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks.

---

## 👋 Hello there, curious minds!

This project is designed to help identify manipulated media content—*DeepFakes*—using deep learning models trained on image and video data.
Whether you’re a student, researcher, or just an enthusiast, this project provides you a solid base to dive into media forensics using AI.

---

## 🚀 Getting Started

To run this project on your machine, you'll need to have the following Python packages installed:

```
1. streamlit
2. pillow
3. numpy
4. tensorflow
5. opencv-python
6. matplotlib
7. scikit-learn
```

You can install all of them in one go using:

```bash
pip install streamlit pillow numpy tensorflow opencv-python matplotlib scikit-learn
```

---

## 📦 Datasets
Due to the size of the datasets, they are **not included** in the repository. Please download them from the following sources:

🔗 **Image Dataset**
[DeepFake and Real Images – Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data)

🔗 **Video Dataset**
[DeepFake Video Dataset – Kaggle](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/data)

Once downloaded, place both datasets inside the `backend/` folder of this project.
📁 Your folder structure should look like this:

```
deepfake_detection/
├── backend/
│   ├── images/
│   └── videos/
├── training_scripts/
├── app.py
└── README.md
```

> ✨ **Tip**: Ensure your dataset folders are named properly. Modify the training scripts if your folder names are different.

---

## 🛠 Notes

1. All listed dependencies are **required**. Ensure they are installed before running the app.
2. **Adjust paths** and folder names in training scripts to match the names you give to the datasets.
3. The system uses a CNN for **spatial feature extraction** and an LSTM for **temporal sequence analysis**, especially in videos.

---

## 🎯 Final Thoughts

This project brings together **deep learning** and **media forensics** to combat the rising threat of DeepFakes.
Let’s use AI for truth. 💡

If you found this project interesting, feel free to ⭐ star it and contribute!
Happy Coding 👨‍💻👩‍💻

---

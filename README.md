# Medicinal-Plant-Leaf-Identification-in-Real-Time-Using-Deep-Learning-Model
This project focuses on building a real-time medicinal plant identification system that can classify plant species using leaf images and display relevant details about the plant. It combines deep learning (CNN - MobileNetV2), Flask web framework, and a SQLite database to provide an end-to-end solution that is fast, scalable, and user-friendly.
# 🌿 Medicinal Plant Leaf Identification in Real Time Using Deep Learning

This project is a **real-time medicinal plant identification system** that uses **deep learning** to classify plant species from leaf images and display their scientific names and medicinal uses.  
It is designed as an educational tool for students, researchers, and herbal medicine enthusiasts.  

---

## 🚀 Features
- **Deep Learning Model:** MobileNetV2-based CNN trained using transfer learning.
- **Real-Time Prediction:** Supports image upload or webcam capture.
- **Plant Details Retrieval:** Fetches plant information (scientific name, uses) from SQLite database.
- **Web Application:** Built using Flask with a clean HTML/CSS/Bootstrap frontend.
- **Offline Friendly:** Works locally on a laptop, no internet required after setup.

---

## 🧠 How It Works
1. **Training Phase**
   - Collect and preprocess a dataset of medicinal plant leaves.
   - Fine-tune MobileNetV2 on the dataset.
   - Save trained model as `leaf_model.h5`.

2. **Prediction Phase**
   - User uploads or captures a leaf image via the web interface.
   - Flask backend loads the trained model and predicts the plant name.
   - Flask queries SQLite database to fetch scientific name and medicinal uses.
   - Webpage displays results in a user-friendly format.

---

## 🛠 Tech Stack
- **Language:** Python 3.10+
- **Deep Learning:** TensorFlow, Keras, MobileNetV2
- **Web Framework:** Flask
- **Database:** SQLite
- **Frontend:** HTML, CSS, Bootstrap
- **Image Processing:** OpenCV
- **Development Tools:** VS Code, Jupyter Notebook, Google Colab

---

## 📂 Project Structure
```
project/
│
├── dataset/               # Leaf images organized by plant classes
├── model_training.ipynb   # Notebook for training and fine-tuning MobileNetV2
├── leaf_model.h5          # Trained model file (after training)
├── app.py                 # Flask backend
├── templates/             # HTML templates
├── static/                # CSS, JS, and images
├── plants.db              # SQLite database storing plant details
└── requirements.txt       # Dependencies
```

---

## 🎯 Expected Outcome
- Real-time identification of medicinal plants with >90% accuracy.
- Educational and scalable solution that can easily support more plant species in the future.

---

## 🌱 Future Enhancements
- Deploy model on mobile devices using TensorFlow Lite.
- Add plant disease detection module.
- Expand dataset and database to cover 100+ plant species.
- Deploy on cloud (Heroku/AWS) for multi-user access.

---

## 🖼 Sample Output
![Sample Result](docs/sample_result.png)

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

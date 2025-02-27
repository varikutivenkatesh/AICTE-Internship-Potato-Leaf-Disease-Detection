# AICTE-Internship-Potato-Leaf-Disease-Detection
🥔🍃 Potato Leaf Disease Detection
## Overview
This project is a deep learning-based system for detecting potato leaf diseases using Convolutional Neural Networks (CNNs). The model classifies potato leaves into three categories:

✅ Healthy

⚠️ Early Blight

❌ Late Blight

The system takes an image of a potato leaf as input, processes it using a trained CNN model, and predicts the disease type.

## Features
✔️ Automated disease detection using deep learning

✔️ Image preprocessing for enhanced accuracy

✔️ Trained on a diverse dataset for robust classification

✔️ Web/Mobile deployment for real-time detection

✔️ Fast and accurate classification

##  Tech Stack
Programming Language: Python 🐍

Deep Learning Framework: TensorFlow / PyTorch

Image Processing: OpenCV

Web Deployment: Flask / FastAPI (optional)

📂 Project Structure

📁 Potato_Leaf_Disease_Detection  
│── 📂 dataset/              # Contains training images  
│── 📂 models/               # Trained model files  
│── 📂 notebooks/            # Jupyter notebooks for training & testing  
│── 📂 static/               # Images & assets for web app  
│── 📂 templates/            # HTML templates (if using Flask)  
│── 📜 app.py                # Flask/Streamlit API for deployment  
│── 📜 model_train.py        # CNN model training script  
│── 📜 model_predict.py      # Leaf disease classification script  
│── 📜 requirements.txt      # Dependencies  
│── 📜 README.md             # Project documentation  

## how to run:
1.select command prompt

2.run -r requirements.txt

3.after we have use streamlit

4.run: streamlit run web.py

## Dataset
The model is trained on a Potato Leaf Dataset obtained from PlantVillage. It contains thousands of labeled images of healthy and diseased potato leaves.

## Model Architecture
The CNN model is built using TensorFlow/Keras with the following layers:

Conv2D + ReLU Activation (Feature Extraction)

MaxPooling2D (Dimensionality Reduction)

Flatten + Dense Layers (Classification)

Softmax Activation (Final Classification)

📈 Results & Accuracy

✅ Achieved 90%+ accuracy on test data

📉 Optimized model for fast inference on mobile/web apps

📌 Supports real-time detection for farmers & researchers

##  Future Enhancements
🔹 Improve accuracy with transfer learning (e.g., ResNet, MobileNet)

🔹 Deploy as a mobile app for farmers

🔹 Add multi-class classification for more plant diseases

## 🤝 Contributing
Feel free to fork this repo and contribute by submitting Pull Requests! 😊

## 📜 License
This project is open-source under the MIT License.

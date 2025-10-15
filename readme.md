# 🩻 Chest X-ray Pneumonia Detection

A deep learning-powered web application for detecting pneumonia from chest X-ray images. Built with CNN, FastAPI, and Streamlit.

## 📋 Table of Contents

- Overview
- Features
- Model Architecture
- Project Structure
- Installation
- Usage
- Deployment
- Dataset
- API Documentation
- Contributing

## 🔍 Overview

This project implements a Convolutional Neural Network (CNN) to classify chest X-ray images as either **Normal** or **Pneumonia**. The system consists of:

- **Deep Learning Model**: Custom CNN trained on chest X-ray dataset
- **FastAPI Backend**: RESTful API for serving predictions
- **Streamlit Frontend**: Interactive web interface for easy image upload and visualization

## ✨ Features

- 🎯 **Accurate Pneumonia Detection**: CNN model with high accuracy
- 🚀 **Real-time Predictions**: Fast inference through FastAPI
- 🖼️ **User-Friendly Interface**: Intuitive Streamlit web app
- 📊 **Confidence Scores**: Displays prediction confidence percentage
- 🔒 **CORS Enabled**: Secure cross-origin resource sharing
- 📱 **Responsive Design**: Works on desktop and mobile devices

## 🏗️ Model Architecture

The CNN model consists of the following layers:

```
Input Layer (150x150x3)
    ↓
Conv2D (32 filters, 3x3) + ReLU + BatchNormalization
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU + BatchNormalization
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3) + ReLU + BatchNormalization
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dense (128 units) + ReLU + Dropout (0.5)
    ↓
Dense (1 unit) + Sigmoid
```

**Training Details:**

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Epochs**: 20
- **Batch Size**: 32
- **Data Augmentation**: Random flip, rotation, zoom, and translation
- **Normalization**: Pixel values scaled to [0, 1]

## 📁 Project Structure

```
chest-xray-pneumonia-detection/
├── app.py                          # Streamlit frontend
├── main_fastapi.py                 # FastAPI backend
├── main.py                         # Model training script
├── chest_xray_cnn_model1_.h5       # Trained model (150MB)
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── chest_xray/                     # Dataset (not included in repo)
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── val/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test/
│       ├── NORMAL/
│       └── PNEUMONIA/
└── .gitignore
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
```

### Step 2: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset (Optional - for training)

Download the Chest X-ray dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place it in the `chest_xray/` directory.

## 🚀 Usage

### Option 1: Run Locally (Both Services)

#### Start FastAPI Backend

```bash
python main_fastapi.py
```

The API will be available at `http://localhost:8000`

#### Start Streamlit Frontend (in a new terminal)

```bash
streamlit run app.py
```

The web app will open at `http://localhost:8501`

### Option 2: Run Only FastAPI

```bash
uvicorn main_fastapi:app --reload --host 0.0.0.0 --port 8000
```

### Option 3: Train the Model from Scratch

```bash
python main.py
```

This will train a new model and save it as `chest_xray_cnn_model1_.h5`

## 🌐 Deployment

### Deploy on Render

1. **Push to GitHub**

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

2. **Deploy FastAPI**

   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Create new Web Service
   - Connect your GitHub repo
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main_fastapi:app --host 0.0.0.0 --port $PORT`

3. **Deploy Streamlit**

   - Update `FASTAPI_URL` in `app.py` with your FastAPI Render URL
   - Create another Web Service on Render
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## 📊 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset:

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: 5,863 X-ray images
- **Classes**: 2 (NORMAL, PNEUMONIA)
- **Split**:
  - Training: 5,216 images
  - Validation: 16 images
  - Test: 624 images

### Data Distribution

- **NORMAL**: ~1,583 images
- **PNEUMONIA**: ~4,273 images (bacterial and viral)

## 📚 API Documentation

### FastAPI Endpoints

#### `GET /`

Returns API status

```json
{
  "message": "Chest X-ray Classification API is running!"
}
```

#### `POST /predict`

Predicts pneumonia from chest X-ray image

**Request:**

- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**

```json
{
  "prediction": "Pneumonia" | "Normal",
  "confidence": 95.23
}
```

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"
```

**Example using Python:**

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("chest_xray.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 🙏 Acknowledgments

- Dataset provided by [Paul Mooney on Kaggle](https://www.kaggle.com/paultimothymooney)
- Inspiration from various deep learning tutorials and research papers
- Built with TensorFlow, FastAPI, and Streamlit

## 📧 Contact

For questions or feedback, please reach out:

- GitHub: [@tarunpatelofficial](https://github.com/tarunpatelofficial)
- Email: tarunpatelofficial@gmail.com

## ⚠️ Disclaimer

This application is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice.

---

**Made by Tarun Patel**

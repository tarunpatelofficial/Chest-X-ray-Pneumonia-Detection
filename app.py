import streamlit as st
import numpy as np
from PIL import Image
import io
import requests

# FastAPI endpoint URL
FASTAPI_URL = "https://chest-x-ray-pneumonia-detection-lzlb.onrender.com/predict"  # Change this if your FastAPI runs on a different host/port

# -------------------------------
# Function to call FastAPI
# -------------------------------
def predict_via_fastapi(image_file):
    """Send image to FastAPI endpoint and get prediction"""
    try:
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image_file.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Send POST request to FastAPI
        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        response = requests.post(FASTAPI_URL, files=files, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"FastAPI returned error: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to FastAPI server. Make sure it's running at " + FASTAPI_URL)
        return None
    except Exception as e:
        st.error(f"Error calling FastAPI: {str(e)}")
        return None

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🩻", layout="centered")
st.title("🩻 Chest X-ray Pneumonia Detection")
st.markdown("Upload a chest X-ray image and the model will classify it as **Normal** or **Pneumonia**.")
st.write("---")

uploaded_file = st.file_uploader("📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_container_width=True)
    st.write("---")

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing the image via FastAPI..."):
            # Call FastAPI endpoint
            api_response = predict_via_fastapi(img)
            
            if api_response:
                result = api_response.get("prediction")
                conf_percent = api_response.get("confidence")
                
                # Calculate raw prediction value for display
                if result == "Pneumonia":
                    pred = conf_percent / 100.0
                else:
                    pred = 1.0 - (conf_percent / 100.0)
                
                # Debug: Show raw prediction value
                st.info(f"🔢 Raw model output: {pred:.4f} (Higher value = model thinks it's Pneumonia)")
                
                # Prediction card
                st.subheader("🧠 Model Prediction Result")
                if result == "Pneumonia":
                    st.markdown(
                        f"<div style='background-color:#ffe5e5;padding:20px;border-radius:15px;text-align:center;'>"
                        f"<h2 style='color:#b30000;'>⚠️ Pneumonia Detected</h2>"
                        f"<p style='font-size:18px;'>Confidence: <b>{conf_percent:.2f}%</b></p></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='background-color:#e6ffed;padding:20px;border-radius:15px;text-align:center;'>"
                        f"<h2 style='color:#007a33;'>✅ Normal</h2>"
                        f"<p style='font-size:18px;'>Confidence: <b>{conf_percent:.2f}%</b></p></div>",
                        unsafe_allow_html=True
                    )
                st.progress(int(conf_percent))


st.write("---")
st.caption("🧬 Developed using **Streamlit** & **FastAPI** | Model: chest_xray_cnn_model1_.h5")
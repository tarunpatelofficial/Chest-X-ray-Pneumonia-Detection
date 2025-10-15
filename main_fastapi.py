from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from io import BytesIO
import uvicorn
from pydantic import BaseModel
from PIL import Image

app = FastAPI()

model = load_model("chest_xray_cnn_model1_.h5")

def preprocessing_image(img: Image.Image):
    img = img.resize((150, 150))          # match your model input size
    img_array = np.array(img) / 255.0     # normalize if you trained like this
    if img_array.shape[-1] == 4:          # remove alpha channel if exists
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    
@app.get("/")
def home():
    return {"message": "Chest X-ray Classification API is running!"}

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_array = preprocessing_image(image)

    pred = model.predict(image_array)[0][0]
    confidence = float(pred)
    result = "Pneumonia" if pred > 0.5 else "Normal"
    
    if result == "Pneumonia":
        return {"prediction": result, "confidence": confidence*100}
    else:
        return {"prediction": result, "confidence": (100 - confidence*100)}
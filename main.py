from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
from utils.preprocess import preprocess_image

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # later replace * with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_PATH = "model/custom_cnn.h5"
model = load_model(MODEL_PATH)
CLASS_NAMES = ["No Diabetic Retinopathy", "Mild", "Moderate", "Severe", "Proliferative DR"]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    processed = preprocess_image(image)
    preds = model.predict(processed)
    result = CLASS_NAMES[np.argmax(preds)]
    confidence = float(np.max(preds))
    return {"prediction": result, "confidence": confidence}

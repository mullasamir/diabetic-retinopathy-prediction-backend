from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import uuid
import datetime
from dotenv import load_dotenv
import motor.motor_asyncio
from utils.preprocess import preprocess_image

# -------------------- Load environment variables --------------------
load_dotenv()

app = FastAPI()

# -------------------- CORS Setup --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # replace later with your frontend deployed URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MongoDB Setup --------------------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "dr_predictions")

if not MONGO_URI:
    raise ValueError("Missing MONGO_URI in .env file")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
database = client[DB_NAME]
predictions_collection = database.get_collection("predictions")

# -------------------- Model Setup --------------------
MODEL_PATHS = {
    "Custom CNN": "model/custom_cnn.h5",
    "ResNet152": "model/ResNet152.h5",
    
}

CLASS_NAMES = ["No Diabetic Retinopathy", "Mild", "Moderate", "Severe", "Proliferative DR"]

# Cache models to avoid reloading every time
loaded_models = {}

def get_model(model_name: str):
    """Load model dynamically and cache it."""
    if model_name not in MODEL_PATHS:
        raise HTTPException(status_code=400, detail="Invalid model name selected.")
    
    if model_name not in loaded_models:
        model_path = MODEL_PATHS[model_name]
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")
        print(f"🔹 Loading model: {model_name}")
        loaded_models[model_name] = load_model(model_path)
    
    return loaded_models[model_name]

# Directory to store uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# -------------------- Prediction Endpoint --------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    try:
        # Save uploaded image
        file_ext = file.filename.split(".")[-1]
        file_name = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, file_name)
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Preprocess the image
        image = Image.open(io.BytesIO(contents))
        processed = preprocess_image(image)

        # Load selected model
        model = get_model(model_name)

        # Make prediction
        preds = model.predict(processed)
        result = CLASS_NAMES[np.argmax(preds)]
        confidence = round(float(np.max(preds) * 100), 2)

        # Store record in MongoDB
        record = {
            "image_name": file_name,
            "model_used": model_name,
            "prediction": result,
            "confidence": confidence,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        await predictions_collection.insert_one(record)

        return {
            "model_used": model_name,
            "prediction": result,
            "confidence": f"{confidence}%",
            "image_name": file_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Fetch All Prediction Records --------------------
@app.get("/predictions")
async def get_predictions():
    try:
        records = await predictions_collection.find().sort("timestamp", -1).to_list(10)
        for rec in records:
            rec["_id"] = str(rec["_id"])  # Convert ObjectId to string for JSON serialization
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Root Endpoint --------------------
@app.get("/")
async def root():
    return {"message": "Diabetic Retinopathy Prediction API is running 🚀"}

# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import custom_object_scope
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import os

# # Define your custom layer
# class CustomScaleLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(CustomScaleLayer, self).__init__(**kwargs)

#     def call(self, inputs):
#         return inputs * 255.0  # Adjust this based on what your custom layer does

#     def get_config(self):
#         return super(CustomScaleLayer, self).get_config()

# app = FastAPI()

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Model path
# MODEL_PATH = 'models/inceptionResnetV2.h5'  # Update with your actual model path

# # Load model with custom objects
# try:
#     with custom_object_scope({'CustomScaleLayer': CustomScaleLayer}):
#         model = load_model(MODEL_PATH, compile=False)
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Error loading model: {e}")
#     model = None

# # Class names for diabetic retinopathy
# CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

# def preprocess_image(image: Image.Image):
#     """Preprocess the image for model prediction"""
#     try:
#         # Resize to match your model's expected input
#         image = image.resize((224, 224))  # Adjust based on your model
        
#         # Convert to RGB if needed
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         # Convert to numpy array and normalize
#         image_array = np.array(image) / 255.0
        
#         # Add batch dimension
#         image_array = np.expand_dims(image_array, axis=0)
        
#         return image_array
#     except Exception as e:
#         raise ValueError(f"Image preprocessing failed: {e}")

# @app.get("/")
# async def root():
#     return {"message": "RetinaScan API is running!"}

# @app.get("/api/health")
# async def health_check():
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded")
#     return {"status": "healthy", "message": "API and model are ready"}

# @app.post("/api/predict")
# async def predict(file: UploadFile = File(...)):
#     if model is None:
#         raise HTTPException(status_code=503, detail="Model not loaded. Please check server logs.")
    
#     # Validate file type
#     if not file.content_type.startswith('image/'):
#         raise HTTPException(status_code=400, detail="File must be an image")
    
#     try:
#         # Read image file
#         contents = await file.read()
#         image = Image.open(io.BytesIO(contents))
        
#         # Preprocess image
#         processed_image = preprocess_image(image)
        
#         # Make prediction
#         predictions = model.predict(processed_image)
#         predicted_class = np.argmax(predictions[0])
#         confidence = float(np.max(predictions[0]))
        
#         # Get results
#         result = {
#             "success": True,
#             "prediction": {
#                 "class": CLASS_NAMES[predicted_class],
#                 "confidence": confidence,
#                 "severity": int(predicted_class),
#                 "all_predictions": predictions[0].tolist()
#             },
#             "message": f"Analysis complete: {CLASS_NAMES[predicted_class]} with {confidence:.2%} confidence"
#         }
        
#         return result
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
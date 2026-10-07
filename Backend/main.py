from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
class_names = [
    "acne",
    "blackheades",
    "dark spots",
    "pigmentation",
    "pores",
    "redness",
    "wrinkles"
]
lst=["image/png","image/jpeg","image/jpg"]
app=FastAPI(
    title="DERMAAI API",
    description="Skin conditiom classification API",
    version="1.0"
)

model=tf.keras.models.load_model(r"C:\Users\PARMOD\learn_python\skin_analyzer\dermaai_efficientnetb0.keras",
                                 compile=False,custom_objects={
                                     "preprocess_input":preprocess_input
                                 })
@app.get("/")
def home():
    return{
        "message":"DermaAI API is running"
    }

@app.get("/health")
def health():
    return{
        "status":"healthy",
        "model_loaded": True
    }
@app.get("/model-info")
def model_info():
    return{
    "model": "MobileNetV2",
    "classes": 7,
    "input_size": "224x224",
    "status": "loaded"
    }
@app.post("/predict")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type not in lst:
        raise HTTPException(
            status_code=400,
            detail="Only jpg and png images are allowed "
        )
    image=Image.open(file.file)
    image=image.convert("RGB")
    image=image.resize((224,224))
    image=np.array(image)
    image=np.expand_dims(image,axis=0)
    predictions=model.predict(image)
    prediction_index=np.argsort(predictions[0])[::-1]
    prediction_class=[class_names[i] for i in prediction_index][:3]
    confidence=[float(predictions[0][i]) for i in prediction_index][:3]
    return{
        "predictions":prediction_class,
        "confidence":confidence
    }
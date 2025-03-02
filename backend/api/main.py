from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2 
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],  
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get("/")
def health_check():
    return {"status": "OK"}


@app.post("/api/recognize")
async def recognize_letters(image: UploadFile = File(..., media_type="image/*")):
    pass
    


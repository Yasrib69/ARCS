import base64

import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI
from pydantic import BaseModel
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

app = FastAPI()

options = vision.FaceDetectorOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path="blaze_face_short_range.tflite"
    ),
    min_detection_confidence=0.5,
)
detector = vision.FaceDetector.create_from_options(options)


class Frame(BaseModel):
    image: str  # base64-encoded JPEG


@app.post("/detect")
async def detect(frame: Frame):
    data = base64.b64decode(frame.image)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"faces": []}

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = detector.detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    print(f"image {w}x{h}, faces found: {len(result.detections)}")

    faces = []
    for d in result.detections:
        b = d.bounding_box
        faces.append({
            "x": b.origin_x / w,
            "y": b.origin_y / h,
            "width": b.width / w,
            "height": b.height / h,
        })
    return {"faces": faces}
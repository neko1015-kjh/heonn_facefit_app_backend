# FaceFit 백엔드 서버 (FastAPI)
# 모바일 앱(heonn_facefit_app_mobile)과 통신하는 Python 서버입니다.

import os

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 앱(서버)을 만듭니다.
app = FastAPI(title="FaceFit API")

# CORS 설정입니다.
# 웹 브라우저 미리보기(localhost:8081)에서 이 서버(localhost:8000)로
# 요청을 보낼 수 있도록 허용해 줍니다. (이 설정이 없으면 브라우저가 연결을 막습니다.)
# 개발 단계에서는 모든 주소를 허용하고, 나중에 실제 배포할 때 좁히면 됩니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 모든 주소 허용 (개발용)
    allow_methods=["*"],
    allow_headers=["*"],
)

# 얼굴 랜드마크(특징점) 검출기를 준비합니다.
# MediaPipe의 새로운 방식(Tasks API)을 사용하며, 모델 파일이 필요합니다.
# 모델 파일(face_landmarker.task)은 아래 주소에서 한 번 내려받아 이 폴더에 둡니다.
#   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# 사진 한 장에서 얼굴 점 478개를 찾아냅니다. (기획서의 "68개"보다 훨씬 촘촘합니다.)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_landmarker_options = vision.FaceLandmarkerOptions(
    base_options=_base_options,
    num_faces=1,  # 얼굴 1명만 검출
)
# 검출기는 서버가 켜질 때 한 번만 만들어 재사용합니다(빠른 응답을 위해).
face_landmarker = vision.FaceLandmarker.create_from_options(_landmarker_options)


# 기본 주소("/")로 들어오는 요청을 처리합니다.
# 앱의 "연결 확인" 기능이 바로 이 주소로 신호를 보냅니다.
@app.get("/")
def read_root():
    return {"message": "FaceFit 백엔드 서버가 정상 동작 중입니다."}


# 서버 상태를 확인하는 별도 주소입니다. ("/health")
@app.get("/health")
def health_check():
    return {"status": "ok"}


# [실제 AI 기능 1단계] 얼굴 랜드마크 검출 주소입니다. ("/scan/landmarks")
# 앱에서 얼굴 사진을 보내면, 얼굴 특징점을 찾아 좌표를 돌려줍니다.
@app.post("/scan/landmarks")
async def detect_landmarks(file: UploadFile = File(...)):
    # 1) 업로드된 사진 파일을 읽어 이미지로 변환합니다.
    raw = await file.read()
    image_array = np.frombuffer(raw, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # 사진을 읽지 못한 경우(잘못된 파일 등)
    if image is None:
        return {
            "detected": False,
            "message": "이미지를 읽을 수 없습니다. 올바른 사진 파일인지 확인해 주세요.",
        }

    height, width = image.shape[:2]
    # MediaPipe는 RGB 순서의 이미지를 사용하므로 색상 순서를 바꿔줍니다.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 2) 얼굴 랜드마크를 검출합니다.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = face_landmarker.detect(mp_image)

    # 얼굴을 찾지 못한 경우
    if not result.face_landmarks:
        return {
            "detected": False,
            "message": "사진에서 얼굴을 찾지 못했습니다. 정면 얼굴이 잘 보이는 사진을 사용해 주세요.",
            "image_size": {"width": width, "height": height},
        }

    # 3) 찾은 점들의 좌표를 정리합니다.
    #    x, y는 0~1 사이 비율값이라, 실제 픽셀 위치는 width/height를 곱하면 됩니다.
    face = result.face_landmarks[0]
    landmarks = [
        {"x": round(p.x, 5), "y": round(p.y, 5), "z": round(p.z, 5)}
        for p in face
    ]

    # 4) 결과를 앱으로 돌려줍니다.
    return {
        "detected": True,
        "message": "얼굴 분석이 완료되었습니다.",
        "landmark_count": len(landmarks),
        "image_size": {"width": width, "height": height},
        "landmarks": landmarks,
    }

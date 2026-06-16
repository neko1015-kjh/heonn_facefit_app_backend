# FaceFit 백엔드 서버 (FastAPI)
# 모바일 앱(heonn_facefit_app_mobile)과 통신하는 Python 서버입니다.

import os
import math
import uuid
import sqlite3
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# FastAPI 앱(서버)을 만듭니다.
app = FastAPI(title="FaceFit API")

# CORS 설정입니다.
# 웹 브라우저 미리보기(localhost:8081)에서 이 서버(localhost:8000)로
# 요청을 보낼 수 있도록 허용해 줍니다. (이 설정이 없으면 브라우저가 연결을 막습니다.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 모든 주소 허용 (개발용)
    allow_methods=["*"],
    allow_headers=["*"],
)

# 이 파일이 있는 폴더 경로입니다.
BASE_DIR = os.path.dirname(__file__)

# 업로드된 얼굴 사진을 저장할 폴더입니다. (없으면 새로 만듭니다.)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
# 저장된 사진을 앱에서 볼 수 있도록 "/uploads" 주소로 공개합니다.
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 사용 이력을 저장할 데이터베이스 파일입니다. (SQLite — 파일 하나로 동작)
DB_PATH = os.path.join(BASE_DIR, "facefit.db")


def _init_db():
    """서버가 켜질 때 이력 저장용 표(table)를 준비합니다."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,      -- 분석한 시각
            symmetry INTEGER NOT NULL,     -- 안면 비대칭 점수
            balance INTEGER NOT NULL,      -- 좌우 균형(부기) 점수
            image_filename TEXT NOT NULL   -- 저장된 사진 파일 이름
        )
        """
    )
    conn.commit()
    conn.close()


_init_db()

# 얼굴 랜드마크(특징점) 검출기를 준비합니다.
# 모델 파일(face_landmarker.task)은 아래 주소에서 한 번 내려받아 이 폴더에 둡니다.
#   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
_base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
_landmarker_options = vision.FaceLandmarkerOptions(
    base_options=_base_options,
    num_faces=1,  # 얼굴 1명만 검출
)
# 검출기는 서버가 켜질 때 한 번만 만들어 재사용합니다(빠른 응답을 위해).
face_landmarker = vision.FaceLandmarker.create_from_options(_landmarker_options)


# ─────────────────────────────────────────────────────────────
# 기본 주소들
# ─────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "FaceFit 백엔드 서버가 정상 동작 중입니다."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


# [실제 AI 기능 1단계] 얼굴 랜드마크 검출 주소입니다. ("/scan/landmarks")
@app.post("/scan/landmarks")
async def detect_landmarks(file: UploadFile = File(...)):
    raw = await file.read()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"detected": False, "message": "이미지를 읽을 수 없습니다."}

    height, width = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        return {
            "detected": False,
            "message": "사진에서 얼굴을 찾지 못했습니다. 정면 얼굴이 잘 보이는 사진을 사용해 주세요.",
            "image_size": {"width": width, "height": height},
        }

    landmarks = [
        {"x": round(p.x, 5), "y": round(p.y, 5), "z": round(p.z, 5)}
        for p in result.face_landmarks[0]
    ]
    return {
        "detected": True,
        "message": "얼굴 분석이 완료되었습니다.",
        "landmark_count": len(landmarks),
        "image_size": {"width": width, "height": height},
        "landmarks": landmarks,
    }


# ─────────────────────────────────────────────────────────────
# 얼굴 점수 계산 도우미 함수들
# ─────────────────────────────────────────────────────────────
def _point(face, idx, w, h):
    """특징점 번호(idx)의 픽셀 좌표(x, y)를 돌려줍니다."""
    p = face[idx]
    return (p.x * w, p.y * h)


def _distance(a, b):
    """두 점 사이의 거리입니다."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _compute_scores(face, w, h):
    """
    얼굴 특징점들로 두 가지 점수를 계산합니다. (0~100점, 높을수록 좋음)
    - 안면 비대칭 점수: 좌우 짝이 되는 점들이 얼마나 대칭인지
    - 좌우 균형(부기) 점수: 얼굴 왼쪽/오른쪽 폭이 얼마나 비슷한지
    """
    center_ids = [10, 168, 1, 152]
    midline_x = sum(_point(face, i, w, h)[0] for i in center_ids) / len(center_ids)

    face_width = _distance(_point(face, 234, w, h), _point(face, 454, w, h))
    face_height = _distance(_point(face, 10, w, h), _point(face, 152, w, h))
    if face_width < 1 or face_height < 1:
        return None

    pairs = [(33, 263), (133, 362), (61, 291), (105, 334), (129, 358), (50, 280)]
    errors = []
    for left_id, right_id in pairs:
        lx, ly = _point(face, left_id, w, h)
        rx, ry = _point(face, right_id, w, h)
        horizontal = abs((midline_x - lx) - (rx - midline_x)) / face_width
        vertical = abs(ly - ry) / face_height
        errors.append(horizontal + vertical)
    mean_error = sum(errors) / len(errors)
    symmetry_score = round(max(0.0, min(100.0, 100.0 - mean_error * 250.0)))

    left_width = abs(midline_x - _point(face, 234, w, h)[0])
    right_width = abs(_point(face, 454, w, h)[0] - midline_x)
    avg_width = (left_width + right_width) / 2
    imbalance = abs(left_width - right_width) / avg_width if avg_width > 0 else 0
    balance_score = round(max(0.0, min(100.0, 100.0 - imbalance * 150.0)))

    return {"symmetry": symmetry_score, "balance": balance_score}


def _detect_and_score(raw):
    """사진 데이터(raw)를 받아 얼굴을 검출하고 점수를 계산합니다."""
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"detected": False, "message": "이미지를 읽을 수 없습니다."}

    height, width = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = face_landmarker.detect(mp_image)
    if not result.face_landmarks:
        return {
            "detected": False,
            "message": "사진에서 얼굴을 찾지 못했습니다. 정면 얼굴이 잘 보이는 사진을 사용해 주세요.",
        }

    scores = _compute_scores(result.face_landmarks[0], width, height)
    if scores is None:
        return {"detected": False, "message": "얼굴이 너무 작습니다. 더 가까이서 찍은 사진을 사용해 주세요."}

    return {
        "detected": True,
        "width": width,
        "height": height,
        "scores": scores,
        "landmark_count": len(result.face_landmarks[0]),
    }


def _score_list(symmetry, balance):
    """점수를 앱이 쓰기 좋은 목록 형태로 만듭니다."""
    return [
        {"key": "symmetry", "label": "안면 비대칭 개선도", "value": symmetry},
        {"key": "balance", "label": "좌우 균형 (부기)", "value": balance},
    ]


def _record_dict(rid, created_at, symmetry, balance, image_filename):
    """이력 한 건을 앱에 돌려줄 형태로 정리합니다."""
    return {
        "id": rid,
        "created_at": created_at,
        "image_url": f"/uploads/{image_filename}",
        "scores": _score_list(symmetry, balance),
    }


# [실제 AI 기능 2단계] 얼굴 점수 분석 주소입니다. ("/scan/analyze")
# 사진을 받아 점수만 계산해서 돌려줍니다(저장은 하지 않음).
@app.post("/scan/analyze")
async def analyze_face(file: UploadFile = File(...)):
    raw = await file.read()
    res = _detect_and_score(raw)
    if not res["detected"]:
        return res
    return {
        "detected": True,
        "message": "얼굴 점수 분석이 완료되었습니다.",
        "image_size": {"width": res["width"], "height": res["height"]},
        "scores": _score_list(res["scores"]["symmetry"], res["scores"]["balance"]),
    }


# ─────────────────────────────────────────────────────────────
# [실제 AI 기능 3단계] 사용 이력 저장 & 변화 추적
# ─────────────────────────────────────────────────────────────

# 사진을 분석하고 그 결과(사진 + 점수)를 이력으로 저장합니다. ("/history/scan")
@app.post("/history/scan")
async def save_scan(file: UploadFile = File(...)):
    raw = await file.read()
    res = _detect_and_score(raw)
    if not res["detected"]:
        return res  # 얼굴을 못 찾으면 저장하지 않음

    # 사진을 고유한 이름으로 저장합니다.
    filename = f"{uuid.uuid4().hex}.jpg"
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as f:
        f.write(raw)

    created_at = datetime.now().isoformat(timespec="seconds")
    symmetry = res["scores"]["symmetry"]
    balance = res["scores"]["balance"]

    # 데이터베이스에 기록을 추가합니다.
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "INSERT INTO scans (created_at, symmetry, balance, image_filename) VALUES (?, ?, ?, ?)",
        (created_at, symmetry, balance, filename),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    return {
        "detected": True,
        "message": "분석 결과를 기록에 저장했습니다.",
        "landmark_count": res["landmark_count"],
        "record": _record_dict(new_id, created_at, symmetry, balance, filename),
    }


# 저장된 모든 이력을 최신순으로 돌려줍니다. ("/history")
@app.get("/history")
def get_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, created_at, symmetry, balance, image_filename FROM scans ORDER BY id DESC"
    ).fetchall()
    conn.close()
    records = [_record_dict(*row) for row in rows]
    return {"count": len(records), "records": records}

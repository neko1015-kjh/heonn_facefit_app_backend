# FaceFit 백엔드 서버 (FastAPI)
# 모바일 앱(heonn_facefit_app_mobile)과 통신하는 Python 서버입니다.

import os
import math
import json
import time
import uuid
import urllib.request
from datetime import datetime

import db  # DB 연결 도우미 (환경에 따라 PostgreSQL/SQLite 자동 선택)

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, UploadFile, File, Header
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

# 사용 이력은 db.py를 통해 저장합니다.
# (배포 환경: PostgreSQL / 로컬 개발: SQLite 파일 — db.py가 자동 선택)


def _init_db():
    """서버가 켜질 때 이력 저장용 표(table)를 준비합니다."""
    conn = db.connect()
    # 분석 기록 표 (전체 컬럼 포함 — 새 DB에서는 이 한 번으로 완성됩니다)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS scans (
            id {db.PK},
            created_at TEXT NOT NULL,        -- 분석한 시각
            symmetry INTEGER NOT NULL,       -- 안면 비대칭 점수
            balance INTEGER NOT NULL,        -- 좌우 균형(부기) 점수
            skin_brightness INTEGER DEFAULT 0,  -- 피부 밝기(0~255)
            skin_redness INTEGER DEFAULT 0,     -- 피부 붉은기
            care_side TEXT DEFAULT '',          -- 케어가 더 필요한 쪽
            signature TEXT DEFAULT '',          -- 동일인 판별용 얼굴 서명
            user_id INTEGER DEFAULT 0,          -- 분석한 사용자(없으면 0)
            image_filename TEXT NOT NULL     -- 저장된 사진 파일 이름
        )
        """
    )
    # 예전 버전(SQLite) DB에 컬럼이 없으면 추가합니다. (PostgreSQL은 위 CREATE로 완성)
    if not db.USE_PG:
        existing = [row[1] for row in conn.execute("PRAGMA table_info(scans)").fetchall()]
        if "skin_brightness" not in existing:
            conn.execute("ALTER TABLE scans ADD COLUMN skin_brightness INTEGER DEFAULT 0")
        if "skin_redness" not in existing:
            conn.execute("ALTER TABLE scans ADD COLUMN skin_redness INTEGER DEFAULT 0")
        if "care_side" not in existing:
            conn.execute("ALTER TABLE scans ADD COLUMN care_side TEXT DEFAULT ''")
        if "signature" not in existing:
            conn.execute("ALTER TABLE scans ADD COLUMN signature TEXT DEFAULT ''")
        if "user_id" not in existing:
            conn.execute("ALTER TABLE scans ADD COLUMN user_id INTEGER DEFAULT 0")

    # 사용자 계정 표 (간단 세션 토큰 방식)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS users (
            id {db.PK},
            token TEXT UNIQUE NOT NULL,    -- 로그인 토큰(기기에 저장)
            provider TEXT,                 -- 카카오/네이버/구글 등
            display_name TEXT,             -- 표시 이름
            created_at TEXT NOT NULL
        )
        """
    )
    # 랜드마크 검출 측정 표 (성공률·처리 시간 baseline 산출용)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS detections (
            id {db.PK},
            created_at TEXT NOT NULL,      -- 검출 시각
            success INTEGER NOT NULL,      -- 1=검출 성공, 0=실패
            reason TEXT DEFAULT '',        -- 성공/실패 사유
            duration_ms INTEGER DEFAULT 0  -- 분석 처리 시간(밀리초)
        )
        """
    )
    # 예전 버전(SQLite) detections 표에 처리 시간 컬럼이 없으면 추가합니다.
    if not db.USE_PG:
        det_cols = [row[1] for row in conn.execute("PRAGMA table_info(detections)").fetchall()]
        if "duration_ms" not in det_cols:
            conn.execute("ALTER TABLE detections ADD COLUMN duration_ms INTEGER DEFAULT 0")
    # 만족도(CSAT) 측정 표
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS feedback (
            id {db.PK},
            created_at TEXT NOT NULL,      -- 평가 시각
            user_id INTEGER DEFAULT 0,     -- 평가한 사용자(없으면 0)
            satisfied INTEGER NOT NULL     -- 1=만족(도움됨), 0=불만족
        )
        """
    )
    conn.commit()
    conn.close()


_init_db()

# 얼굴 랜드마크(특징점) 검출기를 준비합니다.
# 모델 파일(face_landmarker.task)이 없으면 자동으로 내려받습니다.
# (클라우드에 배포할 때 모델 파일을 따로 올리지 않아도 되도록)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")
if not os.path.exists(MODEL_PATH):
    try:
        print("얼굴 모델 다운로드 중...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("얼굴 모델 다운로드 완료")
    except Exception as e:
        print("얼굴 모델 다운로드 실패:", e)

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


# ─────────────────────────────────────────────────────────────
# 로그인 / 사용자 (간단 토큰 세션)
# ─────────────────────────────────────────────────────────────
def _current_user_id(authorization: str | None):
    """요청 헤더의 토큰으로 현재 사용자 id를 찾습니다. 없으면 None."""
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        return None
    conn = db.connect()
    row = conn.execute("SELECT id FROM users WHERE token = ?", (token,)).fetchone()
    conn.close()
    return row[0] if row else None


# 로그인: 소셜 버튼을 누르면 사용자를 만들고 토큰을 발급합니다.
@app.post("/auth/login")
def auth_login(payload: dict):
    provider = (payload or {}).get("provider", "guest")
    name = (payload or {}).get("name") or f"{provider} 사용자"
    token = uuid.uuid4().hex
    created_at = datetime.now().isoformat(timespec="seconds")
    conn = db.connect()
    # RETURNING id 는 SQLite(3.35+)와 PostgreSQL 양쪽에서 새 id를 돌려줍니다.
    cur = conn.execute(
        "INSERT INTO users (token, provider, display_name, created_at) VALUES (?, ?, ?, ?) RETURNING id",
        (token, provider, name, created_at),
    )
    uid = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return {"token": token, "user": {"id": uid, "provider": provider, "display_name": name}}


# 저장된 토큰으로 로그인 상태를 확인합니다(자동 로그인).
@app.get("/auth/me")
def auth_me(authorization: str = Header(None)):
    if not authorization:
        return {"authenticated": False}
    token = authorization.replace("Bearer ", "").strip()
    conn = db.connect()
    row = conn.execute(
        "SELECT id, provider, display_name FROM users WHERE token = ?", (token,)
    ).fetchone()
    conn.close()
    if not row:
        return {"authenticated": False}
    return {"authenticated": True, "user": {"id": row[0], "provider": row[1], "display_name": row[2]}}


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
            "message": "얼굴 사진이 아니거나 얼굴을 찾지 못했습니다. 얼굴이 정면으로 잘 보이는 사진을 사용해 주세요.",
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

    # 좌우 끝점(234=사진 왼쪽=사용자 오른쪽, 454=사진 오른쪽=사용자 왼쪽)
    left_width = abs(midline_x - _point(face, 234, w, h)[0])
    right_width = abs(_point(face, 454, w, h)[0] - midline_x)
    avg_width = (left_width + right_width) / 2
    imbalance = abs(left_width - right_width) / avg_width if avg_width > 0 else 0
    balance_score = round(max(0.0, min(100.0, 100.0 - imbalance * 150.0)))

    # 더 넓은(부은) 쪽 = 케어가 필요한 쪽. 사용자 기준으로 표기합니다.
    care_side = "오른쪽" if left_width > right_width else "왼쪽"

    return {"symmetry": symmetry_score, "balance": balance_score, "care_side": care_side}


def _compute_skin_tone(image, face, w, h):
    """
    볼·이마 등 피부 영역의 색을 모아 피부톤을 분석합니다.
    - brightness: 피부 밝기(0~255, 높을수록 밝음)
    - redness: 붉은기(높을수록 홍조/붉은 편)
    image는 OpenCV 기준 BGR 순서입니다.
    """
    # 피부가 잘 드러나는 특징점들(양 볼, 이마 중앙, 코)
    skin_ids = [50, 280, 101, 330, 151, 1]
    patch = max(2, int(min(w, h) * 0.01))  # 사진 크기에 비례한 표본 영역
    samples = []
    for idx in skin_ids:
        x = int(face[idx].x * w)
        y = int(face[idx].y * h)
        x0, x1 = max(0, x - patch), min(w, x + patch)
        y0, y1 = max(0, y - patch), min(h, y + patch)
        region = image[y0:y1, x0:x1]
        if region.size > 0:
            samples.append(region.reshape(-1, 3).mean(axis=0))  # (B, G, R) 평균

    if not samples:
        return {"brightness": 0, "redness": 0}

    mean_bgr = np.mean(samples, axis=0)
    b, g, r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
    brightness = round((r + g + b) / 3)
    redness = round(r - (g + b) / 2)
    return {"brightness": brightness, "redness": redness}


def _validate_face(face, w, h):
    """
    검출된 얼굴이 '분석 가능한 정확한 정면 얼굴'인지 확인합니다.
    문제가 있으면 사용자에게 보여줄 안내 문구를, 정상이면 None을 돌려줍니다.
    """
    # 1) 얼굴이 사진에서 너무 작은지 (멀리서 찍었거나 얼굴이 작게 나온 경우)
    top = _point(face, 10, w, h)
    chin = _point(face, 152, w, h)
    face_height = _distance(top, chin)
    if face_height < h * 0.12:
        return "얼굴이 너무 작게 나왔어요. 얼굴이 화면에 크게 보이도록 정면에서 다시 촬영해 주세요."

    # 2) 얼굴이 많이 기울어졌는지 (양 눈을 잇는 선의 각도로 판단)
    left_eye = _point(face, 33, w, h)
    right_eye = _point(face, 263, w, h)
    angle = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    if abs(angle) > 30:
        return "얼굴이 기울어져 있어요. 정면을 향해 똑바로 촬영해 주세요."

    # 3) 얼굴 윤곽이 화면 밖으로 많이 잘렸는지 (정면·전체가 보여야 함)
    for idx in (10, 152, 234, 454):
        px, py = face[idx].x, face[idx].y
        if px < -0.03 or px > 1.03 or py < -0.03 or py > 1.03:
            return "얼굴이 화면 밖으로 잘렸어요. 얼굴 전체가 보이도록 다시 촬영해 주세요."

    return None


def _compute_signature(face, w, h):
    """
    얼굴 비율로 '얼굴 서명(특징 벡터)'을 만듭니다.
    두 눈 사이 거리로 나눠 크기에 무관한 비율값들로 구성하므로,
    같은 사람은 비슷하고 다른 사람은 차이가 큽니다. (동일인 판별용 휴리스틱)
    """
    def d(i, j):
        return _distance(_point(face, i, w, h), _point(face, j, w, h))

    iod = d(33, 263)  # 두 눈 바깥 끝 사이 거리(기준)
    if iod < 1:
        return []
    ratios = [
        d(234, 454),  # 얼굴 너비
        d(10, 152),   # 얼굴 높이
        d(129, 358),  # 코 너비
        d(61, 291),   # 입 너비
        d(133, 362),  # 양 눈 안쪽 거리
        d(105, 334),  # 눈썹 사이
        d(168, 1),    # 코 길이
        d(1, 152),    # 코끝~턱
        d(105, 33),   # 눈썹~눈
    ]
    return [round(x / iod, 4) for x in ratios]


def _signature_distance(a, b):
    """두 얼굴 서명 사이의 거리(작을수록 같은 사람일 가능성↑). 비교 불가 시 None."""
    if not a or not b or len(a) != len(b):
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _log_detection(success, reason="", duration_ms=0):
    """얼굴 랜드마크 검출 시도 한 건을 측정용으로 기록합니다(성공률·처리 시간 baseline 산출)."""
    try:
        conn = db.connect()
        conn.execute(
            "INSERT INTO detections (created_at, success, reason, duration_ms) VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), 1 if success else 0, reason, int(duration_ms)),
        )
        conn.commit()
        conn.close()
    except Exception:
        # 측정 기록 실패는 분석 자체를 막지 않도록 조용히 넘어갑니다.
        pass


def _detect_and_score(raw):
    """사진 데이터(raw)를 받아 얼굴을 검출하고 점수·피부톤을 계산합니다."""
    # 처리 시간 측정 시작 (사진 1장 분석에 걸리는 시간 = fps 대체 지표)
    t0 = time.perf_counter()
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        # 이미지 자체를 못 읽은 경우는 '검출 대상'이 아니므로 측정에서 제외합니다.
        return {"detected": False, "message": "이미지를 읽을 수 없습니다. 올바른 사진 파일인지 확인해 주세요."}

    height, width = image.shape[:2]
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    result = face_landmarker.detect(mp_image)
    elapsed_ms = (time.perf_counter() - t0) * 1000  # 검출까지 걸린 시간(ms)
    if not result.face_landmarks:
        # 얼굴 랜드마크를 찾지 못함 → 검출 실패로 기록
        _log_detection(False, "얼굴 미검출")
        return {
            "detected": False,
            "message": "얼굴 사진이 아니거나 얼굴을 찾지 못했습니다. 얼굴이 정면으로 잘 보이는 사진을 사용해 주세요.",
        }

    # 얼굴 랜드마크 검출 성공 → 성공 + 처리 시간 기록 (품질 검사와 무관하게 '검출'은 성공)
    _log_detection(True, "검출 성공", elapsed_ms)
    face = result.face_landmarks[0]

    # 정확한 정면 얼굴인지 품질 검사
    problem = _validate_face(face, width, height)
    if problem:
        return {"detected": False, "message": problem}

    scores = _compute_scores(face, width, height)
    if scores is None:
        return {"detected": False, "message": "얼굴이 정확히 인식되지 않았어요. 정면 얼굴이 잘 보이는 사진으로 다시 시도해 주세요."}

    skin = _compute_skin_tone(image, face, width, height)
    signature = _compute_signature(face, width, height)
    # 화면 표시용 좌표. x, y는 0~1 비율, z는 상대 깊이(간이 3D 표시에 사용).
    landmarks = [{"x": round(p.x, 4), "y": round(p.y, 4), "z": round(p.z, 4)} for p in face]
    return {
        "detected": True,
        "width": width,
        "height": height,
        "scores": scores,
        "skin": skin,
        "signature": signature,
        "landmark_count": len(face),
        "landmarks": landmarks,
    }


def _score_list(symmetry, balance):
    """점수를 앱이 쓰기 좋은 목록 형태로 만듭니다."""
    return [
        {"key": "symmetry", "label": "안면 비대칭 개선도", "value": symmetry},
        {"key": "balance", "label": "좌우 균형 (부기)", "value": balance},
    ]


def _record_dict(rid, created_at, symmetry, balance, image_filename, care_side="", signature_json=""):
    """이력 한 건을 앱에 돌려줄 형태로 정리합니다."""
    try:
        signature = json.loads(signature_json) if signature_json else []
    except (ValueError, TypeError):
        signature = []
    return {
        "id": rid,
        "created_at": created_at,
        "image_url": f"/uploads/{image_filename}",
        "scores": _score_list(symmetry, balance),
        "care_side": care_side,
        "signature": signature,
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
async def save_scan(file: UploadFile = File(...), authorization: str = Header(None)):
    user_id = _current_user_id(authorization)
    if not user_id:
        return {"detected": False, "message": "로그인이 필요합니다."}
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
    care_side = res["scores"]["care_side"]
    brightness = res["skin"]["brightness"]
    redness = res["skin"]["redness"]
    signature_json = json.dumps(res["signature"])

    # 데이터베이스에 기록을 추가합니다. (로그인한 사용자에 귀속)
    conn = db.connect()
    cur = conn.execute(
        """
        INSERT INTO scans (created_at, symmetry, balance, skin_brightness, skin_redness, care_side, signature, image_filename, user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
        """,
        (created_at, symmetry, balance, brightness, redness, care_side, signature_json, filename, user_id),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    conn.close()

    return {
        "detected": True,
        "message": "분석 결과를 기록에 저장했습니다.",
        "landmark_count": res["landmark_count"],
        "image_size": {"width": res["width"], "height": res["height"]},
        "landmarks": res["landmarks"],
        "record": _record_dict(new_id, created_at, symmetry, balance, filename, care_side, signature_json),
    }


# 저장된 이력을 최신순으로 돌려줍니다. (로그인한 사용자의 기록만)
@app.get("/history")
def get_history(authorization: str = Header(None)):
    user_id = _current_user_id(authorization)
    if not user_id:
        return {"count": 0, "records": []}
    conn = db.connect()
    rows = conn.execute(
        """
        SELECT id, created_at, symmetry, balance, image_filename, care_side, signature
        FROM scans WHERE user_id = ? ORDER BY id DESC
        """,
        (user_id,),
    ).fetchall()
    conn.close()
    records = [_record_dict(*row) for row in rows]
    return {"count": len(records), "records": records}


# ─────────────────────────────────────────────────────────────
# [실제 AI 기능 4단계] 부기 + 피부톤 기반 맞춤 제품 추천
# ─────────────────────────────────────────────────────────────

# 추천 후보 제품 목록입니다. (프로토타입용)
# 제품별 대표 이미지(온라인 화장품 사진)도 함께 둡니다.
_PRODUCT_CATALOG = {
    "cooling": {
        "name": "HeOnn 쿨링 디톡스 앰플",
        "desc": "부기 완화·림프 케어",
        "image": "https://images.unsplash.com/photo-1620916566398-39f1143ab7be?auto=format&fit=crop&w=400&q=70",
    },
    "soothing": {
        "name": "HeOnn 시카 진정 크림",
        "desc": "붉은기·민감 진정",
        "image": "https://images.unsplash.com/photo-1556228578-8c89e6adf883?auto=format&fit=crop&w=400&q=70",
    },
    "brightening": {
        "name": "HeOnn 비타민C 세럼",
        "desc": "칙칙한 톤 보정·브라이트닝",
        "image": "https://images.unsplash.com/photo-1608248543803-ba4f8c70ae0b?auto=format&fit=crop&w=400&q=70",
    },
    "moisture": {
        "name": "HeOnn 딥 모이스처 크림",
        "desc": "기본 보습·장벽 강화",
        "image": "https://images.unsplash.com/photo-1556228720-195a672e8a03?auto=format&fit=crop&w=400&q=70",
    },
    "lifting": {
        "name": "HeOnn 탄력 리프팅 크림",
        "desc": "탄력·비대칭 케어",
        "image": "https://images.unsplash.com/photo-1598440947619-2c35fc9aa908?auto=format&fit=crop&w=400&q=70",
    },
}


def _build_recommendations(symmetry, balance, brightness, redness):
    """부기(balance)·피부톤(밝기/붉은기)·비대칭(symmetry)에 맞춰 제품을 고릅니다.
    분석에 맞는 제품을 앞쪽에 두고, 최대 5개까지 채워서 돌려줍니다."""
    items = []
    used = set()

    def add(key, reason):
        if key not in used:
            items.append({**_PRODUCT_CATALOG[key], "reason": reason})
            used.add(key)

    # 1) 분석 결과에 맞는 맞춤 추천 (이유 포함)
    if balance < 85:
        add("cooling", f"좌우 균형(부기) {balance}점 — 부기 완화 케어가 필요해요")
    if redness >= 30:
        add("soothing", "피부에 붉은기가 있어 진정 케어를 추천해요")
    if brightness < 130:
        add("brightening", "피부 톤이 다소 어두워 톤 보정을 추천해요")
    if symmetry < 85:
        add("lifting", f"안면 비대칭 {symmetry}점 — 탄력 리프팅 케어")

    # 2) 5개가 될 때까지 나머지 제품으로 채웁니다.
    for key in ["cooling", "soothing", "brightening", "moisture", "lifting"]:
        if len(items) >= 5:
            break
        add(key, "피부 컨디션 관리에 함께 추천해요")

    return items[:5]


def _tone_labels(brightness, redness):
    """피부톤을 사람이 읽기 쉬운 말로 바꿉니다."""
    if brightness >= 170:
        tone = "밝은 톤"
    elif brightness >= 130:
        tone = "중간 톤"
    else:
        tone = "어두운 톤"
    if redness >= 30:
        red = "붉은기 있음"
    elif redness >= 15:
        red = "약간 붉은기"
    else:
        red = "차분한 톤"
    return tone, red


# 가장 최근 분석 결과를 바탕으로 맞춤 추천을 돌려줍니다. (로그인 사용자 기준)
@app.get("/recommendations")
def get_recommendations(authorization: str = Header(None)):
    user_id = _current_user_id(authorization)
    row = None
    if user_id:
        conn = db.connect()
        row = conn.execute(
            """
            SELECT symmetry, balance, skin_brightness, skin_redness
            FROM scans WHERE user_id = ? ORDER BY id DESC LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        conn.close()

    # 아직 분석 기록이 없으면 기본 추천(전체 제품)을 돌려줍니다.
    if row is None:
        return {
            "has_record": False,
            "message": "AI스캔에서 얼굴을 분석하면 맞춤 추천을 받을 수 있어요.",
            "products": [
                {**_PRODUCT_CATALOG[key], "reason": "인기 추천 제품"}
                for key in ["moisture", "cooling", "soothing", "brightening", "lifting"]
            ],
        }

    symmetry, balance, brightness, redness = row
    tone, red = _tone_labels(brightness, redness)
    return {
        "has_record": True,
        "summary": {
            "balance": balance,
            "symmetry": symmetry,
            "skin_tone": tone,
            "skin_redness": red,
        },
        "products": _build_recommendations(symmetry, balance, brightness, redness),
    }


# ─────────────────────────────────────────────────────────────
# [측정 지표] 랜드마크 검출 성공률 (PoC 검증 표의 baseline 산출용)
# 지금까지 분석을 시도한 사진들 중 얼굴 검출에 성공한 비율(%)을 돌려줍니다.
# ─────────────────────────────────────────────────────────────
@app.get("/metrics/landmark")
def landmark_metrics():
    conn = db.connect()
    total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    success = conn.execute("SELECT COUNT(*) FROM detections WHERE success = 1").fetchone()[0]
    conn.close()
    # 측정 기록이 하나도 없으면 0%로 표시합니다(아직 분석한 사진이 없음).
    success_rate = round(success / total * 100, 1) if total else 0.0
    return {
        "metric": "landmark_detection_success_rate",
        "total_attempts": total,   # 분석 시도(검출 대상) 총 횟수
        "success_count": success,  # 얼굴 검출 성공 횟수
        "success_rate": success_rate,  # 검출 성공률(%)
    }


# ─────────────────────────────────────────────────────────────
# [측정 지표] 재사용률 (PoC 검증 표 ❸의 baseline 산출용)
# 분석을 1회 이상 한 로그인 사용자 중, 2회 이상 분석한 사용자의 비율(%)입니다.
# (다시 돌아와 사용한 사람의 비율 = 리텐션의 가장 간단한 측정)
# ─────────────────────────────────────────────────────────────
@app.get("/metrics/retention")
def retention_metrics():
    conn = db.connect()
    # 로그인 사용자(user_id > 0)별 분석 횟수를 셉니다. (레거시 0번 기록은 제외)
    rows = conn.execute(
        "SELECT user_id, COUNT(*) FROM scans WHERE user_id > 0 GROUP BY user_id"
    ).fetchall()
    conn.close()
    active_users = len(rows)  # 분석을 1회 이상 한 사용자 수
    returning_users = sum(1 for _, cnt in rows if cnt >= 2)  # 2회 이상 분석한 사용자 수
    # 활성 사용자가 없으면 0%로 표시합니다.
    reuse_rate = round(returning_users / active_users * 100, 1) if active_users else 0.0
    return {
        "metric": "reuse_rate",
        "active_users": active_users,        # 분석 1회 이상 사용자 수
        "returning_users": returning_users,  # 2회 이상 분석 사용자 수
        "reuse_rate": reuse_rate,            # 재사용률(%)
    }


# 만족도 평가 제출: 사용자가 분석/케어가 도움이 됐는지 평가합니다.
@app.post("/feedback")
def submit_feedback(payload: dict, authorization: str = Header(None)):
    user_id = _current_user_id(authorization) or 0
    satisfied = 1 if (payload or {}).get("satisfied") else 0
    conn = db.connect()
    conn.execute(
        "INSERT INTO feedback (created_at, user_id, satisfied) VALUES (?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), user_id, satisfied),
    )
    conn.commit()
    conn.close()
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# [측정 지표] 만족도(CSAT) (PoC 검증 표 ❸의 baseline 산출용)
# 만족도 평가 중 '만족(도움됨)'으로 응답한 비율(%)입니다.
# ─────────────────────────────────────────────────────────────
@app.get("/metrics/csat")
def csat_metrics():
    conn = db.connect()
    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    satisfied = conn.execute("SELECT COUNT(*) FROM feedback WHERE satisfied = 1").fetchone()[0]
    conn.close()
    # 평가가 하나도 없으면 0%로 표시합니다.
    csat = round(satisfied / total * 100, 1) if total else 0.0
    return {
        "metric": "csat",
        "total_responses": total,      # 전체 평가 수
        "satisfied_count": satisfied,  # 만족 응답 수
        "csat": csat,                  # 만족도(%)
    }


# ─────────────────────────────────────────────────────────────
# [측정 지표] 분석 처리 시간 (PoC 검증 표 ❶의 fps 대체 baseline)
# 검출에 성공한 분석들의 평균 처리 시간(ms)입니다. (사진 1장당 소요 시간)
# ─────────────────────────────────────────────────────────────
@app.get("/metrics/latency")
def latency_metrics():
    conn = db.connect()
    row = conn.execute(
        "SELECT COUNT(*), AVG(duration_ms) FROM detections WHERE success = 1 AND duration_ms > 0"
    ).fetchone()
    conn.close()
    count = row[0] or 0
    # PostgreSQL의 AVG는 Decimal을 돌려주므로 float으로 변환해 통일합니다.
    avg_ms = round(float(row[1]), 1) if row[1] else 0.0
    # 처리 시간으로 환산한 초당 처리 장수(참고용). 시간이 0이면 0으로 둡니다.
    approx_fps = round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0
    return {
        "metric": "analysis_latency",
        "sample_count": count,     # 측정에 쓰인 분석 건수
        "avg_duration_ms": avg_ms, # 평균 처리 시간(ms)
        "approx_fps": approx_fps,  # 환산 처리량(장/초, 참고용)
    }

# FaceFit 백엔드 서버 (FastAPI)
# 모바일 앱(heonn_facefit_app_mobile)과 통신하는 Python 서버입니다.

import os
import math
import json
import time
import uuid
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime

import db  # DB 연결 도우미 (환경에 따라 PostgreSQL/SQLite 자동 선택)

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response, HTMLResponse

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

# 분석 사진은 데이터베이스(scans.image_data)에 저장하고 "/uploads/{파일명}" 주소로 돌려줍니다.
# (예전에는 파일 폴더 + StaticFiles를 썼지만, 서버 재시작 시 사진이 사라져서 DB 저장으로 변경)

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
            gender TEXT DEFAULT '',             -- 추정 성별(동일인 판별 보조)
            age TEXT DEFAULT '',                -- 추정 나이대(참고용)
            dark_circle INTEGER DEFAULT 0,      -- 다크서클 점수(높을수록 양호)
            wrinkle INTEGER DEFAULT 0,          -- 주름 점수(높을수록 양호)
            user_id INTEGER DEFAULT 0,          -- 분석한 사용자(없으면 0)
            image_filename TEXT NOT NULL,       -- 사진 식별용 이름
            image_data {db.BLOB}                -- 사진 바이너리 (영구 저장)
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
        if "gender" not in existing:
            conn.execute("ALTER TABLE scans ADD COLUMN gender TEXT DEFAULT ''")
        if "age" not in existing:
            conn.execute("ALTER TABLE scans ADD COLUMN age TEXT DEFAULT ''")
        if "dark_circle" not in existing:
            conn.execute("ALTER TABLE scans ADD COLUMN dark_circle INTEGER DEFAULT 0")
        if "wrinkle" not in existing:
            conn.execute("ALTER TABLE scans ADD COLUMN wrinkle INTEGER DEFAULT 0")
        if "user_id" not in existing:
            conn.execute("ALTER TABLE scans ADD COLUMN user_id INTEGER DEFAULT 0")
        if "image_data" not in existing:
            conn.execute(f"ALTER TABLE scans ADD COLUMN image_data {db.BLOB}")

    # 사용자 계정 표 (간단 세션 토큰 방식)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS users (
            id {db.PK},
            token TEXT UNIQUE NOT NULL,    -- 로그인 토큰(기기에 저장)
            provider TEXT,                 -- 카카오/네이버/구글 등
            provider_id TEXT DEFAULT '',   -- 소셜 계정 고유 id(같은 사람 식별용)
            display_name TEXT,             -- 표시 이름
            created_at TEXT NOT NULL,
            consent_at TEXT DEFAULT '',    -- 약관·개인정보(얼굴 포함) 동의 시각(없으면 미동의)
            marketing INTEGER DEFAULT 0    -- 마케팅 수신 동의(선택, 1=동의)
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
    # PostgreSQL: 기존 테이블에 나중에 추가된 컬럼이 없을 수 있으니 보강합니다.
    # (PostgreSQL은 ADD COLUMN IF NOT EXISTS 를 지원하므로 안전하게 추가됩니다)
    if db.USE_PG:
        conn.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS gender TEXT DEFAULT ''")
        conn.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS age TEXT DEFAULT ''")
        conn.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS dark_circle INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS wrinkle INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS care_side TEXT DEFAULT ''")
        conn.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS signature TEXT DEFAULT ''")
        conn.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS user_id INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE detections ADD COLUMN IF NOT EXISTS duration_ms INTEGER DEFAULT 0")
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS provider_id TEXT DEFAULT ''")
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS consent_at TEXT DEFAULT ''")
        conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS marketing INTEGER DEFAULT 0")
        conn.execute(f"ALTER TABLE scans ADD COLUMN IF NOT EXISTS image_data {db.BLOB}")
    else:
        ucols = [row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()]
        if "provider_id" not in ucols:
            conn.execute("ALTER TABLE users ADD COLUMN provider_id TEXT DEFAULT ''")
        if "consent_at" not in ucols:
            conn.execute("ALTER TABLE users ADD COLUMN consent_at TEXT DEFAULT ''")
        if "marketing" not in ucols:
            conn.execute("ALTER TABLE users ADD COLUMN marketing INTEGER DEFAULT 0")
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
    output_facial_transformation_matrixes=True,  # 머리 자세(각도) 계산용 행렬 출력
)
# 검출기는 서버가 켜질 때 한 번만 만들어 재사용합니다(빠른 응답을 위해).
face_landmarker = vision.FaceLandmarker.create_from_options(_landmarker_options)

# 성별 추정 모델(OpenCV용 Caffe)도 준비합니다. 없으면 자동으로 내려받습니다.
# (동일인 판별의 보조 신호로 사용 — 성별 추정은 100% 정확하지 않음에 유의)
GENDER_PROTO_URL = "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_deploy.prototxt"
GENDER_MODEL_URL = "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel"
GENDER_PROTO_PATH = os.path.join(BASE_DIR, "gender_deploy.prototxt")
GENDER_MODEL_PATH = os.path.join(BASE_DIR, "gender_net.caffemodel")
GENDER_LIST = ["Male", "Female"]  # 모델 출력 순서
GENDER_MEAN = (78.4263377603, 87.7689143744, 114.895847746)  # 모델 학습 시 평균값

gender_net = None
try:
    if not os.path.exists(GENDER_PROTO_PATH):
        print("성별 모델(구조) 다운로드 중...")
        urllib.request.urlretrieve(GENDER_PROTO_URL, GENDER_PROTO_PATH)
    if not os.path.exists(GENDER_MODEL_PATH):
        print("성별 모델(가중치) 다운로드 중...")
        urllib.request.urlretrieve(GENDER_MODEL_URL, GENDER_MODEL_PATH)
    gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO_PATH, GENDER_MODEL_PATH)
    print("성별 모델 준비 완료")
except Exception as e:
    # 성별 모델 로드 실패해도 서버는 정상 동작(성별 추정만 비활성).
    print("성별 모델 준비 실패(성별 추정 비활성):", e)
    gender_net = None

# 나이 추정 모델(OpenCV용 Caffe) — 성별 모델과 같은 곳/같은 방식. 8개 나이대 구간을 출력합니다.
# (참고용 추정으로만 사용 — 정확한 실제 나이가 아님)
AGE_PROTO_URL = "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_deploy.prototxt"
AGE_MODEL_URL = "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel"
AGE_PROTO_PATH = os.path.join(BASE_DIR, "age_deploy.prototxt")
AGE_MODEL_PATH = os.path.join(BASE_DIR, "age_net.caffemodel")
AGE_LIST = ["0-2세", "4-6세", "8-12세", "15-20세", "25-32세", "38-43세", "48-53세", "60세 이상"]

age_net = None
try:
    if not os.path.exists(AGE_PROTO_PATH):
        print("나이 모델(구조) 다운로드 중...")
        urllib.request.urlretrieve(AGE_PROTO_URL, AGE_PROTO_PATH)
    if not os.path.exists(AGE_MODEL_PATH):
        print("나이 모델(가중치) 다운로드 중...")
        urllib.request.urlretrieve(AGE_MODEL_URL, AGE_MODEL_PATH)
    age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO_PATH, AGE_MODEL_PATH)
    print("나이 모델 준비 완료")
except Exception as e:
    # 나이 모델 로드 실패해도 서버는 정상 동작(나이 추정만 비활성).
    print("나이 모델 준비 실패(나이 추정 비활성):", e)
    age_net = None


def _estimate_gender(image, face, w, h):
    """얼굴 영역을 잘라 성별('Male'/'Female')을 추정합니다. 실패 시 ''."""
    if gender_net is None:
        return ""
    xs = [p.x for p in face]
    ys = [p.y for p in face]
    x1 = max(0, int(min(xs) * w))
    x2 = min(w, int(max(xs) * w))
    y1 = max(0, int(min(ys) * h))
    y2 = min(h, int(max(ys) * h))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    try:
        blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227), GENDER_MEAN, swapRB=False)
        gender_net.setInput(blob)
        pred = gender_net.forward()
        return GENDER_LIST[int(pred[0].argmax())]
    except Exception:
        return ""


def _estimate_age(image, face, w, h):
    """얼굴 영역을 잘라 나이대('25-32세' 등)를 추정합니다. 실패 시 ''. (참고용 추정)"""
    if age_net is None:
        return ""
    xs = [p.x for p in face]
    ys = [p.y for p in face]
    x1 = max(0, int(min(xs) * w))
    x2 = min(w, int(max(xs) * w))
    y1 = max(0, int(min(ys) * h))
    y2 = min(h, int(max(ys) * h))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return ""
    try:
        blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227), GENDER_MEAN, swapRB=False)
        age_net.setInput(blob)
        pred = age_net.forward()
        return AGE_LIST[int(pred[0].argmax())]
    except Exception:
        return ""


def _estimate_head_pose(result):
    """
    검출 결과에서 머리 자세를 도(°) 단위로 추정합니다.
    - yaw: 좌우로 돌아간 정도 / pitch: 위아래로 돌아간 정도 / roll: 갸웃 기울인 정도
    값이 0에 가까울수록 '정면'입니다. 행렬 정보가 없으면 None을 돌려줍니다.
    (정면이 아닐수록 좌우 폭이 왜곡돼 비대칭·균형 점수가 부정확해지므로 사전 판단에 사용)
    """
    mats = getattr(result, "facial_transformation_matrixes", None)
    if not mats:
        return None
    R = np.array(mats[0])[:3, :3].astype(float)
    # 열 단위 스케일 제거 → 순수 회전 행렬로 정리
    for c in range(3):
        n = np.linalg.norm(R[:, c])
        if n > 0:
            R[:, c] /= n
    try:
        ang = cv2.RQDecomp3x3(R)[0]  # (pitch, yaw, roll) 근사값(도)
    except Exception:
        return None
    return {
        "pitch": round(float(ang[0]), 1),
        "yaw": round(float(ang[1]), 1),
        "roll": round(float(ang[2]), 1),
    }


# 정면으로 인정하는 머리 각도 한계(도). 이보다 많이 돌아가면 정확한 측정이 어려워 다시 촬영을 안내합니다.
# (실제 셀카 검증 결과 정면은 yaw·pitch가 대체로 ±10° 이내였음 → 여유를 둬 설정)
MAX_YAW = 22.0    # 좌우 회전 한계
MAX_PITCH = 25.0  # 상하 회전 한계(폰을 내려다보는 경향 고려해 약간 넉넉히)


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
    # 새 계정은 아직 동의 전(consented=False)
    return {"token": token, "user": {"id": uid, "provider": provider, "display_name": name, "consented": False}}


# 약관·개인정보(얼굴 포함) 동의를 기록합니다. (로그인 후 동의 화면에서 호출)
@app.post("/auth/consent")
def auth_consent(payload: dict, authorization: str = Header(None)):
    user_id = _current_user_id(authorization)
    if not user_id:
        return {"ok": False, "message": "로그인이 필요합니다."}
    marketing = 1 if (payload or {}).get("marketing") else 0
    consent_at = datetime.now().isoformat(timespec="seconds")
    conn = db.connect()
    try:
        conn.execute(
            "UPDATE users SET consent_at = ?, marketing = ? WHERE id = ?",
            (consent_at, marketing, user_id),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "consent_at": consent_at, "marketing": bool(marketing)}


# ─────────────────────────────────────────────────────────────
# 카카오 로그인 (실제 OAuth)
# 흐름: 프론트 → /auth/kakao/login → 카카오 인증 → /auth/kakao/callback → 웹으로 토큰 전달
# ─────────────────────────────────────────────────────────────
KAKAO_REST_KEY = os.environ.get("KAKAO_REST_KEY", "")
KAKAO_CLIENT_SECRET = os.environ.get("KAKAO_CLIENT_SECRET", "")  # 카카오 보안 설정이 ON일 때 필요
KAKAO_REDIRECT_URI = "https://neko1015-facefit-backend.hf.space/auth/kakao/callback"
WEB_URL = "https://neko1015-heonn-web.static.hf.space"

# 구글 로그인 설정 (Google Cloud Console에서 발급)
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = "https://neko1015-facefit-backend.hf.space/auth/google/callback"

# 휴대폰 앱(네이티브)으로 돌아갈 때 쓰는 앱 주소(딥링크). app.json의 scheme과 같아야 합니다.
APP_NATIVE_REDIRECT = "facefit://auth"


def _oauth_redirect(state, token=None):
    """로그인 후 돌아갈 주소를 만듭니다. state='native'면 앱으로, 아니면 웹으로 보냅니다."""
    base = APP_NATIVE_REDIRECT if state == "native" else f"{WEB_URL}/"
    if token:
        return f"{base}?token={token}"
    return f"{base}?login_error=1"


@app.get("/auth/kakao/login")
def kakao_login(state: str = "web"):
    """카카오 인증 페이지로 보냅니다. state로 웹/앱(native) 복귀 대상을 구분합니다."""
    params = urllib.parse.urlencode({
        "client_id": KAKAO_REST_KEY,
        "redirect_uri": KAKAO_REDIRECT_URI,
        "response_type": "code",
        "state": state,  # 콜백까지 그대로 전달됨(웹/네이티브 구분용)
    })
    return RedirectResponse(f"https://kauth.kakao.com/oauth/authorize?{params}")


@app.get("/auth/kakao/callback")
def kakao_callback(code: str = "", state: str = "web"):
    """카카오가 보내준 code로 사용자 정보를 받아 우리 계정을 만들고, 웹/앱으로 토큰을 전달합니다."""
    if not code:
        return RedirectResponse(_oauth_redirect(state))
    try:
        # 1) code → 카카오 access_token
        token_params = {
            "grant_type": "authorization_code",
            "client_id": KAKAO_REST_KEY,
            "redirect_uri": KAKAO_REDIRECT_URI,
            "code": code,
        }
        # 카카오 보안(Client Secret) 설정이 켜져 있으면 시크릿을 함께 보냅니다.
        if KAKAO_CLIENT_SECRET:
            token_params["client_secret"] = KAKAO_CLIENT_SECRET
        token_body = urllib.parse.urlencode(token_params).encode()
        req = urllib.request.Request("https://kauth.kakao.com/oauth/token", data=token_body)
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=10) as r:
            access_token = json.loads(r.read())["access_token"]

        # 2) access_token → 카카오 사용자 정보
        me_req = urllib.request.Request("https://kapi.kakao.com/v2/user/me")
        me_req.add_header("Authorization", f"Bearer {access_token}")
        with urllib.request.urlopen(me_req, timeout=10) as r:
            me = json.loads(r.read())
        kakao_id = str(me.get("id", ""))
        try:
            nickname = me["kakao_account"]["profile"]["nickname"]
        except Exception:
            nickname = "카카오 사용자"
        if not kakao_id:
            return RedirectResponse(_oauth_redirect(state))

        # 3) 같은 카카오 계정이면 기존 사용자 재사용, 없으면 새로 생성
        conn = db.connect()
        row = conn.execute(
            "SELECT token FROM users WHERE provider = 'kakao' AND provider_id = ?", (kakao_id,)
        ).fetchone()
        if row:
            token = row[0]
        else:
            token = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO users (token, provider, provider_id, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (token, "kakao", kakao_id, nickname, datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
        conn.close()

        # 4) 웹/앱으로 토큰 전달 (프론트가 URL의 token을 읽어 로그인 처리)
        return RedirectResponse(_oauth_redirect(state, token))
    except urllib.error.HTTPError as he:
        # 카카오가 돌려준 상세 에러 내용을 로그에 남깁니다(원인 진단용).
        try:
            detail = he.read().decode("utf-8", "ignore")
        except Exception:
            detail = ""
        print(f"카카오 로그인 실패 {he.code}: {detail}")
        return RedirectResponse(_oauth_redirect(state))
    except Exception as e:
        print("카카오 로그인 실패:", e)
        return RedirectResponse(_oauth_redirect(state))


# ─────────────────────────────────────────────────────────────
# 구글 로그인 (실제 OAuth) — 카카오와 같은 흐름
# ─────────────────────────────────────────────────────────────
@app.get("/auth/google/login")
def google_login():
    """구글 인증 페이지로 보냅니다."""
    params = urllib.parse.urlencode({
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
    })
    return RedirectResponse(f"https://accounts.google.com/o/oauth2/v2/auth?{params}")


@app.get("/auth/google/callback")
def google_callback(code: str = ""):
    """구글이 보내준 code로 사용자 정보를 받아 우리 계정을 만들고, 웹으로 토큰을 전달합니다."""
    if not code:
        return RedirectResponse(f"{WEB_URL}/?login_error=1")
    try:
        # 1) code → 구글 access_token (구글은 client_secret 필수)
        token_body = urllib.parse.urlencode({
            "grant_type": "authorization_code",
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "code": code,
        }).encode()
        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=token_body)
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=10) as r:
            access_token = json.loads(r.read())["access_token"]

        # 2) access_token → 구글 사용자 정보
        ui_req = urllib.request.Request("https://www.googleapis.com/oauth2/v2/userinfo")
        ui_req.add_header("Authorization", f"Bearer {access_token}")
        with urllib.request.urlopen(ui_req, timeout=10) as r:
            ui = json.loads(r.read())
        google_id = str(ui.get("id", ""))
        name = ui.get("name") or ui.get("email") or "구글 사용자"
        if not google_id:
            return RedirectResponse(f"{WEB_URL}/?login_error=1")

        # 3) 같은 구글 계정이면 기존 사용자 재사용, 없으면 새로 생성
        conn = db.connect()
        row = conn.execute(
            "SELECT token FROM users WHERE provider = 'google' AND provider_id = ?", (google_id,)
        ).fetchone()
        if row:
            token = row[0]
        else:
            token = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO users (token, provider, provider_id, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
                (token, "google", google_id, name, datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
        conn.close()

        return RedirectResponse(f"{WEB_URL}/?token={token}")
    except urllib.error.HTTPError as he:
        try:
            detail = he.read().decode("utf-8", "ignore")
        except Exception:
            detail = ""
        print(f"구글 로그인 실패 {he.code}: {detail}")
        return RedirectResponse(f"{WEB_URL}/?login_error=1")
    except Exception as e:
        print("구글 로그인 실패:", e)
        return RedirectResponse(f"{WEB_URL}/?login_error=1")


# 저장된 토큰으로 로그인 상태를 확인합니다(자동 로그인).
@app.get("/auth/me")
def auth_me(authorization: str = Header(None)):
    if not authorization:
        return {"authenticated": False}
    token = authorization.replace("Bearer ", "").strip()
    conn = db.connect()
    row = conn.execute(
        "SELECT id, provider, display_name, consent_at FROM users WHERE token = ?", (token,)
    ).fetchone()
    conn.close()
    if not row:
        return {"authenticated": False}
    consented = bool(row[3])  # 동의 시각이 있으면 동의 완료
    return {
        "authenticated": True,
        "user": {"id": row[0], "provider": row[1], "display_name": row[2], "consented": consented},
    }


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


def _inplane_roll(face, w, h):
    """
    [2단계 각도 보정] 얼굴이 화면에서 얼마나 '갸웃' 기울었는지(roll)를 라디안으로 구합니다.
    얼굴 세로축(이마 10 → 턱 152)이 똑바로 서 있으면 아래로 향합니다(기울기 0).
    이 각도만큼 점수 계산 전에 얼굴을 똑바로 세워, 기울임 때문에 생기는 가짜 비대칭을 없앱니다.
    """
    fx, fy = _point(face, 10, w, h)   # 이마 위 중앙
    cx, cy = _point(face, 152, w, h)  # 턱 아래 중앙
    return math.atan2(cx - fx, cy - fy)  # 똑바로(아래) 기준에서 벗어난 각


def _compute_scores(face, w, h):
    """
    얼굴 특징점들로 두 가지 점수를 계산합니다. (0~100점, 높을수록 좋음)
    - 안면 비대칭 점수: 좌우 짝이 되는 점들이 얼마나 대칭인지
    - 좌우 균형(부기) 점수: 얼굴 왼쪽/오른쪽 폭이 얼마나 비슷한지

    [2단계 각도 보정] 고개가 기울어진(roll) 사진은 그대로 재면 좌우 비교가 틀어져
    멀쩡한 얼굴도 비대칭으로 잡힙니다. 그래서 먼저 얼굴을 똑바로 세운 좌표(P)에서 계산합니다.
    """
    # 기울기만큼 반대로 회전시켜 얼굴을 똑바로 세웁니다(회전 중심: 코끝 1번).
    # (이미지 좌표는 y축이 아래로 향하므로, 보정 회전각은 +roll 입니다.)
    roll = _inplane_roll(face, w, h)
    ox, oy = _point(face, 1, w, h)
    cos_a, sin_a = math.cos(roll), math.sin(roll)

    def P(idx):
        """똑바로 세운 좌표로 변환한 특징점 위치."""
        x, y = _point(face, idx, w, h)
        dx, dy = x - ox, y - oy
        return (ox + dx * cos_a - dy * sin_a, oy + dx * sin_a + dy * cos_a)

    center_ids = [10, 168, 1, 152]
    midline_x = sum(P(i)[0] for i in center_ids) / len(center_ids)

    face_width = _distance(P(234), P(454))
    face_height = _distance(P(10), P(152))
    if face_width < 1 or face_height < 1:
        return None

    pairs = [(33, 263), (133, 362), (61, 291), (105, 334), (129, 358), (50, 280)]
    errors = []
    for left_id, right_id in pairs:
        lx, ly = P(left_id)
        rx, ry = P(right_id)
        horizontal = abs((midline_x - lx) - (rx - midline_x)) / face_width
        vertical = abs(ly - ry) / face_height
        errors.append(horizontal + vertical)
    mean_error = sum(errors) / len(errors)
    symmetry_score = round(max(0.0, min(100.0, 100.0 - mean_error * 250.0)))

    # 좌우 끝점(234=사진 왼쪽=사용자 오른쪽, 454=사진 오른쪽=사용자 왼쪽)
    left_width = abs(midline_x - P(234)[0])
    right_width = abs(P(454)[0] - midline_x)
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


def _gray_patch(gray, cx, cy, w, h, rad):
    """정규화 좌표(cx, cy) 주변의 작은 사각형 영역(밝기/텍스처 표본)을 잘라 돌려줍니다."""
    x, y = int(cx * w), int(cy * h)
    x0, x1 = max(0, x - rad), min(w, x + rad)
    y0, y1 = max(0, y - rad), min(h, y + rad)
    return gray[y0:y1, x0:x1]


def _compute_dark_circles(image, face, w, h):
    """
    눈밑(다크서클)과 볼의 밝기를 비교해 다크서클 점수를 냅니다. (0~100, 높을수록 양호=옅음)
    눈밑이 볼보다 어두울수록 점수가 낮아집니다. (참고용 추정)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fw = abs(face[454].x - face[234].x) * w
    rad = max(3, int(fw * 0.045))

    def under(lid_idx, cheek_idx):
        cx = face[lid_idx].x * 0.6 + face[cheek_idx].x * 0.4
        cy = face[lid_idx].y * 0.6 + face[cheek_idx].y * 0.4
        return _gray_patch(gray, cx, cy, w, h, rad)

    ul, ur = under(145, 50), under(374, 280)
    cl = _gray_patch(gray, face[50].x, face[50].y, w, h, rad)
    cr = _gray_patch(gray, face[280].x, face[280].y, w, h, rad)
    if min(ul.size, ur.size, cl.size, cr.size) == 0:
        return 0
    under_m = float(np.mean([ul.mean(), ur.mean()]))
    cheek_m = float(np.mean([cl.mean(), cr.mean()]))
    rel = max(0.0, (cheek_m - under_m) / (cheek_m + 1e-6))  # 눈밑이 더 어두운 비율
    return round(max(0.0, min(100.0, 100.0 - rel * 240.0)))


def _compute_wrinkles(image, face, w, h):
    """
    이마·미간·눈가·팔자 부위의 잔주름(텍스처)을 매끈한 볼과 비교해 주름 점수를 냅니다.
    (0~100, 높을수록 양호=매끈) 텍스처가 볼보다 많을수록 점수가 낮아집니다. (참고용 추정)
    사진 선명도 영향을 줄이려 '매끈한 볼' 기준으로 상대 비교합니다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fw = abs(face[454].x - face[234].x) * w
    rad = max(3, int(fw * 0.045))

    def lapvar(cx, cy):
        p = _gray_patch(gray, cx, cy, w, h, rad)
        if p.size < 9:
            return None
        return float(cv2.Laplacian(p, cv2.CV_32F).var())

    regions = [
        lapvar(face[151].x, face[151].y),          # 이마 중앙
        lapvar(face[9].x, face[9].y),               # 미간
        lapvar(face[33].x - 0.02, face[33].y),      # 왼쪽 눈가
        lapvar(face[263].x + 0.02, face[263].y),    # 오른쪽 눈가
        lapvar(face[205].x, face[205].y),           # 왼쪽 팔자
        lapvar(face[425].x, face[425].y),           # 오른쪽 팔자
    ]
    regions = [v for v in regions if v is not None]
    base = [lapvar(face[50].x, face[50].y), lapvar(face[280].x, face[280].y)]
    base = [v for v in base if v is not None]
    if not regions or not base:
        return 0
    ratio = (sum(regions) / len(regions)) / (sum(base) / len(base) + 1e-6)
    return round(max(0.0, min(100.0, 100.0 - (ratio - 1.0) * 18.0)))


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


# 얼굴 서명 거리가 이 값보다 크면 '다른 사람'으로 봅니다.
# (리포트 화면의 동일인 판별 기준과 동일하게 맞춥니다)
SIG_THRESHOLD = 0.2


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

    # 머리 각도(포즈) 검사: 고개가 옆/위아래로 많이 돌아가면 좌우 폭이 왜곡돼 점수가 부정확해집니다.
    # 정면이 아닐 경우 다시 촬영하도록 안내해, 매번 같은 조건에서 측정되게 합니다(일관성↑).
    pose = _estimate_head_pose(result)
    if pose is not None and (abs(pose["yaw"]) > MAX_YAW or abs(pose["pitch"]) > MAX_PITCH):
        return {
            "detected": False,
            "message": "고개가 옆이나 위아래로 돌아갔어요. 카메라를 정면으로 바라보고 다시 촬영해 주세요.",
        }

    scores = _compute_scores(face, width, height)
    if scores is None:
        return {"detected": False, "message": "얼굴이 정확히 인식되지 않았어요. 정면 얼굴이 잘 보이는 사진으로 다시 시도해 주세요."}

    skin = _compute_skin_tone(image, face, width, height)
    signature = _compute_signature(face, width, height)
    gender = _estimate_gender(image, face, width, height)
    age = _estimate_age(image, face, width, height)
    dark_circle = _compute_dark_circles(image, face, width, height)
    wrinkle = _compute_wrinkles(image, face, width, height)
    # 화면 표시용 좌표. x, y는 0~1 비율, z는 상대 깊이(간이 3D 표시에 사용).
    landmarks = [{"x": round(p.x, 4), "y": round(p.y, 4), "z": round(p.z, 4)} for p in face]
    return {
        "detected": True,
        "width": width,
        "height": height,
        "scores": scores,
        "skin": skin,
        "signature": signature,
        "gender": gender,
        "age": age,                  # 추정 나이대(참고용)
        "dark_circle": dark_circle,  # 다크서클 점수(높을수록 양호)
        "wrinkle": wrinkle,          # 주름 점수(높을수록 양호)
        "pose": pose,  # 머리 각도(정면 정도) — 측정 보정·검증용
        "landmark_count": len(face),
        "landmarks": landmarks,
    }


def _score_list(symmetry, balance, dark_circle=None, wrinkle=None):
    """점수를 앱이 쓰기 좋은 목록 형태로 만듭니다. (다크서클·주름은 값이 있을 때만 추가)"""
    items = [
        {"key": "symmetry", "label": "안면 비대칭 개선도", "value": symmetry},
        {"key": "balance", "label": "좌우 균형 (부기)", "value": balance},
    ]
    if dark_circle is not None:
        items.append({"key": "dark_circle", "label": "다크서클", "value": dark_circle})
    if wrinkle is not None:
        items.append({"key": "wrinkle", "label": "주름", "value": wrinkle})
    return items


def _record_dict(rid, created_at, symmetry, balance, image_filename, care_side="", signature_json="",
                 dark_circle=None, wrinkle=None, age=""):
    """이력 한 건을 앱에 돌려줄 형태로 정리합니다."""
    try:
        signature = json.loads(signature_json) if signature_json else []
    except (ValueError, TypeError):
        signature = []
    return {
        "id": rid,
        "created_at": created_at,
        "image_url": f"/uploads/{image_filename}",
        "scores": _score_list(symmetry, balance, dark_circle, wrinkle),
        "care_side": care_side,
        "signature": signature,
        "age": age or "",
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
        "scores": _score_list(res["scores"]["symmetry"], res["scores"]["balance"], res["dark_circle"], res["wrinkle"]),
        "age": res.get("age", ""),
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

    # 직전에 분석한 내 사진과 '같은 사람'인지 확인합니다.
    # ① 얼굴 특징(signature) 거리 ② 추정 성별(보조) — 둘 중 하나라도 다르면 다른 사람으로 봅니다.
    cur_gender = res.get("gender", "")
    conn = db.connect()
    prev = conn.execute(
        "SELECT signature, gender FROM scans WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    conn.close()
    if prev:
        prev_sig_json, prev_gender = prev[0], prev[1]
        # ① 얼굴 특징 비교
        if prev_sig_json:
            try:
                prev_sig = json.loads(prev_sig_json)
            except (ValueError, TypeError):
                prev_sig = []
            dist = _signature_distance(res["signature"], prev_sig)
            if dist is not None and dist > SIG_THRESHOLD:
                return {
                    "detected": False,
                    "different_person": True,
                    "message": "이전에 분석한 얼굴과 다른 사람으로 보입니다. 본인 얼굴 사진으로 다시 시도해 주세요.",
                }
        # ② 성별 비교 (둘 다 추정됐고 서로 다를 때만)
        if prev_gender and cur_gender and prev_gender != cur_gender:
            return {
                "detected": False,
                "different_person": True,
                "message": "이전에 분석한 얼굴과 다른 사람으로 보입니다. 본인 얼굴 사진으로 다시 시도해 주세요.",
            }

    # 사진 식별용 이름(실제 사진 데이터는 DB에 함께 저장합니다)
    filename = f"{uuid.uuid4().hex}.jpg"

    created_at = datetime.now().isoformat(timespec="seconds")
    symmetry = res["scores"]["symmetry"]
    balance = res["scores"]["balance"]
    care_side = res["scores"]["care_side"]
    brightness = res["skin"]["brightness"]
    redness = res["skin"]["redness"]
    signature_json = json.dumps(res["signature"])
    age = res.get("age", "")
    dark_circle = res.get("dark_circle", 0)
    wrinkle = res.get("wrinkle", 0)

    # 데이터베이스에 기록 + 사진 바이너리를 함께 저장합니다. (재시작해도 사진 유지)
    conn = db.connect()
    cur = conn.execute(
        """
        INSERT INTO scans (created_at, symmetry, balance, skin_brightness, skin_redness, care_side, signature, gender, age, dark_circle, wrinkle, image_filename, user_id, image_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id
        """,
        (created_at, symmetry, balance, brightness, redness, care_side, signature_json, cur_gender, age, dark_circle, wrinkle, filename, user_id, raw),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    conn.close()

    # (이미지 서빙 라우트는 아래 get_uploaded_image 에서 처리)
    return {
        "detected": True,
        "message": "분석 결과를 기록에 저장했습니다.",
        "landmark_count": res["landmark_count"],
        "image_size": {"width": res["width"], "height": res["height"]},
        "landmarks": res["landmarks"],
        "age": age,
        "record": _record_dict(new_id, created_at, symmetry, balance, filename, care_side, signature_json, dark_circle, wrinkle, age),
    }


# 서비스 대시보드 페이지 — 외부에서도 볼 수 있는 공개 대시보드 ("/dashboard")
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page():
    path = os.path.join(BASE_DIR, "dashboard.html")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return "<h1>대시보드 준비 중입니다.</h1>"


# 계정 로그인 정보 히스토리 페이지 — 가입(첫 로그인)한 계정들을 모아 봅니다. ("/logins")
# 참고: 매번 로그인할 때마다 기록을 따로 남기지는 않습니다(같은 계정은 같은 행을 재사용).
#       그래서 "가입(첫 로그인) 시각"과 동의·분석 횟수를 기준으로 보여줍니다.
@app.get("/logins", response_class=HTMLResponse)
def logins_page():
    conn = db.connect()
    try:
        rows = conn.execute(
            """
            SELECT u.id, u.provider, u.provider_id, u.display_name, u.created_at,
                   u.consent_at, u.marketing,
                   (SELECT COUNT(*) FROM scans s WHERE s.user_id = u.id) AS scan_count
            FROM users u
            ORDER BY u.id DESC
            """
        ).fetchall()
    finally:
        conn.close()

    def esc(s):
        return str(s).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

    # 로그인 방식별 표시(아이콘·이름·색)
    prov_meta = {
        "kakao": ("💬", "카카오", "#fbbf24"),
        "google": ("🔵", "구글", "#60a5fa"),
        "naver": ("🟢", "네이버", "#34d399"),
        "guest": ("👤", "게스트", "#a1a1aa"),
    }

    def mask_id(pid):
        # 소셜 계정 고유번호는 일부만 보여 줍니다(개인정보 최소 노출).
        pid = str(pid or "")
        if len(pid) <= 4:
            return pid or "—"
        return pid[:3] + "…" + pid[-2:]

    # 통계
    total = len(rows)
    consented = sum(1 for r in rows if r[5])
    by_prov = {}
    for r in rows:
        key = (r[1] or "guest").lower()
        by_prov[key] = by_prov.get(key, 0) + 1

    trs = []
    for r in rows:
        uid, provider, pid, name, created_at, consent_at, marketing, scan_count = r
        pkey = (provider or "guest").lower()
        emoji, pname, pcolor = prov_meta.get(pkey, ("👤", esc(provider or "기타"), "#a1a1aa"))
        consent_badge = (
            f'<span class="badge ok">동의 {esc(consent_at)}</span>' if consent_at
            else '<span class="badge no">미동의</span>'
        )
        mkt_badge = '<span class="badge mkt">마케팅 ✓</span>' if marketing else ''
        trs.append(
            f'<tr>'
            f'<td class="num">#{uid}</td>'
            f'<td><span class="prov" style="color:{pcolor}">{emoji} {esc(pname)}</span></td>'
            f'<td><b>{esc(name or "이름없음")}</b></td>'
            f'<td class="mono">{esc(mask_id(pid))}</td>'
            f'<td class="when">{esc(created_at)}</td>'
            f'<td>{consent_badge} {mkt_badge}</td>'
            f'<td class="num">{scan_count}</td>'
            f'</tr>'
        )
    table_body = "".join(trs) if trs else '<tr><td colspan="7" class="empty">아직 로그인한 계정이 없습니다.</td></tr>'

    prov_chips = "".join(
        f'<span class="chip">{prov_meta.get(k, ("👤", k, ""))[0]} {prov_meta.get(k, ("", k, ""))[1]} <b>{v}</b></span>'
        for k, v in sorted(by_prov.items(), key=lambda x: -x[1])
    )

    html = """<!doctype html><html lang="ko"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>HeOnn FaceFit — 로그인 계정 히스토리</title>
<style>
  body { margin:0; background:#09090b; color:#f4f4f5; font-family:-apple-system,'Malgun Gothic',sans-serif; padding:24px; }
  h1 { color:#fbbf24; font-size:22px; text-align:center; margin-bottom:4px; }
  .sub { color:#a1a1aa; font-size:13px; text-align:center; margin-bottom:18px; }
  .stats { display:flex; flex-wrap:wrap; gap:10px; justify-content:center; max-width:1000px; margin:0 auto 18px; }
  .stat { background:#18181b; border:1px solid #27272a; border-radius:12px; padding:12px 18px; text-align:center; }
  .stat .v { color:#fbbf24; font-size:24px; font-weight:800; }
  .stat .l { color:#a1a1aa; font-size:12px; margin-top:2px; }
  .chips { text-align:center; margin-bottom:18px; }
  .chip { display:inline-block; background:#18181b; border:1px solid #27272a; border-radius:999px; padding:6px 14px; margin:3px; font-size:13px; color:#d4d4d8; }
  .chip b { color:#fbbf24; margin-left:4px; }
  .wrap { max-width:1000px; margin:0 auto; overflow-x:auto; }
  table { width:100%; border-collapse:collapse; background:#18181b; border-radius:14px; overflow:hidden; font-size:13px; }
  th, td { padding:11px 12px; text-align:left; border-bottom:1px solid #27272a; white-space:nowrap; }
  th { background:#1f1f23; color:#fbbf24; font-size:12px; }
  td.num { text-align:right; color:#a1a1aa; }
  td.mono { font-family:ui-monospace,monospace; color:#a1a1aa; }
  td.when { color:#d4d4d8; }
  tr:hover td { background:#1c1c20; }
  .prov { font-weight:700; }
  .badge { display:inline-block; border-radius:6px; padding:2px 8px; font-size:11px; }
  .badge.ok { background:rgba(52,211,153,.15); color:#34d399; }
  .badge.no { background:rgba(248,113,113,.15); color:#f87171; }
  .badge.mkt { background:rgba(96,165,250,.15); color:#60a5fa; }
  .empty { text-align:center; color:#71717a; padding:40px; }
  .note { max-width:1000px; margin:16px auto 0; color:#71717a; font-size:12px; line-height:1.6; }
  .back { display:inline-block; margin-bottom:16px; color:#fbbf24; text-decoration:none; font-size:13px; }
</style></head><body>
<a class="back" href="/dashboard">← 대시보드로</a>
<h1>로그인 계정 히스토리</h1>
<div class="sub">앱·웹에서 로그인한 계정 기록 (최신순)</div>
<div class="stats">
  <div class="stat"><div class="v">__TOTAL__</div><div class="l">전체 계정</div></div>
  <div class="stat"><div class="v">__CONSENT__</div><div class="l">약관·개인정보 동의</div></div>
</div>
<div class="chips">__CHIPS__</div>
<div class="wrap">
<table>
<thead><tr>
  <th>번호</th><th>로그인 방식</th><th>이름</th><th>계정 고유번호</th><th>가입(첫 로그인)</th><th>동의 상태</th><th>분석 수</th>
</tr></thead>
<tbody>__ROWS__</tbody>
</table>
</div>
<p class="note">※ 같은 계정으로 다시 로그인해도 행이 새로 늘지 않습니다(같은 계정은 한 줄로 유지). 그래서 "가입(첫 로그인)" 시각을 기준으로 보여 드립니다.<br/>
※ 계정 고유번호는 개인정보 보호를 위해 일부만 표시합니다.</p>
</body></html>"""

    html = (
        html.replace("__TOTAL__", str(total))
        .replace("__CONSENT__", str(consented))
        .replace("__CHIPS__", prov_chips or '<span class="chip">아직 없음</span>')
        .replace("__ROWS__", table_body)
    )
    return html


# 분석 사진 갤러리 페이지 — 저장된 분석 사진을 모아 봅니다. ("/gallery")
@app.get("/gallery", response_class=HTMLResponse)
def gallery():
    conn = db.connect()
    rows = conn.execute(
        """
        SELECT s.id, s.created_at, s.image_filename, s.symmetry, s.balance, s.gender,
               s.age, s.dark_circle, s.wrinkle, u.display_name
        FROM scans s LEFT JOIN users u ON s.user_id = u.id
        WHERE s.image_data IS NOT NULL
        ORDER BY s.id DESC LIMIT 200
        """
    ).fetchall()
    conn.close()

    def esc(s):
        # HTML 속성/본문에 안전하게 넣기 위한 최소 이스케이프
        return str(s).replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")

    cards = []
    for r in rows:
        rid, created_at, fname, symmetry, balance, gender, age, dark_circle, wrinkle, name = r
        who = esc(name or "게스트")
        extra = ""
        if gender:
            extra += " · " + esc(gender)
        if age:
            extra += " · " + esc(age)
        img_src = f"/uploads/{fname}"
        cards.append(
            f'<figure class="card" data-id="{rid}" data-label="{who}" data-img="{img_src}">'
            f'<img loading="lazy" src="{img_src}" alt="분석 사진"/>'
            f'<figcaption><b>{who}</b> · {created_at}<br/>'
            f'비대칭 {symmetry} · 부기 {balance}{extra}<br/>'
            f'다크서클 {dark_circle} · 주름 {wrinkle}'
            f'<div class="btns">'
            f'<button class="btn3d" data-id="{rid}" data-label="{who}">🧊 3D 복원 보기</button>'
            f'<button class="btnrg" data-id="{rid}" data-label="{who}" data-img="{img_src}">📍 분석 부위 보기</button>'
            f'</div></figcaption>'
            f'</figure>'
        )
    body = "".join(cards) if cards else '<p class="empty">아직 저장된 분석 사진이 없습니다.</p>'

    # CSS/JS에는 중괄호가 많아 f-string을 쓰지 않고, 동적 값(개수·카드)만 끼워 넣습니다.
    head = """<!doctype html><html lang="ko"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>HeOnn FaceFit — 분석 사진 갤러리</title>
<style>
  body { margin:0; background:#09090b; color:#f4f4f5; font-family:-apple-system,'Malgun Gothic',sans-serif; padding:24px; }
  h1 { color:#fbbf24; font-size:22px; text-align:center; }
  .sub { color:#a1a1aa; font-size:13px; text-align:center; margin-bottom:24px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:14px; max-width:1100px; margin:0 auto; }
  .card { background:#18181b; border:1px solid #27272a; border-radius:14px; overflow:hidden; margin:0; transition:border-color .15s, transform .15s; }
  .card:hover { border-color:#fbbf24; transform:translateY(-2px); }
  .card img { width:100%; height:200px; object-fit:cover; display:block; background:#27272a; }
  figcaption { padding:10px 12px; font-size:12px; color:#a1a1aa; line-height:1.5; }
  figcaption b { color:#f4f4f5; }
  .btns { display:flex; gap:6px; margin-top:8px; }
  .btns button { flex:1; background:#27272a; border:1px solid #3f3f46; color:#f4f4f5; border-radius:8px; padding:7px 4px; font-size:11px; cursor:pointer; transition:background .12s, border-color .12s; }
  .btns button:hover { border-color:#fbbf24; background:#2e2e33; }
  .empty { text-align:center; color:#71717a; margin-top:60px; }
  .tabs { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; }
  .tabs button { background:#27272a; border:1px solid #3f3f46; color:#d4d4d8; border-radius:999px; padding:5px 12px; font-size:12px; cursor:pointer; }
  .tabs button.on { background:#fbbf24; color:#09090b; border-color:#fbbf24; font-weight:700; }
  #rg { width:100%; height:auto; background:#09090b; border-radius:12px; display:block; }
  .rgdesc { color:#a1a1aa; font-size:12px; line-height:1.6; margin-top:10px; }
  .ov { position:fixed; inset:0; background:rgba(0,0,0,.72); display:none; align-items:center; justify-content:center; z-index:50; }
  .ov.show { display:flex; }
  .modal { background:#18181b; border:1px solid #3f3f46; border-radius:16px; padding:16px; width:min(92vw,400px); }
  .mhead { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
  .mhead .mt { color:#f4f4f5; font-weight:700; font-size:15px; }
  .mhead button { background:none; border:none; color:#a1a1aa; font-size:20px; cursor:pointer; line-height:1; }
  #pc { width:100%; height:auto; background:#09090b; border-radius:12px; display:block; }
  .mmsg { color:#a1a1aa; font-size:12px; text-align:center; margin-top:10px; }
</style></head><body>
<h1>HeOnn FaceFit — 분석 사진 갤러리</h1>"""

    modal = """
<div id="ov" class="ov" onclick="closeM()">
  <div class="modal" onclick="event.stopPropagation()">
    <div class="mhead"><span class="mt" id="mtitle">분석 보기</span><button onclick="closeM()">✕</button></div>
    <div id="view3d" style="display:none">
      <canvas id="pc" width="360" height="360"></canvas>
    </div>
    <div id="viewrg" style="display:none">
      <div class="tabs" id="rgtabs"></div>
      <canvas id="rg" width="360" height="360"></canvas>
      <div class="rgdesc" id="rgdesc"></div>
    </div>
    <div class="mmsg" id="mmsg">불러오는 중…</div>
  </div>
</div>
<script>
  // 앱(리포트 화면)과 똑같은 분석 부위 설정입니다.
  // points: 강조 점 / pairs: 좌우로 잇는 짝 / midline: 중앙 기준선 표시
  var SCORE_DETAIL = {
    symmetry: { label:'비대칭(대칭)',
      points:[33,263,133,362,61,291,105,334,129,358,50,280],
      pairs:[[33,263],[133,362],[61,291],[105,334],[129,358],[50,280]], midline:true,
      desc:'좌우 짝이 되는 눈·눈썹·입꼬리·코·볼의 위치가 중앙선을 기준으로 얼마나 대칭인지 봅니다. 양쪽(초록 선) 점이 대칭일수록 점수가 높아요.' },
    balance: { label:'부기(좌우 폭)',
      points:[234,454,10,152], pairs:[[234,454]], midline:true,
      desc:'얼굴 중앙선에서 왼쪽·오른쪽 끝(볼)까지의 폭을 비교해 한쪽이 부었는지 봅니다. 양쪽 폭(초록 선)이 비슷할수록 점수가 높아요.' },
    dark_circle: { label:'다크서클',
      points:[145,374,50,280],
      desc:'눈 아래(노란 점)와 볼(주황 점)의 밝기를 비교합니다. 눈 밑이 볼보다 어두울수록 다크서클로 보고 점수가 낮아져요.' },
    wrinkle: { label:'주름',
      points:[151,9,33,263,205,425],
      desc:'이마·미간·눈가·팔자(노란 점) 부위의 잔주름(결)을 매끈한 볼과 비교합니다. 결이 많을수록 점수가 낮아져요.' }
  };
  var METRIC_ORDER = ['symmetry','balance','dark_circle','wrinkle'];

  var raf = null;             // 3D 회전 애니메이션 핸들
  var rgImg = null;           // 분석 부위용 얼굴 이미지
  var rgLm = null;            // 분석 부위용 특징점
  var rgMetric = 'symmetry';  // 현재 보고 있는 분석 항목

  function closeM(){
    document.getElementById('ov').classList.remove('show');
    if(raf){ cancelAnimationFrame(raf); raf=null; }
  }
  function showView(which){
    document.getElementById('view3d').style.display = (which==='3d') ? 'block' : 'none';
    document.getElementById('viewrg').style.display = (which==='rg') ? 'block' : 'none';
  }

  // ── 3D 복원 보기 ──
  async function open3D(id, label){
    var ov = document.getElementById('ov'); ov.classList.add('show');
    showView('3d');
    document.getElementById('mtitle').textContent = '🧊 3D 복원 · ' + label;
    var msg = document.getElementById('mmsg'); msg.textContent = '3D 복원 중… (얼굴 점 추출)';
    if(raf){ cancelAnimationFrame(raf); raf=null; }
    var ctx = document.getElementById('pc').getContext('2d'); ctx.clearRect(0,0,360,360);
    try {
      var res = await fetch('/scan/' + id + '/landmarks');
      var d = await res.json();
      if(!d.detected){ msg.textContent = d.message || '얼굴을 찾지 못했어요.'; return; }
      msg.textContent = '특징점 ' + d.landmark_count + '개 · 자동 회전 중';
      render3D(d.landmarks);
    } catch(e){ msg.textContent = '불러오지 못했어요. 잠시 후 다시 시도해 주세요.'; }
  }
  function render3D(lm){
    var c = document.getElementById('pc'), ctx = c.getContext('2d');
    var W = c.width, H = c.height, n = lm.length;
    var cx=0, cy=0, cz=0;
    for(var i=0;i<n;i++){ cx+=lm[i].x; cy+=lm[i].y; cz+=lm[i].z; }
    cx/=n; cy/=n; cz/=n;
    var pts = lm.map(function(p){ return { x:p.x-cx, y:p.y-cy, z:p.z-cz }; });
    var ang = 0;
    function frame(){
      ctx.clearRect(0,0,W,H);
      ang += 0.012;
      var co = Math.cos(ang), si = Math.sin(ang), scale = Math.min(W,H)*0.85;
      for(var i=0;i<pts.length;i++){
        var p = pts[i];
        var rx = p.x*co - p.z*si;
        var rz = p.x*si + p.z*co;
        var sx = W/2 + rx*scale;
        var sy = H/2 + p.y*scale;
        var t = Math.max(0, Math.min(1, (rz+0.08)/0.16));
        var size = 1 + t*2;
        ctx.fillStyle = 'rgba(251,191,36,' + (0.35 + 0.65*t).toFixed(2) + ')';
        ctx.fillRect(sx-size/2, sy-size/2, size, size);
      }
      raf = requestAnimationFrame(frame);
    }
    frame();
  }

  // ── 분석 부위 보기 ──
  async function openRegion(id, label, imgSrc){
    var ov = document.getElementById('ov'); ov.classList.add('show');
    showView('rg');
    if(raf){ cancelAnimationFrame(raf); raf=null; }
    document.getElementById('mtitle').textContent = '📍 분석 부위 · ' + label;
    var msg = document.getElementById('mmsg'); msg.textContent = '분석 부위를 불러오는 중…';
    rgImg = null; rgLm = null; rgMetric = 'symmetry';
    buildTabs();
    try {
      // 얼굴 사진과 특징점을 함께 불러옵니다.
      var imgP = new Promise(function(resolve, reject){
        var im = new Image(); im.onload = function(){ resolve(im); }; im.onerror = reject; im.src = imgSrc;
      });
      var lmP = fetch('/scan/' + id + '/landmarks').then(function(r){ return r.json(); });
      var arr = await Promise.all([imgP, lmP]);
      rgImg = arr[0];
      var d = arr[1];
      if(!d.detected){ msg.textContent = d.message || '이 사진에서는 분석 부위를 표시할 수 없어요.'; return; }
      rgLm = d.landmarks;
      msg.textContent = '점을 누르면 항목별 분석 부위를 볼 수 있어요.';
      drawRegion();
    } catch(e){ msg.textContent = '불러오지 못했어요. 잠시 후 다시 시도해 주세요.'; }
  }
  function buildTabs(){
    var box = document.getElementById('rgtabs'); box.innerHTML = '';
    METRIC_ORDER.forEach(function(k){
      var b = document.createElement('button');
      b.textContent = SCORE_DETAIL[k].label;
      if(k===rgMetric) b.className = 'on';
      b.addEventListener('click', function(){ rgMetric = k; buildTabs(); drawRegion(); });
      box.appendChild(b);
    });
  }
  function drawRegion(){
    if(!rgImg || !rgLm) return;
    var c = document.getElementById('rg'), ctx = c.getContext('2d');
    // 사진 비율에 맞춰 캔버스 크기 조정
    var W = 360, H = Math.round(360 * (rgImg.height / rgImg.width));
    c.width = W; c.height = H;
    ctx.clearRect(0,0,W,H);
    ctx.drawImage(rgImg, 0, 0, W, H);
    var cfg = SCORE_DETAIL[rgMetric];
    var lm = rgLm;
    // 중앙 기준선 (점 10·1·152 의 평균 x)
    if(cfg.midline){
      var ids = [10,1,152].filter(function(i){ return lm[i]; });
      var mx = 0; ids.forEach(function(i){ mx += lm[i].x; }); mx /= (ids.length || 1);
      ctx.save();
      ctx.strokeStyle = 'rgba(251,191,36,0.8)'; ctx.lineWidth = 1; ctx.setLineDash([5,5]);
      ctx.beginPath(); ctx.moveTo(mx*W, 0); ctx.lineTo(mx*W, H); ctx.stroke();
      ctx.restore();
    }
    // 좌우 짝을 잇는 선(초록)
    (cfg.pairs || []).forEach(function(pr){
      var a = lm[pr[0]], b = lm[pr[1]];
      if(a && b){
        ctx.strokeStyle = 'rgba(52,211,153,0.9)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(a.x*W, a.y*H); ctx.lineTo(b.x*W, b.y*H); ctx.stroke();
      }
    });
    // 강조 점(노란 원 + 어두운 테두리)
    cfg.points.forEach(function(i){
      var p = lm[i];
      if(p){
        ctx.beginPath(); ctx.arc(p.x*W, p.y*H, 4.5, 0, Math.PI*2);
        ctx.fillStyle = '#fbbf24'; ctx.fill();
        ctx.lineWidth = 1.5; ctx.strokeStyle = '#09090b'; ctx.stroke();
      }
    });
    document.getElementById('rgdesc').textContent = cfg.desc;
  }

  // 버튼 연결
  document.querySelectorAll('.btn3d').forEach(function(el){
    el.addEventListener('click', function(e){ e.stopPropagation(); open3D(el.dataset.id, el.dataset.label || '분석'); });
  });
  document.querySelectorAll('.btnrg').forEach(function(el){
    el.addEventListener('click', function(e){ e.stopPropagation(); openRegion(el.dataset.id, el.dataset.label || '분석', el.dataset.img); });
  });
</script>
</body></html>"""

    html = (
        head
        + f'<div class="sub">최근 분석 사진 {len(rows)}장 (최신순) · 사진마다 <b>3D 복원</b>·<b>분석 부위</b>를 볼 수 있어요</div>'
        + f'<div class="grid">{body}</div>'
        + modal
    )
    return html


# 저장된 분석 사진에서 3D 얼굴 점(특징점 478개)을 다시 계산해 돌려줍니다.
# 갤러리에서 사진을 누르면 이 주소로 점을 받아 3D 복원 팝업을 그립니다. ("/scan/{id}/landmarks")
@app.get("/scan/{scan_id}/landmarks")
def scan_landmarks(scan_id: int):
    conn = db.connect()
    try:
        row = conn.execute("SELECT image_data FROM scans WHERE id = ?", (scan_id,)).fetchone()
    finally:
        conn.close()
    if not row or row[0] is None:
        return {"detected": False, "message": "사진을 찾을 수 없습니다."}
    try:
        raw = bytes(row[0])
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return {"detected": False, "message": "이미지를 읽을 수 없습니다."}
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = face_landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return {"detected": False, "message": "이 사진에서 얼굴을 찾지 못했어요."}
        face = result.face_landmarks[0]
        landmarks = [{"x": round(p.x, 4), "y": round(p.y, 4), "z": round(p.z, 4)} for p in face]
        return {"detected": True, "landmark_count": len(landmarks), "landmarks": landmarks}
    except Exception as e:
        print("3D 복원 실패:", e)
        return {"detected": False, "message": "3D 복원 중 오류가 발생했어요."}


# 저장된 사진을 DB에서 읽어 돌려줍니다. ("/uploads/{파일명}")
@app.get("/uploads/{filename}")
def get_uploaded_image(filename: str):
    conn = db.connect()
    row = conn.execute(
        "SELECT image_data FROM scans WHERE image_filename = ?", (filename,)
    ).fetchone()
    conn.close()
    if not row or row[0] is None:
        return Response(status_code=404)
    return Response(content=bytes(row[0]), media_type="image/jpeg")


# 저장된 이력을 최신순으로 돌려줍니다. (로그인한 사용자의 기록만)
@app.get("/history")
def get_history(authorization: str = Header(None)):
    user_id = _current_user_id(authorization)
    if not user_id:
        return {"count": 0, "records": []}
    conn = db.connect()
    rows = conn.execute(
        """
        SELECT id, created_at, symmetry, balance, image_filename, care_side, signature, dark_circle, wrinkle, age
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

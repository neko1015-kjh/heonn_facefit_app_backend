# 자동 테스트 — 핵심 로직·엔드포인트가 정상인지 검사합니다.
# 실행: (백엔드 폴더에서)  pytest -q
# CI(GitHub Actions)에서 코드를 올릴 때마다 자동으로 돌아, 실수를 조기에 잡습니다.
import os
import sys

# 테스트는 로컬 SQLite로 동작(운영 DB에 영향 없음)
os.environ.pop("DATABASE_URL", None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import main  # noqa: E402  (임포트 자체가 '앱이 정상 부팅되는지' 검사)
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)


# ── 엔드포인트 스모크 테스트 ──
def test_root_ok():
    r = client.get("/")
    assert r.status_code == 200


def test_health_selfcheck():
    r = client.get("/health")
    data = r.json()
    # 로컬(SQLite)에선 DB 연결 성공 + 얼굴 검출 모델 로드 → healthy
    assert "status" in data and "db" in data and "healthy" in data
    assert data["db"] is True
    assert data["healthy"] is True
    assert r.status_code == 200
    # 데이터 수 지표(백업 점검용)가 숫자로 온다
    assert isinstance(data.get("users"), int)
    assert isinstance(data.get("scans"), int)


# ── 동일인 판별(임베딩 코사인) 로직 ──
def test_cosine_identical_is_one():
    v = [0.1 * i for i in range(128)]
    assert main._embedding_cosine(v, v) > 0.999


def test_cosine_orthogonal_is_zero():
    assert abs(main._embedding_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_cosine_incomparable_is_none():
    assert main._embedding_cosine([], [1.0, 2.0]) is None       # 빈 벡터
    assert main._embedding_cosine([1.0, 2.0, 3.0], [1.0, 2.0]) is None  # 길이 불일치


def test_identity_thresholds_sane():
    assert 0.0 < main.EMB_THRESHOLD < 1.0
    assert main.EMB_REF_COUNT >= 1


# ── 얼굴 크롭: 잘못된 입력은 안전하게 None ──
def test_face_crop_bad_input_returns_none():
    assert main._face_crop_jpeg(b"not-an-image") is None


# ── 재현성 변형 이미지가 형태를 유지하는지 ──
def test_perturb_keeps_image_shape():
    img = np.full((240, 200, 3), 128, dtype=np.uint8)
    out = main._perturb_image(img, 5, 1.05, 1.0, 1.0, 2, -2)
    assert out.ndim == 3 and out.shape[2] == 3

# FaceFit 백엔드 서버 (FastAPI)
# 모바일 앱(heonn_facefit_app_mobile)과 통신하는 Python 서버입니다.

from fastapi import FastAPI
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


# 기본 주소("/")로 들어오는 요청을 처리합니다.
# 앱의 "연결 확인" 기능이 바로 이 주소로 신호를 보냅니다.
@app.get("/")
def read_root():
    return {"message": "FaceFit 백엔드 서버가 정상 동작 중입니다."}


# 서버 상태를 확인하는 별도 주소입니다. ("/health")
@app.get("/health")
def health_check():
    return {"status": "ok"}

# HeOnn FaceFit 백엔드용 컨테이너 (Hugging Face Spaces - Docker)
# mediapipe + opencv가 들어가므로 시스템 라이브러리도 함께 설치합니다.

FROM python:3.12-slim

# opencv/mediapipe 실행에 필요한 시스템 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 일반 사용자로 실행 (Hugging Face Spaces 권장 방식)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

# 의존성 먼저 설치 (코드만 바뀔 때 재설치를 피해 빌드가 빨라집니다)
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 앱 코드 전체 복사
COPY --chown=user . .

# Hugging Face Spaces 기본 포트(7860)에서 서버 실행
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

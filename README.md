---
title: HeOnn FaceFit Backend
emoji: 🧖‍♀️
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# HeOnn FaceFit 백엔드

얼굴 분석 AI 서버입니다. (FastAPI + MediaPipe)

## 구성

- **웹 프레임워크**: FastAPI
- **AI**: MediaPipe FaceLandmarker (얼굴 특징점 478개 검출)
- **데이터베이스**: PostgreSQL (환경변수 `DATABASE_URL`이 있을 때) / SQLite (없을 때, 로컬 개발용)
- **배포**: Hugging Face Spaces (Docker)

## 필요한 환경변수 (Hugging Face Space의 Settings → Secrets)

- `DATABASE_URL` — PostgreSQL 접속 주소 (예: Neon). 설정하지 않으면 데이터가 보존되지 않습니다.

## 주요 기능

- 얼굴 랜드마크 검출 / 점수 분석 / 사용자별 기록 저장
- 맞춤 제품 추천
- AI 검증 지표(검출 성공률·재사용률·만족도·처리 시간) 측정

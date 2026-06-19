# 데이터베이스 연결 도우미입니다.
#
# 핵심: 환경에 따라 자동으로 다른 DB를 사용합니다.
# - 배포 서버에 DATABASE_URL(PostgreSQL 주소)이 있으면 → PostgreSQL 사용(데이터 영구 보존)
# - 없으면(내 PC에서 개발) → 기존처럼 SQLite 파일 사용
#
# 이렇게 하면 로컬 개발 방식은 그대로 두고, 배포 환경에서만 PostgreSQL로 동작합니다.

import os
import sqlite3

# 배포 환경에서 설정하는 PostgreSQL 접속 주소입니다. (없으면 로컬 SQLite 사용)
# 비밀값을 붙여넣을 때 끝에 줄바꿈/공백이 섞여 들어가는 경우가 많아 정리합니다.
DATABASE_URL = (os.environ.get("DATABASE_URL") or "").strip() or None
USE_PG = bool(DATABASE_URL)

BASE_DIR = os.path.dirname(__file__)
SQLITE_PATH = os.path.join(BASE_DIR, "facefit.db")

# 자동 증가 기본키(PK) 정의 — DB 종류마다 문법이 다릅니다.
PK = "SERIAL PRIMARY KEY" if USE_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"

# 이미지 등 바이너리 데이터 타입 (PostgreSQL: BYTEA / SQLite: BLOB)
BLOB = "BYTEA" if USE_PG else "BLOB"


class _ConnWrapper:
    """
    SQLite와 PostgreSQL 양쪽에서 같은 코드로 쿼리를 쓸 수 있게 감싼 연결입니다.
    SQLite는 물음표(?)를, PostgreSQL(psycopg)은 %s를 자리표시자로 쓰므로,
    PostgreSQL일 때만 물음표를 %s로 자동 바꿔 줍니다.
    """

    def __init__(self, raw):
        self._raw = raw

    def execute(self, sql, params=()):
        if USE_PG:
            sql = sql.replace("?", "%s")
        return self._raw.execute(sql, params)

    def commit(self):
        self._raw.commit()

    def close(self):
        self._raw.close()


def connect():
    """현재 환경에 맞는 DB 연결을 돌려줍니다."""
    if USE_PG:
        import psycopg  # 배포(PostgreSQL) 환경에서만 필요합니다.

        raw = psycopg.connect(DATABASE_URL)
    else:
        raw = sqlite3.connect(SQLITE_PATH)
    return _ConnWrapper(raw)

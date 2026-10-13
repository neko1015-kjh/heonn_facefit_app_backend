"""헤온(HeOnn) 유기판 맞춤 설계 — 외부 라이브러리 없이(표준 라이브러리만) 동작합니다.

원리
- 헤온 바디에 나사 4개로 고정되는 '방짜유기 판'의 원본 외곽선(디자이너 CAD, Final-M-08)을 기준으로 삼습니다.
- 바디에 붙는 안쪽 선·나사 구멍은 절대 바꾸지 않습니다 → 어떤 사람 도면이든 바디에 그대로 조립됩니다.
- 얼굴에 닿는 바깥 테두리의 '오목한 두 곳' 중 측정 가능한 곳만 그 사람 얼굴 곡률에 맞게 깊이를 조절합니다.
    · chin  : 위쪽 홈(폭 약 36mm) — 턱뼈 단면을 감싸는 노치. 정면 윤곽으로는 잴 수 없어 현재 원본 유지
    · cheek : 왼쪽 긴 오목면(폭 약 60mm) — 턱끝→귀 방향 턱선 곡률에 맞춤
- 바뀐 구간 양 끝은 원래 선과 기울기까지 부드럽게 이어집니다(꺾임 없음).
- 판이 너무 얇아지지 않게(최소 띠 폭) 안전 한도를 넘으면 자동으로 깊이를 줄이고 표시합니다.

원본 외곽선은 디자인 자산이라 코드에 넣지 않고, 서버 비밀 설정(HEONN_PLATE)에 압축해 보관합니다.
"""
import base64
import json
import math
import os
import zlib

# 판이 얇아지면 깨지거나 휘므로 남겨야 하는 최소 띠 폭(mm) — 디자이너 확인 필요 값
MIN_BAND_MM = 6.0
# 나사 구멍 중심에서 바깥 테두리까지 최소 거리(mm) = 바디 나사 기둥 반지름 2.5 + 여유 3.0
MIN_HOLE_EDGE_MM = 5.5
# 오목면 최소 깊이(mm) — 얼굴이 아주 평평해도 홈 모양은 남깁니다
MIN_SAGITTA_MM = 0.5
# 얼굴에 맞춰 바꾸는 오목면. 위쪽 홈(chin)은 턱뼈 '단면'을 감싸는 홈이라 정면 윤곽(사진·스캔 실루엣)으로는
# 잴 수 없어 원본(R≈30, 디자이너 설계)을 그대로 둡니다. 단면 측정이 생기면 여기에 "chin"을 추가하세요.
CUSTOM_NOTCHES = ("cheek",)
# 바뀐 구간 양 끝을 부드럽게 잇는 비율(구간 길이의 앞뒤 20%)
_BLEND = 0.2

_PLATE = None


def load_plate():
    """비밀 설정에서 헤온 유기판 원본을 읽습니다. 없으면 None.
    HEONN_PLATE      : zlib 압축 JSON을 base64로 인코딩한 값 (서버 Secret)
    HEONN_PLATE_FILE : (로컬 시험용) JSON 파일 경로"""
    global _PLATE
    if _PLATE is not None:
        return _PLATE
    try:
        raw = os.environ.get("HEONN_PLATE", "").strip()
        if raw:
            _PLATE = json.loads(zlib.decompress(base64.b64decode(raw)).decode("utf-8"))
        elif os.environ.get("HEONN_PLATE_FILE"):
            with open(os.environ["HEONN_PLATE_FILE"], encoding="utf-8") as f:
                _PLATE = json.load(f)
    except Exception as e:
        print("헤온 유기판 데이터 읽기 실패:", e)
        _PLATE = None
    return _PLATE


# ── 기하 도우미 ──────────────────────────────────────────────

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _seg_dist(p, a, b):
    """점 p와 선분 ab 사이 거리."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return _dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - (a[0] + dx * t), p[1] - (a[1] + dy * t))


def _poly_dist(p, poly):
    return min(_seg_dist(p, poly[i], poly[i + 1]) for i in range(len(poly) - 1))


def _poly_len(pts):
    return sum(_dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _chord_sagitta(pts):
    """곡선 조각의 현 길이, 가장 깊은 곳까지의 깊이(부호: 현의 왼쪽 +), 깊은 점 인덱스."""
    a, b = pts[0], pts[-1]
    c = _dist(a, b)
    if c < 1e-9:
        return 0.0, 0.0, 0
    best, bi = 0.0, 0
    for i, p in enumerate(pts):
        d = ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) / c
        if abs(d) > abs(best):
            best, bi = d, i
    return c, best, bi


def _radius_from(c, s):
    """현 길이 c, 깊이 s → 원 반지름. 거의 평평하면 큰 값."""
    s = abs(s)
    if s < 1e-6:
        return 9999.0
    return c * c / (8 * s) + s / 2


def _sagitta_from(c, R):
    """현 길이 c, 반지름 R → 깊이."""
    if R <= c / 2:
        return c / 2
    return R - math.sqrt(R * R - (c / 2) ** 2)


def _arc_offset(u, c, s):
    """현 위 위치 u(0~1)에서 깊이 s짜리 원호가 현으로부터 떨어진 거리."""
    if s < 1e-9:
        return 0.0
    R = _radius_from(c, s)
    x = (u - 0.5) * c
    return math.sqrt(max(R * R - x * x, 0.0)) - (R - s)


def _smooth_window(u):
    """양 끝 0(기울기도 0) → 가운데 1인 창 함수. 바뀐 구간을 원래 선과 부드럽게 잇습니다."""
    if u <= 0 or u >= 1:
        return 0.0
    t = min(u, 1 - u) / _BLEND
    if t >= 1:
        return 1.0
    return t * t * (3 - 2 * t)


def _resample(pts, step=1.0):
    """폴리라인을 일정 간격으로 다시 찍습니다."""
    out = [pts[0]]
    acc = 0.0
    for i in range(1, len(pts)):
        p, q = pts[i - 1], pts[i]
        d = _dist(p, q)
        while acc + d >= step and d > 1e-12:
            t = (step - acc) / d
            p = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
            out.append(p)
            d = _dist(p, q)
            acc = 0.0
        acc += d
    if _dist(out[-1], pts[-1]) > 1e-6:
        out.append(pts[-1])
    return out


# ── 얼굴(턱선)에서 각 오목면에 맞는 곡률 재기 ──────────────────

def _window_by_chord(curve, start, chord, direction):
    """curve[start]에서 direction(+1/-1) 방향으로 걸어 직선거리가 chord가 되는 조각. 모자라면 None."""
    seg = [curve[start]]
    i = start
    while 0 <= i + direction < len(curve):
        i += direction
        seg.append(curve[i])
        if _dist(seg[0], seg[-1]) >= chord:
            return seg
    return None


def face_radius(jaw, chin_idx, notch_id, chord):
    """턱선(mm, 2D)에서 오목면 폭(chord)과 같은 길이로 얼굴 곡률(반지름)을 잽니다. 실패 시 None.
    chin : 턱끝을 가운데 두고 좌우로 같은 거리 → 턱끝 둥근 정도
    cheek: 턱끝에서 귀 쪽으로(좌·우 각각) → 턱선 곡률, 두 쪽 평균"""
    fine = _resample([tuple(p) for p in jaw], 0.5)
    ci = min(range(len(fine)), key=lambda k: _dist(fine[k], jaw[chin_idx]))
    if notch_id == "chin":
        # 턱끝 중심 대칭 창: 반쪽 길이를 늘려가며 양 끝 직선거리가 chord가 되는 지점
        for half in range(1, len(fine)):
            a, b = ci - half, ci + half
            if a < 0 or b >= len(fine):
                return None
            if _dist(fine[a], fine[b]) >= chord:
                c, s, _ = _chord_sagitta(fine[a:b + 1])
                return _radius_from(c, s)
        return None
    rs = []
    for d in (-1, 1):
        seg = _window_by_chord(fine, ci, chord, d)
        if seg:
            c, s, _ = _chord_sagitta(seg)
            rs.append(_radius_from(c, s))
    return sum(rs) / len(rs) if rs else None


# ── 유기판 맞춤 설계 ─────────────────────────────────────────

def _reshape_notch(outer, i0, i1, s_target):
    """outer[i0..i1] 오목 구간을 깊이 s_target 원호에 가깝게 바꾼 새 점 목록(구간만) 반환."""
    seg = outer[i0:i1 + 1]
    a, b = seg[0], seg[-1]
    c, s_signed, _ = _chord_sagitta(seg)
    s_orig = abs(s_signed)
    # 현에서 홈 안쪽(판 안쪽)으로 향하는 단위 법선
    ux, uy = (b[0] - a[0]) / c, (b[1] - a[1]) / c
    nx, ny = (-uy, ux) if s_signed > 0 else (uy, -ux)
    out = []
    for p in seg:
        u = ((p[0] - a[0]) * ux + (p[1] - a[1]) * uy) / c
        delta = (_arc_offset(u, c, s_target) - _arc_offset(u, c, s_orig)) * _smooth_window(u)
        out.append((p[0] + nx * delta, p[1] + ny * delta))
    return out


def _safe(seg_new, inner, holes):
    """바뀐 구간이 최소 띠 폭·나사 구멍 여유를 지키는지 확인. (가장 얇은 폭, 통과 여부)"""
    w = min(_poly_dist(p, inner) for p in seg_new)
    h = min((_dist(p, (hh["x"], hh["y"])) for p in seg_new for hh in holes), default=99.0)
    return w, (w >= MIN_BAND_MM and h >= MIN_HOLE_EDGE_MM)


def design_plate(jaw, chin_idx, plate, source="정면 사진", ipd_mm=63.0):
    """턱선(mm 2D) + 헤온 유기판 원본 → 맞춤 유기판 설계(cad dict). 실패 시 None."""
    if not plate or not jaw or chin_idx is None:
        return None
    outer = [tuple(p) for p in plate["outer"]]
    inner = [tuple(p) for p in plate["inner"]]
    holes = plate["holes"]
    new_outer = list(outer)
    notches = []
    for nt in plate["notches"]:
        i0, i1 = nt["i0"], nt["i1"]
        seg = outer[i0:i1 + 1]
        c, s0, _ = _chord_sagitta(seg)
        s0 = abs(s0)
        R0 = _radius_from(c, s0)
        if nt["id"] not in CUSTOM_NOTCHES:
            # 원본 유지 구간 — 측정값 없이 원래 모양 그대로
            notches.append({
                "id": nt["id"], "label": nt["label"], "i0": i0, "i1": i1, "chord": round(c, 1),
                "orig_depth": round(s0, 1), "orig_radius": round(R0, 1), "face_radius": None,
                "depth": round(s0, 1), "radius": round(R0, 1),
                "min_band": round(min(_poly_dist(p, inner) for p in seg), 1),
                "limited": False, "fixed": True,
            })
            continue
        Rf = face_radius(jaw, chin_idx, nt["id"], c)
        if Rf is None:
            return None
        # 1:1 밀착 — 판 홈 반지름 = 얼굴 곡률 반지름
        s_want = max(_sagitta_from(c, Rf), MIN_SAGITTA_MM)
        s_ok = s_want
        new_seg = _reshape_notch(outer, i0, i1, s_ok)
        w, ok = _safe(new_seg, inner, holes)
        limited = False
        if not ok:
            # 안전 한도 안에서 가능한 가장 깊은 값(이분 탐색)
            lo, hi = MIN_SAGITTA_MM, s_want
            for _ in range(30):
                mid = (lo + hi) / 2
                if _safe(_reshape_notch(outer, i0, i1, mid), inner, holes)[1]:
                    lo = mid
                else:
                    hi = mid
            s_ok = lo
            new_seg = _reshape_notch(outer, i0, i1, s_ok)
            w, _ = _safe(new_seg, inner, holes)
            limited = True
        new_outer[i0:i1 + 1] = new_seg
        c1, s1, _ = _chord_sagitta(new_seg)
        notches.append({
            "id": nt["id"], "label": nt["label"], "i0": i0, "i1": i1,
            "chord": round(c, 1),
            "orig_depth": round(s0, 1), "orig_radius": round(R0, 1),
            "face_radius": round(Rf, 1),
            "depth": round(abs(s1), 1), "radius": round(_radius_from(c1, s1), 1),
            "min_band": round(w, 1), "limited": limited, "fixed": False,
        })
    band_all = min(_poly_dist(p, inner) for k, p in enumerate(new_outer) if 20 < k < len(new_outer) - 20)
    by = {n["id"]: n for n in notches}
    chin = by.get("chin", notches[0])
    return {
        "kind": "heonn_plate",
        "plate_name": plate.get("name", "HeOnn plate"),
        "outer": new_outer, "inner": inner, "orig_outer": outer, "holes": holes,
        "notches": notches, "source": source, "ipd_mm": round(float(ipd_mm), 1),
        # 앱(기존 화면)과 호환되는 요약값
        "arc_len": round(_poly_len(new_outer), 1),
        "radius": chin["radius"],
        "min_radius": min(n["radius"] for n in notches),
        "chord": chin["chord"], "sagitta": chin["depth"],
        "head_width": round(band_all, 1), "thickness": 2.0,
        "mount_w": None, "hole_gap": None,
        "hole_d": holes[0]["d"] if holes else None,
        "profile": plate.get("profile"),   # 바깥 표면 원본 단면(3D 보기용)
    }


# ── 출력: DXF · SVG ──────────────────────────────────────────

def _closed_outline(outer, inner):
    """바깥선 + 안쪽선을 이어 닫힌 외곽 하나로."""
    inn = inner if _dist(outer[-1], inner[0]) < _dist(outer[-1], inner[-1]) else list(reversed(inner))
    return list(outer) + list(inn[1:])


def plate_to_dxf(cad):
    """맞춤 유기판 → DXF(R12). 좌표는 디자이너 원본 CAD와 같은 위치(mm)라 원본 위에 그대로 겹쳐 쓸 수 있습니다.
    레이어: PLATE_OUTLINE(가공 외곽) · CUSTOM_ZONE(맞춤으로 바뀐 구간) · ORIGINAL_EDGE(원본 바깥선, 참고) · HOLES · NOTE"""
    ent = []

    def line(p, q, layer):
        ent.append(f"0\nLINE\n8\n{layer}\n10\n{p[0]:.3f}\n20\n{p[1]:.3f}\n30\n0\n11\n{q[0]:.3f}\n21\n{q[1]:.3f}\n31\n0\n")

    def poly(pts, layer):
        for i in range(len(pts) - 1):
            line(pts[i], pts[i + 1], layer)

    def text(x, y, h, s, layer="NOTE"):
        ent.append(f"0\nTEXT\n8\n{layer}\n10\n{x:.3f}\n20\n{y:.3f}\n30\n0\n40\n{h:.2f}\n1\n{s}\n")

    outline = _closed_outline(cad["outer"], cad["inner"])
    poly(outline + [outline[0]], "PLATE_OUTLINE")
    poly(cad["orig_outer"], "ORIGINAL_EDGE")
    for n in cad["notches"]:
        if not n.get("fixed"):
            poly(cad["outer"][n["i0"]:n["i1"] + 1], "CUSTOM_ZONE")
    for h in cad["holes"]:
        ent.append(f"0\nCIRCLE\n8\nHOLES\n10\n{h['x']:.3f}\n20\n{h['y']:.3f}\n30\n0\n40\n{h['d'] / 2:.3f}\n")
    xs = [p[0] for p in outline]; ys = [p[1] for p in outline]
    x0, y0 = min(xs), min(ys) - 10
    text(x0, y0, 4.0, "HeOnn custom brass plate - plan view (mm), same coords as Final-M-08")
    for k, n in enumerate(cad["notches"]):
        if n.get("fixed"):
            msg = f"{n['id']}: ORIGINAL kept (chord={n['chord']} depth={n['depth']} R={n['radius']})"
        else:
            msg = (f"{n['id']}: chord={n['chord']} depth {n['orig_depth']}->{n['depth']} R {n['orig_radius']}->{n['radius']}"
                   f" (face R={n['face_radius']}) minBand={n['min_band']}{' LIMITED' if n['limited'] else ''}")
        text(x0, y0 - 7 - 6 * k, 3.0, msg)
    # 디자이너 안내: 이 도면은 가운데 높이의 평면 외곽선 — 바뀐 구간은 3D 바디 표면도 함께 다시 만들어야 함
    ny = y0 - 7 - 6 * len(cad["notches"])
    text(x0, ny, 3.0, "NOTE: plan outline at body mid-height (z=0). Outer face surface is the body's rounded section"
                      " (~22mm tall, R10-25) - NOT a flat plate edge.")
    text(x0, ny - 6, 3.0, "In CUSTOM_ZONE, re-loft the body/brass outer surface to follow the new outline,"
                          " keeping the original section shape. Elsewhere keep original 3D model.")
    body = "".join(ent)
    return "0\nSECTION\n2\nENTITIES\n" + body + "0\nENDSEC\n0\nEOF\n"


def plate_svg(cad, W=420, H=480, pad=24):
    """미리보기 SVG — 원본 바깥선(점선) 위에 맞춤 외곽(금색)과 바뀐 구간(강조)을 겹쳐 그립니다."""
    outline = _closed_outline(cad["outer"], cad["inner"])
    allp = outline + cad["orig_outer"]
    minx = min(p[0] for p in allp); maxx = max(p[0] for p in allp)
    miny = min(p[1] for p in allp); maxy = max(p[1] for p in allp)
    s = min((W - 2 * pad) / (maxx - minx), (H - 2 * pad) / (maxy - miny))

    def X(x):
        return pad + (x - minx) * s + ((W - 2 * pad) - (maxx - minx) * s) / 2

    def Y(y):
        return H - pad - (y - miny) * s

    def d(pts, close=False):
        return " ".join(("M" if i == 0 else "L") + f"{X(p[0]):.1f} {Y(p[1]):.1f}" for i, p in enumerate(pts)) + (" Z" if close else "")

    parts = [f'<path d="{d(outline, True)}" fill="rgba(245,158,11,0.16)" stroke="#d97706" stroke-width="1.4"/>',
             f'<path d="{d(cad["orig_outer"])}" fill="none" stroke="#71717a" stroke-width="1.2" stroke-dasharray="4 3"/>']
    for n in cad["notches"]:
        if n.get("fixed"):
            continue
        parts.append(f'<path d="{d(cad["outer"][n["i0"]:n["i1"] + 1])}" fill="none" stroke="#fbbf24" '
                     f'stroke-width="4.5" stroke-linecap="round"/>')
    for h in cad["holes"]:
        parts.append(f'<circle cx="{X(h["x"]):.1f}" cy="{Y(h["y"]):.1f}" r="{max(2.5, h["d"] / 2 * s):.1f}" '
                     f'fill="#09090b" stroke="#a1a1aa" stroke-width="1.3"/>')
    return (f'<svg width="100%" viewBox="0 0 {W} {H}" style="max-width:{W}px">' + "".join(parts) + "</svg>")


def _profile_at(profile, k):
    """원본 바깥선 인덱스 k 위치의 단면(높이별 u) — 측정 스테이션 사이를 선형 보간. 값이 없는 높이는 None."""
    step = profile["step"]; rows = profile["u"]
    a = min(int(k // step), len(rows) - 1); b = min(a + 1, len(rows) - 1)
    t = (k - a * step) / step if b != a else 0.0
    out = []
    for ua, ub in zip(rows[a], rows[b]):
        if ua is None or ub is None:
            out.append(None)
        else:
            out.append(round(ua + (ub - ua) * t, 2))
    return out


def plate_viewer_html(cad):
    """맞춤 유기판 3D 보기(외부 라이브러리 없이 canvas로 직접 그림).
    얼굴에 닿는 바깥 표면을 원본 CAD에서 잰 단면(높이 약 23mm의 둥근 면) 그대로 테두리를 따라 이어 그립니다.
    맞춤 구간은 바뀐 외곽선을 따라 같은 단면을 옮겨 그린 '예상 모양'입니다."""
    outer, n_out = cad["outer"], len(cad["outer"])
    prof = cad.get("profile")
    if not prof:
        return '<div class="v3h">3D 단면 데이터가 서버에 없어요(HEONN_PLATE v2 필요).</div>'
    keep = list(range(0, n_out, 2))          # 1mm 간격
    pts = [[round(outer[k][0], 2), round(outer[k][1], 2)] for k in keep]
    secs = [_profile_at(prof, k) for k in keep]
    zone = []
    for n in cad["notches"]:
        if not n.get("fixed"):
            js = [j for j, k in enumerate(keep) if n["i0"] <= k <= n["i1"]]
            if js:
                zone.append([js[0], js[-1]])
    outline = _closed_outline(cad["outer"], cad["inner"])
    ref = [[round(p[0], 2), round(p[1], 2)] for p in outline[::3]]
    orig = [[round(p[0], 2), round(p[1], 2)] for p in cad["orig_outer"][::2]]
    hz = [z for z, col in zip(prof["z"], zip(*secs)) if any(v is not None for v in col)]
    height = (max(hz) - min(hz)) if hz else 0
    data = json.dumps({"pts": pts, "sec": secs, "z": prof["z"], "zone": zone, "ref": ref, "orig": orig})
    return """
<div class="v3d">
  <canvas id="v3d" width="420" height="380" style="width:100%;max-width:420px;touch-action:none;cursor:grab"></canvas>
  <div class="v3b">
    <button data-v="top">위에서</button><button data-v="iso">비스듬히</button><button data-v="side">옆에서</button>
    <button id="v3o">원본선 숨기기</button>
  </div>
  <div class="v3h">드래그 = 회전 · 휠/두 손가락 = 확대<br/>얼굴에 닿는 <b>바깥 표면</b>을 원본 CAD 단면(높이 약 __H__mm, 둥근 면) 그대로 표시 · 주황 = 이 얼굴에 맞춘 구간(같은 단면을 새 외곽선에 옮긴 예상 모양) · 회색 선 = 가운데 높이 기준선</div>
</div>
<style>
  .v3d { margin-top:14px; } .v3d canvas { display:block; margin:0 auto; background:#0d0d10; border-radius:12px; }
  .v3b { display:flex; gap:6px; flex-wrap:wrap; justify-content:center; margin-top:8px; }
  .v3b button { background:#27272a; color:#e4e4e7; border:1px solid #3f3f46; border-radius:999px; padding:6px 11px; font-size:12px; cursor:pointer; }
  .v3h { color:#71717a; font-size:11.5px; text-align:center; margin-top:6px; line-height:1.6; }
  .v3h b { color:#a1a1aa; }
</style>
<script>
(function(){
  var D = __DATA__;
  var cv = document.getElementById('v3d'), ctx = cv.getContext('2d');
  var W = cv.width, H = cv.height, P = D.pts, N = P.length, Z = D.z, M = Z.length;
  var mnx=1e9,mny=1e9,mxx=-1e9,mxy=-1e9;
  D.ref.forEach(function(p){ mnx=Math.min(mnx,p[0]); mny=Math.min(mny,p[1]); mxx=Math.max(mxx,p[0]); mxy=Math.max(mxy,p[1]); });
  var cx=(mnx+mxx)/2, cy=(mny+mxy)/2, span=Math.max(mxx-mnx,mxy-mny);
  // 기준 외곽(바깥+안쪽) 방향 → 바깥선 점마다 바깥 방향 법선
  var A=0, R=D.ref; for (var i=0;i<R.length;i++){ var a=R[i], b=R[(i+1)%R.length]; A+=a[0]*b[1]-b[0]*a[1]; } var ccw=A>0;
  var VN = P.map(function(p,i){ var a=P[Math.max(i-1,0)], b=P[Math.min(i+1,N-1)], dx=b[0]-a[0], dy=b[1]-a[1], l=Math.hypot(dx,dy)||1; return ccw?[dy/l,-dx/l]:[-dy/l,dx/l]; });
  var inZone = new Array(N).fill(false); D.zone.forEach(function(z){ for (var k=z[0]; k<z[1]; k++) inZone[k]=true; });
  // 바깥 표면 3D 점: 외곽선 점 + 법선 × u(높이별)
  var S3 = P.map(function(p,i){ return D.sec[i].map(function(u,m){ return u===null ? null : [p[0]+VN[i][0]*u, p[1]+VN[i][1]*u, Z[m]]; }); });
  var yaw=-0.5, pitch=0.95, zoom=1, showOrig=true, auto=true, L=norm([0.4,-0.5,0.8]);
  function norm(v){ var l=Math.hypot(v[0],v[1],v[2])||1; return [v[0]/l,v[1]/l,v[2]/l]; }
  function rot(x,y,z){ var c=Math.cos(yaw), s=Math.sin(yaw), x1=x*c-y*s, y1=x*s+y*c, cp=Math.cos(pitch), sp=Math.sin(pitch); return [x1, y1*cp-z*sp, y1*sp+z*cp]; }
  function proj(r){ var sc=(W*0.74/span)*zoom, f=900/(900-r[2]*sc*0.6); return [W/2+r[0]*sc*f, H/2-r[1]*sc*f]; }
  function P3(q){ return proj(rot(q[0]-cx,q[1]-cy,q[2])); }
  function shade(rgb,n){ var k=0.3+0.7*Math.max(0,n[0]*L[0]+n[1]*L[1]+n[2]*L[2]); return 'rgb('+[0,1,2].map(function(i){return Math.round(rgb[i]*k);}).join(',')+')'; }
  var BRASS=[214,172,88], ZONE=[255,150,30];
  function line(pts, z, col, w, dash){ ctx.setLineDash(dash||[]); ctx.lineWidth=w; ctx.strokeStyle=col; ctx.beginPath();
    pts.forEach(function(p,k){ var q=P3([p[0],p[1],z]); k?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]); }); ctx.stroke(); ctx.setLineDash([]); }
  function draw(){
    ctx.clearRect(0,0,W,H);
    // 가운데 높이 기준선(유기판 평면 외곽, 참고)
    line(D.ref.concat([D.ref[0]]), 0, 'rgba(161,161,170,0.45)', 1);
    var quads=[];
    for (var i=0;i<N-1;i++) for (var m=0;m<M-1;m++){
      var a=S3[i][m], b=S3[i+1][m], c=S3[i+1][m+1], d=S3[i][m+1]; if (!a||!b||!c||!d) continue;
      var e1=[b[0]-a[0],b[1]-a[1],b[2]-a[2]], e2=[d[0]-a[0],d[1]-a[1],d[2]-a[2]];
      var n=norm([e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]]);
      if (n[0]*VN[i][0]+n[1]*VN[i][1] < 0) n=[-n[0],-n[1],-n[2]];        // 바깥을 향하게
      var rn=rot(n[0],n[1],n[2]); if (rn[2] <= 0) continue;              // 뒤쪽 면은 생략
      var mid=rot((a[0]+c[0])/2-cx,(a[1]+c[1])/2-cy,(a[2]+c[2])/2);
      quads.push({q:[a,b,c,d], d:mid[2], n:rn, z:inZone[i]});
    }
    quads.sort(function(p,q){ return p.d-q.d; });
    quads.forEach(function(o){ ctx.beginPath(); o.q.forEach(function(v,k){ var s=P3(v); k?ctx.lineTo(s[0],s[1]):ctx.moveTo(s[0],s[1]); }); ctx.closePath();
      var col=shade(o.z?ZONE:BRASS, o.n); ctx.fillStyle=col; ctx.strokeStyle=col; ctx.lineWidth=0.6; ctx.fill(); ctx.stroke(); });
    if (showOrig) line(D.orig, 0, 'rgba(250,250,250,0.6)', 1.2, [4,3]);
  }
  var ptrs={}, last=null, pinch0=null;
  cv.addEventListener('pointerdown', function(e){ auto=false; cv.setPointerCapture(e.pointerId); ptrs[e.pointerId]=[e.clientX,e.clientY]; last=[e.clientX,e.clientY]; cv.style.cursor='grabbing'; });
  cv.addEventListener('pointermove', function(e){
    if (!(e.pointerId in ptrs)) return; ptrs[e.pointerId]=[e.clientX,e.clientY]; var ids=Object.keys(ptrs);
    if (ids.length===2){ var a=ptrs[ids[0]], b=ptrs[ids[1]], dd=Math.hypot(a[0]-b[0],a[1]-b[1]); if (pinch0) zoom=Math.max(0.5,Math.min(6,zoom*dd/pinch0)); pinch0=dd; draw(); return; }
    yaw += (e.clientX-last[0])*0.01; pitch = Math.max(-1.5,Math.min(1.5,pitch+(e.clientY-last[1])*0.01)); last=[e.clientX,e.clientY]; draw();
  });
  function up(e){ delete ptrs[e.pointerId]; pinch0=null; cv.style.cursor='grab'; }
  cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
  cv.addEventListener('wheel', function(e){ e.preventDefault(); auto=false; zoom=Math.max(0.5,Math.min(6,zoom*(e.deltaY<0?1.1:0.9))); draw(); }, {passive:false});
  document.querySelectorAll('.v3b button[data-v]').forEach(function(b){ b.onclick=function(){ auto=false; var v=b.dataset.v;
    if (v==='top'){ yaw=0; pitch=0; } else if (v==='iso'){ yaw=-0.5; pitch=0.95; } else { yaw=0; pitch=1.45; } zoom=1; draw(); }; });
  document.getElementById('v3o').onclick=function(){ showOrig=!showOrig; this.textContent = showOrig?'원본선 숨기기':'원본선 보이기'; draw(); };
  (function spin(){ if (auto){ yaw += 0.006; draw(); requestAnimationFrame(spin); } })();
  draw();
})();
</script>""".replace("__DATA__", data).replace("__H__", f"{height:g}")

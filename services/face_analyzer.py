import math
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from typing import Any, Dict, List, Tuple

from services.face_shape_classifier_v2 import classify_face_shape
from services.face_quality_inspector import inspect_image_quality

# 500장 전역 통계 로드
CSV_PATH = 'results/face_shape_random_500_heart_features_analysis.csv'
DF = pd.read_csv(CSV_PATH)

FEATURES = [
    "face_length_width_ratio", "forehead_to_cheekbone_ratio", "jaw_to_cheekbone_ratio",
    "forehead_to_jaw_ratio", "forehead_to_lower_jaw_ratio", "lower_jaw_width_to_cheekbone_ratio",
    "chin_sharpness_ratio", "jaw_taper_ratio", "lower_face_height_to_face_height_ratio",
    "gonial_angle_proxy", "temple_to_cheekbone_ratio", "jaw_slope_angle",
    "jawline_curvature_index", "midface_height_ratio"
]
FEAT_MAP = {f: (f"ratio_{f}" if f"ratio_{f}" in DF.columns else f) for f in FEATURES}

GLOBAL_STATS = {}
for f in FEATURES:
    s = DF[FEAT_MAP[f]].dropna()
    GLOBAL_STATS[f] = {
        "mean": float(s.mean()),
        "p70": float(np.percentile(s, 70)),
        "p50": float(np.percentile(s, 50)),
        "p30": float(np.percentile(s, 30))
    }

oval_lw_err = np.abs(DF[FEAT_MAP["face_length_width_ratio"]].dropna() - 1.20)
oval_jaw_err = np.abs(DF[FEAT_MAP["jaw_to_cheekbone_ratio"]].dropna() - 0.784)
GLOBAL_STATS["oval_lw_err"] = {
    "mean": float(oval_lw_err.mean()),
    "p30": float(np.percentile(oval_lw_err, 30)),
    "p50": float(np.percentile(oval_lw_err, 50))
}
GLOBAL_STATS["oval_jaw_err"] = {
    "mean": float(oval_jaw_err.mean()),
    "p30": float(np.percentile(oval_jaw_err, 30)),
    "p50": float(np.percentile(oval_jaw_err, 50))
}

FEATURE_SCHEMA = {
    "heart": [
        {
            "key": "chin_sharpness_ratio", "name": "턱 끝 뾰족도 (Sharpness)", "dir": "high", "unit": "",
            "metric_meaning": "귀밑턱 너비 대비 턱 끝 바로 위 지점이 좁아지는 수축 비율을 측정한 지표입니다.",
            "visual_meaning": "턱 끝 정점이 뭉툭하지 않고 날렵한 V라인 정점을 형성하여 세련되고 도회적인 인상을 줍니다."
        },
        {
            "key": "forehead_to_lower_jaw_ratio", "name": "이마-하관 수축비 (Taper)", "dir": "high", "unit": "",
            "metric_meaning": "이마 가로폭 대비 하악 말단(입술 아래) 폭의 급격한 축소 비율을 측정한 지표입니다.",
            "visual_meaning": "상안부가 시원하고 하관으로 내려갈수록 갸름해지는 전형적인 역삼각형 실루엣을 만듭니다."
        },
        {
            "key": "temple_to_cheekbone_ratio", "name": "관자놀이 상안부폭 (Temple)", "dir": "high", "unit": "",
            "metric_meaning": "광대뼈 최대 너비 대비 관자놀이 양 끝의 가로 여백을 측정한 지표입니다.",
            "visual_meaning": "이마 상단이 좁아 보이지 않고 시원하게 확장되어 시선을 상안부로 자연스럽게 유도합니다."
        },
        {
            "key": "jaw_taper_ratio", "name": "하관 급경사 수축률 (Slope)", "dir": "high", "unit": "",
            "metric_meaning": "귀밑턱에서 턱 끝으로 떨어지는 외곽선의 물리적 기울기와 경사도를 측정한 지표입니다.",
            "visual_meaning": "하관 라인이 처지거나 늘어지지 않고 턱 끝으로 가파르게 수렴하여 슬림한 턱선을 강조합니다."
        }
    ],
    "long": [
        {
            "key": "face_length_width_ratio", "name": "얼굴 세로/가로비 (L/W)", "dir": "high", "unit": "",
            "metric_meaning": "좌우 광대 가로폭 대비 이마 정점부터 턱 끝까지의 전체 수직 길이를 측정한 지표입니다.",
            "visual_meaning": "얼굴의 종횡비가 길어 시각적으로 슬림하며, 차분하고 지적이며 성숙한 분위기를 풍깁니다."
        },
        {
            "key": "midface_height_ratio", "name": "중안부 세로 비율 (Midface)", "dir": "high", "unit": "",
            "metric_meaning": "전체 얼굴 세로 길이 중 미간에서 코밑까지의 세로 비중을 측정한 지표입니다.",
            "visual_meaning": "얼굴 중심부 여백이 여유로워 클래식하고 우아하며 도회적인 인상을 완성합니다."
        },
        {
            "key": "lower_face_height_to_face_height_ratio", "name": "하안부 길이 비율 (Lower)", "dir": "high", "unit": "",
            "metric_meaning": "전체 얼굴 세로 길이 중 코밑에서 턱 끝까지의 세로 비중을 측정한 지표입니다.",
            "visual_meaning": "인중과 턱 끝의 길이가 길고 시원하게 뻗어 균형 잡힌 세로 라인을 형성합니다."
        }
    ],
    "square": [
        {
            "key": "gonial_angle_proxy", "name": "귀밑턱 각도 (Gonial Angle)", "dir": "low", "unit": "°",
            "metric_meaning": "광대-귀밑턱-턱끝 정점이 이루는 하악각(Gonial Angle)의 꺾임 각도를 측정한 지표입니다.",
            "visual_meaning": "귀 밑에서 턱 끝으로 이어지는 모서리가 선명하게 살아있어 탄탄하고 고급스러운 골격을 연출합니다."
        },
        {
            "key": "jaw_to_cheekbone_ratio", "name": "광대 대비 턱 폭비 (Jaw/Cheek)", "dir": "high", "unit": "",
            "metric_meaning": "광대뼈 너비 대비 좌우 귀밑턱 사이의 가로폭 비율을 측정한 지표입니다.",
            "visual_meaning": "하관 지지면이 정면에서 넓게 자리 잡아 이목구비를 안정감 있게 받쳐주는 신뢰감을 줍니다."
        },
        {
            "key": "lower_jaw_width_to_cheekbone_ratio", "name": "아래턱 폭비 (Lower Jaw)", "dir": "high", "unit": "",
            "metric_meaning": "광대뼈 너비 대비 입술 양옆 아래턱의 가로 너비 비중을 측정한 지표입니다.",
            "visual_meaning": "하관 양옆이 좁아지지 않고 반듯하게 유지되어 클래식하고 당당한 인상을 돋보이게 합니다."
        },
        {
            "key": "jawline_curvature_index", "name": "턱선 직선성 RMSE (Linearity)", "dir": "high", "unit": "",
            "metric_meaning": "턱선 외곽 점들의 잔차를 통해 직선형 골격 경사를 측정한 지표입니다.",
            "visual_meaning": "턱선이 둥글게 처지지 않고 직선적인 엣지를 띠어 모던하고 또렷한 실루엣을 만듭니다."
        }
    ],
    "round": [
        {
            "key": "face_length_width_ratio", "name": "얼굴 짧음 비율 (Shortness)", "dir": "low", "unit": "",
            "metric_meaning": "얼굴 가로 폭 대비 세로 길이가 짧아 1:1에 근접한 정도를 측정한 지표입니다.",
            "visual_meaning": "상하 길이가 아담하고 균형을 이루어 본래 나이보다 어려 보이는 동안(Baby-face) 인상을 줍니다."
        },
        {
            "key": "gonial_angle_proxy", "name": "귀밑턱 완만도 (Soft Angle)", "dir": "high", "unit": "°",
            "metric_meaning": "귀밑턱에 각진 모서리가 없이 부드럽게 굴려진 내각의 완만도를 측정한 지표입니다.",
            "visual_meaning": "턱 라인에 각진 꺾임이 없어 부드럽고 친근하며 호감도 높은 인상을 형성합니다."
        },
        {
            "key": "jawline_curvature_index", "name": "턱선 곡선성 RMSE (Smoothness)", "dir": "low", "unit": "",
            "metric_meaning": "턱선 점들이 매끄러운 곡선과 얼마나 완벽히 일치하는지 측정한 지표입니다.",
            "visual_meaning": "볼살의 볼륨감과 하관이 매끄러운 곡선 형태로 이어져 온화한 분위기를 줍니다."
        },
        {
            "key": "chin_sharpness_ratio", "name": "턱 끝 둥글기 (Blunt Chin)", "dir": "low", "unit": "",
            "metric_meaning": "턱 끝 정점이 뾰족하게 튀어나오지 않고 둥글게 완충된 정도를 측정한 지표입니다.",
            "visual_meaning": "턱 끝이 둥글고 부드럽게 마감되어 부드럽고 사랑스러운 이미지를 강조합니다."
        }
    ],
    "oval": [
        {
            "key": "face_length_width_ratio", "name": "세로/가로 황금 밸런스 (1.20근접)", "dir": "target_1.20", "unit": "",
            "metric_meaning": "얼굴 종횡비가 가장 이상적인 표준 타원형 비례(1:1.20)에 수렴하는 정도를 측정한 지표입니다.",
            "visual_meaning": "특정 부위로 치우침 없는 황금 비례를 이루어 단정하고 안정적인 인상을 줍니다."
        },
        {
            "key": "jaw_to_cheekbone_ratio", "name": "하관 비례 조화도 (0.784근접)", "dir": "target_0.784", "unit": "",
            "metric_meaning": "광대 대비 하관 폭이 표준 중립 비례(0.784)에 수렴하는 정도를 측정한 지표입니다.",
            "visual_meaning": "턱이 너무 넓지도 좁지도 않아 다양한 헤어 및 메이크업 스타일을 자유롭게 소화합니다."
        }
    ]
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh_detector = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

def dist_2d(p1, p2):
    return float(np.linalg.norm(p1[:2] - p2[:2]))

def get_angle_2d(p1, p2, p3):
    v1 = p1[:2] - p2[:2]
    v2 = p3[:2] - p2[:2]
    dot = np.dot(v1, v2)
    mag1 = np.linalg.norm(v1)
    mag2 = np.linalg.norm(v2)
    if mag1 * mag2 == 0:
        return 0.0
    cos_theta = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
    return float(round(math.degrees(math.acos(cos_theta)), 2))

def get_line_residual_2d(points):
    n = len(points)
    if n <= 2:
        return 0.0
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den = sum((x[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    intercept = mean_y - slope * mean_x
    residuals = [(y[i] - (slope * x[i] + intercept)) ** 2 for i in range(n)]
    rmse = math.sqrt(sum(residuals) / n)
    return float(round(rmse, 4))

def extract_ratios_from_landmarks(pts: np.ndarray, w: int = 1, h: int = 1) -> Dict[str, float]:
    """ratio_calculator.py와 100% 일치하는 14대 안면 기하 비율 계산 함수"""
    top_face = pts[10]
    chin = pts[152]
    left_face_outer = pts[234]
    right_face_outer = pts[454]
    left_forehead = pts[103]
    right_forehead = pts[332]
    left_temple = pts[54]
    right_temple = pts[284]
    left_cheekbone = pts[234]
    right_cheekbone = pts[454]
    left_jaw = pts[172]
    right_jaw = pts[397]
    left_lower_jaw = pts[150]
    right_lower_jaw = pts[379]
    nose_bridge = pts[168]
    nose_tip = pts[1]
    subnasale = pts[2]

    # 기본 기하 거리
    face_height = dist_2d(top_face, chin)
    face_width = dist_2d(left_face_outer, right_face_outer)
    forehead_width = dist_2d(left_forehead, right_forehead)
    temple_width = dist_2d(left_temple, right_temple)
    cheekbone_width = dist_2d(left_cheekbone, right_cheekbone)
    jaw_width = dist_2d(left_jaw, right_jaw)
    lower_jaw_width = dist_2d(left_lower_jaw, right_lower_jaw)

    # 1. 종횡비 및 폭 비율
    face_length_width_ratio = face_height / (face_width + 1e-6)
    forehead_to_cheekbone_ratio = forehead_width / (cheekbone_width + 1e-6)
    jaw_to_cheekbone_ratio = jaw_width / (cheekbone_width + 1e-6)
    forehead_to_jaw_ratio = forehead_width / (jaw_width + 1e-6)
    forehead_to_lower_jaw_ratio = forehead_width / (lower_jaw_width + 1e-6)
    lower_jaw_width_to_cheekbone_ratio = lower_jaw_width / (cheekbone_width + 1e-6)
    temple_to_cheekbone_ratio = temple_width / (cheekbone_width + 1e-6)

    # 2. 턱 테이퍼 및 뾰족도 (ratio_calculator 원본 공식)
    jaw_taper_ratio = lower_jaw_width / (jaw_width + 1e-6)
    jaw_angle_center_y = (left_jaw[1] + right_jaw[1]) / 2.0
    chin_to_jaw_depth = abs(chin[1] - jaw_angle_center_y)
    chin_sharpness_ratio = chin_to_jaw_depth / (lower_jaw_width + 1e-6)

    # 3. 세로 분할 비율
    lower_face_height = abs(chin[1] - nose_tip[1])
    lower_face_height_to_face_height_ratio = lower_face_height / (face_height + 1e-6)
    midface_height = dist_2d(nose_bridge, subnasale)
    midface_height_ratio = midface_height / (face_height + 1e-6)

    # 4. 각도 및 외곽선 곡률 (ratio_calculator 원본 공식)
    gonial_angle_proxy = get_angle_2d(right_cheekbone, right_jaw, chin)
    jaw_line_pts = [pts[idx] for idx in [454, 397, 365, 377, 152]]
    jawline_curvature_index = get_line_residual_2d(jaw_line_pts) / (cheekbone_width + 1e-6)

    dy = chin[1] - right_jaw[1]
    dx = chin[0] - right_jaw[0]
    jaw_slope_angle = float(round(math.degrees(math.atan2(dy, dx)), 2))

    return {
        "face_length_width_ratio": float(round(face_length_width_ratio, 4)),
        "forehead_to_cheekbone_ratio": float(round(forehead_to_cheekbone_ratio, 4)),
        "jaw_to_cheekbone_ratio": float(round(jaw_to_cheekbone_ratio, 4)),
        "forehead_to_jaw_ratio": float(round(forehead_to_jaw_ratio, 4)),
        "forehead_to_lower_jaw_ratio": float(round(forehead_to_lower_jaw_ratio, 4)),
        "lower_jaw_width_to_cheekbone_ratio": float(round(lower_jaw_width_to_cheekbone_ratio, 4)),
        "chin_sharpness_ratio": float(round(chin_sharpness_ratio, 4)),
        "jaw_taper_ratio": float(round(jaw_taper_ratio, 4)),
        "lower_face_height_to_face_height_ratio": float(round(lower_face_height_to_face_height_ratio, 4)),
        "gonial_angle_proxy": float(round(gonial_angle_proxy, 2)),
        "temple_to_cheekbone_ratio": float(round(temple_to_cheekbone_ratio, 4)),
        "jaw_slope_angle": float(round(jaw_slope_angle, 2)),
        "jawline_curvature_index": float(round(jawline_curvature_index, 4)),
        "midface_height_ratio": float(round(midface_height_ratio, 4))
    }

def generate_face_ti_code(ratios: Dict[str, float], clf_scores: Dict[str, float], rank1_cls: str) -> Dict[str, str]:
    """14대 기하 수치와 AI 분류 확률을 조합한 고유 Face-TI 4축 코드 생성"""
    # 1축: V (V라인) vs S (골격 지지형)
    v_score = (ratios["chin_sharpness_ratio"] - 0.486) * 10.0 + (clf_scores.get("heart", 0) + clf_scores.get("oval", 0) * 0.4)
    s_score = (ratios["lower_jaw_width_to_cheekbone_ratio"] - 0.549) * 10.0 + clf_scores.get("square", 0)
    c1 = "V" if v_score >= s_score else "S"

    # 2축: L (세로형/성숙) vs R (아담/동안)
    l_score = (ratios["face_length_width_ratio"] - 1.190) * 10.0 + clf_scores.get("long", 0) * 1.5
    r_score = (1.190 - ratios["face_length_width_ratio"]) * 10.0 + clf_scores.get("round", 0) * 1.5
    c2 = "L" if l_score >= r_score else "R"

    # 3축: S (부드러운 곡선) vs C (선명한 엣지)
    c_edge_score = (137.0 - ratios["gonial_angle_proxy"]) * 0.1 + (ratios["jawline_curvature_index"] - 0.125) * 20.0
    c3 = "C" if c_edge_score > 0 else "S"

    # 4축: O (황금 비례) vs M (개성 믹스)
    is_oval_dominant = clf_scores.get("oval", 0) >= 0.25 or (
        abs(ratios["face_length_width_ratio"] - 1.20) <= 0.040 and 
        abs(ratios["jaw_to_cheekbone_ratio"] - 0.784) <= 0.035
    )
    c4 = "O" if is_oval_dominant else "M"

    code = f"{c1}{c2}{c3}{c4}"

    PERSONA_TITLES = {
        "VLSO": ("샤프 엘레강스 오벌", "슬림한 V라인과 이상적인 황금 종횡비가 어우러진 정석 미모형"),
        "VLSM": ("도회적 시크 롱", "날렵한 턱 끝과 긴 세로 라인이 결합되어 세련되고 지적인 도시형"),
        "VLCO": ("모던 엣지 오벌", "선명한 턱선 엣지와 완벽한 비율로 신뢰감을 주는 이지적 페이스"),
        "VLCM": ("샤프 카리스마 롱", "또렷한 턱선과 뚜렷한 세로감으로 시선을 압도하는 카리스마형"),
        "VRSO": ("스위트 퓨어 하트", "아담한 종횡비에 갸름한 V라인 턱 끝이 더해져 상큼한 아이돌형"),
        "VRSM": ("러블리 베이비 하트", "부드러운 볼선과 앙증맞은 V라인이 결합된 사랑스러운 동안 페이스"),
        "VRCO": ("유니크 엣지 하트", "아담한 페이스라인 속 또렷한 턱선 포인트로 개성 있는 분위기"),
        "VRCM": ("트렌디 입체 하트", "입체적인 이목구비와 슬림한 턱 끝이 공존하는 트렌디형"),
        "SLSO": ("소프트 클래식 롱", "탄탄한 골격과 긴 세로 라인이 주는 차분하고 우아한 귀족상"),
        "SLSM": ("지적 모던 롱", "안정감 있는 하관과 지적인 세로 비율로 신뢰를 주는 프로페셔널형"),
        "SLCO": ("클래식 노블 스퀘어", "선명한 하악각과 우아한 세로 밸런스가 돋보이는 고급스러운 골격"),
        "SLCM": ("카리스마 스트롱 스퀘어", "남다른 존재감의 턱선과 시원한 길이감의 당당한 분위기"),
        "SRSO": ("소프트 퓨어 라운드", "부드러운 볼 라인과 편안한 인상으로 누구에게나 호감을 주는 동안"),
        "SRSM": ("내추럴 큐트 베이비", "아담한 비율과 둥근 하관이 주는 무해하고 사랑스러운 에너지"),
        "SRCO": ("클래식 모던 스퀘어", "정돈된 하악각과 콤팩트한 비율이 주는 단정하고 모던한 인상"),
        "SRCM": ("소프트 스트럭처 믹스", "탄탄한 턱선과 부드러운 볼선이 조화를 이루는 매력적인 복합 골격")
    }

    title, desc = PERSONA_TITLES.get(code, ("하모니 밸런스 페이스", "독창적인 매력과 균형감을 지닌 페이스"))
    return {"code": code, "title": title, "desc": desc}

def analyze_uploaded_image(image_bytes: bytes, user_name: str = "사용자") -> Dict[str, Any]:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("유효하지 않은 이미지 파일입니다.")

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh_detector.process(img_rgb)

    if not results.multi_face_landmarks:
        return {
            "success": False,
            "error": "이미지에서 얼굴을 감지하지 못했습니다. 얼굴이 정면을 향하도록 조정한 후 다시 시도해 주세요."
        }

    mesh = results.multi_face_landmarks[0]
    landmarks_px = np.array([[lm.x * w, lm.y * h] for lm in mesh.landmark])

    quality_report = inspect_image_quality(img_bgr, landmarks_px)
    ratios = extract_ratios_from_landmarks(landmarks_px, w, h)
    clf_res = classify_face_shape(ratios)

    scores = sorted(clf_res["score_breakdown"].items(), key=lambda x: x[1], reverse=True)
    rank1_cls = scores[0][0]
    rank2_cls = scores[1][0]
    rank3_cls = scores[2][0]

    # Face-TI 생성
    face_ti = generate_face_ti_code(ratios, clf_res["score_breakdown"], rank1_cls)

    primary_metrics = []
    sub = DF[DF['expected_label'].str.lower() == rank1_cls]

    for item in FEATURE_SCHEMA[rank1_cls]:
        f_key = item["key"]
        val = ratios[f_key]
        g_mean = GLOBAL_STATS[f_key]["mean"]
        unit = item["unit"]
        sub_series = sub[FEAT_MAP[f_key]].dropna()
        is_below = False

        if item["dir"] == "high":
            if val >= np.percentile(sub_series, 70): grade = "매우 뚜렷함 (상위 30%)"
            elif val >= np.percentile(sub_series, 50): grade = "뚜렷함 (상위 30~50%)"
            elif val >= g_mean: grade = "보통 이상 (전체 평균 초과)"
            else: grade = "참고 수치 (전체 평균 이하)"; is_below = True
            diff_str = f"+{val - g_mean:.3f}" if val >= g_mean else f"{val - g_mean:.3f}"
        elif item["dir"] == "low":
            if val <= np.percentile(sub_series, 30): grade = "매우 뚜렷함 (상위 30%)"
            elif val <= np.percentile(sub_series, 50): grade = "뚜렷함 (상위 30~50%)"
            elif val <= g_mean: grade = "보통 이상 (전체 평균 초과)"
            else: grade = "참고 수치 (전체 평균 이하)"; is_below = True
            diff_str = f"-{g_mean - val:.3f}" if val <= g_mean else f"+{val - g_mean:.3f}"
        else:
            err = abs(val - 1.20) if "1.20" in item["dir"] else abs(val - 0.784)
            sub_err = np.abs(sub_series - (1.20 if "1.20" in item["dir"] else 0.784))
            if err <= np.percentile(sub_err, 30): grade = "매우 뚜렷함 (상위 30%)"
            elif err <= np.percentile(sub_err, 50): grade = "뚜렷함 (상위 30~50%)"
            else: grade = "참고 수치 (전체 평균 이하)"; is_below = True
            diff_str = f"편차 {err:.3f}"

        primary_metrics.append({
            "name": item["name"],
            "value": round(val, 3),
            "unit": unit,
            "global_mean": round(g_mean, 3),
            "diff_str": diff_str,
            "grade": grade,
            "is_below_avg": is_below,
            "metric_meaning": item["metric_meaning"],
            "visual_meaning": "" if is_below else item["visual_meaning"]
        })

    secondary_traits = []
    for sub_cls in [rank2_cls, rank3_cls]:
        for item in FEATURE_SCHEMA[sub_cls]:
            f_key = item["key"]
            val = ratios[f_key]
            gs = GLOBAL_STATS.get(f_key)
            g_mean = gs["mean"] if gs else 0.0
            unit = item["unit"]

            if item["dir"] == "high":
                diff_str = f"+{val - g_mean:.3f}" if val >= g_mean else f"{val - g_mean:.3f}"
                if val >= gs["p70"]: grade = "매우 뚜렷함 (전체 상위 30%)"
                elif val >= gs["p50"]: grade = "뚜렷함 (전체 상위 30~50%)"
                else: continue
            elif item["dir"] == "low":
                diff_str = f"-{g_mean - val:.3f}" if val <= g_mean else f"{val - g_mean:.3f}"
                if val <= gs["p30"]: grade = "매우 뚜렷함 (전체 상위 30%)"
                elif val <= gs["p50"]: grade = "뚜렷함 (전체 상위 30~50%)"
                else: continue
            else:
                target = 1.20 if "1.20" in item["dir"] else 0.784
                err = abs(val - target)
                err_stat = GLOBAL_STATS.get("oval_lw_err" if "1.20" in item["dir"] else "oval_jaw_err", {"p30": 0.02, "p50": 0.04})
                diff_str = f"편차 {err:.3f}"
                if err <= err_stat["p30"]: grade = "매우 뚜렷함 (전체 상위 30%)"
                elif err <= err_stat["p50"]: grade = "뚜렷함 (전체 상위 30~50%)"
                else: continue

            secondary_traits.append({
                "source_shape": sub_cls.upper(),
                "name": item["name"],
                "value": round(val, 3),
                "unit": unit,
                "global_mean": round(g_mean, 3),
                "diff_str": diff_str,
                "grade": grade,
                "metric_meaning": item["metric_meaning"],
                "visual_meaning": item["visual_meaning"]
            })

    friendly_summary = f"{user_name}님의 얼굴은 <b>{clf_res['label_ko']}</b> 특징이 가장 돋보입니다. " + clf_res.get("reason", "")

    return {
        "success": True,
        "user_name": user_name,
        "face_ti": face_ti,
        "quality": quality_report,
        "classification": {
            "rank1": {"type": rank1_cls.upper(), "label_ko": clf_res["label_ko"], "score": round(scores[0][1]*100, 1)},
            "rank2": {"type": rank2_cls.upper(), "score": round(scores[1][1]*100, 1)},
            "rank3": {"type": rank3_cls.upper(), "score": round(scores[2][1]*100, 1)},
            "all_scores": {c.upper(): round(s*100, 1) for c, s in scores}
        },
        "primary_analysis": primary_metrics,
        "secondary_traits": secondary_traits,
        "friendly_summary": friendly_summary
    }

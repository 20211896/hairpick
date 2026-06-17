from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import mediapipe as mp
import math

app = FastAPI(title="Face Analysis AI Backend")

mp_face_mesh = mp.solutions.face_mesh


def distance(p1: dict, p2: dict) -> float:
    """
    두 랜드마크 좌표 사이의 2D 거리 계산
    """
    return math.sqrt((p1["x"] - p2["x"]) ** 2 + (p1["y"] - p2["y"]) ** 2)


def midpoint(p1: dict, p2: dict) -> dict:
    """
    두 점의 중점 계산
    """
    return {
        "x": (p1["x"] + p2["x"]) / 2,
        "y": (p1["y"] + p2["y"]) / 2
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    """
    0으로 나누는 오류 방지용 비율 계산
    """
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    """
    점수를 0~1 범위로 제한
    """
    return max(min_value, min(value, max_value))


def score_range(value: float, center: float, tolerance: float) -> float:
    """
    value가 center에 가까울수록 높은 점수
    """
    return clamp(1 - abs(value - center) / tolerance)


def score_above(value: float, threshold: float, spread: float) -> float:
    """
    value가 threshold보다 클수록 높은 점수
    """
    return clamp((value - threshold) / spread)


def score_below(value: float, threshold: float, spread: float) -> float:
    """
    value가 threshold보다 작을수록 높은 점수
    """
    return clamp((threshold - value) / spread)


def classify_face_shape(ratios: dict) -> dict:
    """
    가중치 기반 얼굴형 분류 함수 v4.

    핵심 방향:
    - oval은 기본형/fallback으로 낮게 둔다.
    - round, long, heart, diamond, square의 고유 특징이 강하면 oval보다 우선한다.
    - 1등과 2등 후보를 함께 반환한다.
    """

    length_width = ratios["face_length_width_ratio"]
    forehead_to_cheekbone = ratios["forehead_to_cheekbone_ratio"]
    jaw_to_cheekbone = ratios["jaw_to_cheekbone_ratio"]

    lower_jaw_to_cheekbone = ratios.get(
        "lower_jaw_width_to_cheekbone_ratio",
        jaw_to_cheekbone
    )

    jaw_taper_ratio = ratios.get(
        "jaw_taper_ratio",
        0.70
    )

    chin_depth_ratio = ratios.get(
        "chin_to_jaw_depth_to_face_height_ratio",
        0.22
    )

    lower_face_ratio = ratios.get(
        "lower_face_height_to_face_height_ratio",
        0.40
    )

    # -----------------------------
    # 공통 특징 점수
    # -----------------------------

    shortness = score_below(length_width, 1.17, 0.12)
    longness = score_above(length_width, 1.22, 0.18)
    very_long = score_above(length_width, 1.30, 0.10)

    jaw_narrowness = score_below(jaw_to_cheekbone, 0.76, 0.16)
    jaw_broadness = score_above(jaw_to_cheekbone, 0.79, 0.16)

    lower_jaw_narrowness = score_below(lower_jaw_to_cheekbone, 0.54, 0.12)
    lower_jaw_broadness = score_above(lower_jaw_to_cheekbone, 0.55, 0.12)

    forehead_narrowness = score_below(forehead_to_cheekbone, 0.72, 0.12)
    forehead_broadness = score_above(forehead_to_cheekbone, 0.74, 0.12)

    taper_strong = score_below(jaw_taper_ratio, 0.69, 0.10)
    taper_weak = score_above(jaw_taper_ratio, 0.70, 0.12)

    cheekbone_dominance = clamp(
        ((1 - forehead_to_cheekbone) * 0.40) +
        ((1 - jaw_to_cheekbone) * 0.35) +
        ((1 - lower_jaw_to_cheekbone) * 0.25)
    )

    # -----------------------------
    # 1. 둥근형
    # -----------------------------
    # 둥근형은 얼굴 세로/가로 차이가 작고, 턱이 지나치게 뾰족하지 않은 경우
    round_score = (
        shortness * 0.55 +
        jaw_broadness * 0.20 +
        score_below(lower_face_ratio, 0.47, 0.15) * 0.15 +
        score_range(jaw_taper_ratio, 0.70, 0.16) * 0.10
    )

    # -----------------------------
    # 2. 사각형
    # -----------------------------
    # 사각형은 턱 폭과 아래턱 폭이 유지되고, 턱선이 급격히 좁아지지 않는 경우
    square_score = (
        jaw_broadness * 0.35 +
        lower_jaw_broadness * 0.30 +
        taper_weak * 0.25 +
        score_below(length_width, 1.24, 0.16) * 0.10
    )

    # 둥근형/사각형 직접 비교 보정
    # sample2처럼 하관이 부드럽게 좁아지면 둥근형,
    # sample4처럼 아래턱 폭과 taper가 유지되면 사각형으로 보정한다.
    if length_width <= 1.17:
        if lower_jaw_to_cheekbone >= 0.57 and jaw_taper_ratio >= 0.705:
            square_score += 0.22
            round_score -= 0.08
        else:
            round_score += 0.12

    # -----------------------------
    # 3. 긴형
    # -----------------------------
    # 긴형은 세로/가로 비율을 가장 강하게 반영
    long_score = (
        longness * 0.55 +
        very_long * 0.35 +
        score_above(lower_face_ratio, 0.39, 0.12) * 0.10
    )

    # 광대 우세가 강하면 긴형에서 약간 감점
    long_score -= cheekbone_dominance * 0.06
    long_score = clamp(long_score)

    # L/W가 1.32 이상이면 긴형을 강하게 보정
    if length_width >= 1.32:
        long_score += 0.18

    long_score = clamp(long_score)

    # -----------------------------
    # 4. 하트형
    # -----------------------------
    # 하트형은 턱과 아래턱이 좁고, 이마/상단 폭이 상대적으로 유지되는 경우
    heart_score = (
        jaw_narrowness * 0.25 +
        lower_jaw_narrowness * 0.25 +
        taper_strong * 0.20 +
        forehead_broadness * 0.25 +
        score_below(length_width, 1.28, 0.18) * 0.05
    )

    # 하트형 직접 보정:
    # 턱이 좁고, 아래턱도 좁고, 이마 폭이 비교적 유지되면 하트형 강화
    if jaw_to_cheekbone <= 0.75 and lower_jaw_to_cheekbone <= 0.52:
        if forehead_to_cheekbone >= 0.745:
            heart_score += 0.22
        elif forehead_to_cheekbone >= 0.73:
            heart_score += 0.12

    heart_score = clamp(heart_score)

    # -----------------------------
    # 5. 마름모형
    # -----------------------------
    # 마름모형은 광대가 두드러지고, 이마와 턱이 모두 좁은 경우
    diamond_score = (
        cheekbone_dominance * 0.30 +
        forehead_narrowness * 0.30 +
        jaw_narrowness * 0.18 +
        lower_jaw_narrowness * 0.17 +
        score_above(length_width, 1.17, 0.18) * 0.05
    )

    # 마름모형 직접 보정:
    # 하트형과 유사하지만 이마 폭이 충분히 넓지 않은 경우는 마름모형으로 보정
    if (
        forehead_to_cheekbone < 0.755
        and jaw_to_cheekbone <= 0.77
        and lower_jaw_to_cheekbone <= 0.53
    ):
        diamond_score += 0.26
        heart_score = max(0.0, heart_score - 0.08)

    # -----------------------------
    # 6. 계란형 또는 일반형
    # -----------------------------
    # oval은 fallback. 다른 특징형이 강하면 강하게 감점한다.
    oval_score = (
        score_range(length_width, 1.18, 0.16) * 0.22 +
        score_range(jaw_to_cheekbone, 0.79, 0.14) * 0.18 +
        score_range(forehead_to_cheekbone, 0.72, 0.14) * 0.16 +
        score_range(lower_jaw_to_cheekbone, 0.54, 0.12) * 0.14 +
        score_range(jaw_taper_ratio, 0.69, 0.12) * 0.14 +
        score_range(lower_face_ratio, 0.40, 0.12) * 0.16
    )

    special_strength = max(
        round_score,
        square_score,
        long_score,
        heart_score,
        diamond_score
    )

    # 특징형이 충분히 강하면 oval을 fallback으로 밀어냄
    if special_strength >= 0.60:
        oval_score -= 0.35
    elif special_strength >= 0.50:
        oval_score -= 0.25
    elif special_strength >= 0.40:
        oval_score -= 0.15
    elif special_strength >= 0.32:
        oval_score -= 0.08

    oval_score = clamp(oval_score)

    # -----------------------------
    # 추가 직접 보정
    # -----------------------------

    # 긴형 직접 보정: L/W가 1.33 이상이면 oval보다 긴형 우선
    if length_width >= 1.33 and long_score < oval_score:
        long_score = min(1.0, oval_score + 0.05)

    # 사각형 직접 보정:
    # L/W가 낮고, 아래턱 폭이 유지되며, 턱선이 덜 좁아지면 사각형 우선
    if length_width <= 1.15 and lower_jaw_to_cheekbone >= 0.57 and jaw_taper_ratio >= 0.705:
        if square_score < oval_score:
            square_score = min(1.0, oval_score + 0.05)

    # 둥근형 직접 보정:
    # L/W가 낮고 jaw가 넓지만, 아래턱 폭이 사각형 기준보다 약하면 둥근형 우선
    elif length_width <= 1.15 and jaw_to_cheekbone >= 0.79:
        if round_score < oval_score:
            round_score = min(1.0, oval_score + 0.04)

    # 하트형 직접 보정:
    # 턱과 아래턱이 좁더라도, 이마/상단 폭이 충분히 유지될 때만 하트형 강화
    if jaw_to_cheekbone <= 0.75 and lower_jaw_to_cheekbone <= 0.52:
        if forehead_to_cheekbone >= 0.765:
            heart_score += 0.22
        elif forehead_to_cheekbone >= 0.755:
            heart_score += 0.10

    # 마름모형 직접 보정:
    # 턱과 아래턱이 좁고, 이마가 하트형 기준만큼 넓지 않으면 마름모형 우선
    if (
        forehead_to_cheekbone < 0.755
        and jaw_to_cheekbone <= 0.77
        and lower_jaw_to_cheekbone <= 0.53
    ):
        if diamond_score < oval_score:
            diamond_score = min(1.0, oval_score + 0.06)

        # 계란형/하트형으로 흡수되는 것 방지
        oval_score = max(0.0, oval_score - 0.08)
        heart_score = max(0.0, heart_score - 0.04)

    # -----------------------------
    # 최종 점수 정리
    # -----------------------------

    scores = {
        "round": round(clamp(round_score), 4),
        "oval": round(clamp(oval_score), 4),
        "long": round(clamp(long_score), 4),
        "square": round(clamp(square_score), 4),
        "heart": round(clamp(heart_score), 4),
        "diamond": round(clamp(diamond_score), 4)
    }

    labels = {
        "round": "둥근형",
        "oval": "계란형 또는 일반형",
        "long": "긴형",
        "square": "사각형",
        "heart": "하트형",
        "diamond": "마름모형"
    }

    reasons = {
        "round": "얼굴 세로와 가로 비율 차이가 작고 하관이 비교적 부드러워 둥근형 점수가 가장 높습니다.",
        "oval": "특정 얼굴형 특징이 과하게 두드러지지 않고 전체 비율이 비교적 자연스러워 계란형 또는 일반형 점수가 가장 높습니다.",
        "long": "얼굴 세로 길이 비중이 가장 크게 나타나 긴형 점수가 가장 높습니다.",
        "square": "턱 폭과 아래턱 폭이 비교적 유지되어 사각형 점수가 가장 높습니다.",
        "heart": "턱과 아래턱은 좁지만 상단 폭이 상대적으로 유지되어 하트형 점수가 가장 높습니다.",
        "diamond": "광대 폭이 상대적으로 강조되고 이마와 턱 폭이 좁아 마름모형 점수가 가장 높습니다."
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_type, best_score = sorted_scores[0]
    second_type, second_score = sorted_scores[1]

    confidence = clamp(0.45 + (best_score - second_score) * 0.8)

    return {
        "type": best_type,
        "label_ko": labels[best_type],
        "confidence": round(confidence, 4),
        "reason": reasons[best_type],
        "score_breakdown": scores,
        "second_candidate": {
            "type": second_type,
            "label_ko": labels[second_type],
            "score": second_score
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Face Analysis AI Backend is running"
    }


@app.post("/api/analyze-face")
async def analyze_face(file: UploadFile = File(...)):
    # 1. 파일 형식 검사
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="JPG, JPEG, PNG 형식의 이미지만 업로드할 수 있습니다."
        )

    # 2. 파일 읽기
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="빈 파일입니다."
        )

    # 3. OpenCV 이미지 디코딩
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="이미지를 읽을 수 없습니다."
        )

    height, width, channels = image_bgr.shape

    # 4. MediaPipe 입력용 RGB 변환
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 5. Face Mesh 실행
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    # 6. 얼굴 미검출 처리
    if not results.multi_face_landmarks:
        return {
            "success": False,
            "face_detected": False,
            "face_count": 0,
            "filename": file.filename,
            "image_size": {
                "width": width,
                "height": height,
                "channels": channels
            },
            "message": "이미지에서 얼굴을 찾을 수 없습니다."
        }

    face_count = len(results.multi_face_landmarks)

    # 7. 여러 얼굴 감지 처리
    if face_count > 1:
        return {
            "success": False,
            "face_detected": True,
            "face_count": face_count,
            "filename": file.filename,
            "image_size": {
                "width": width,
                "height": height,
                "channels": channels
            },
            "message": "여러 명의 얼굴이 감지되었습니다. 한 명의 얼굴만 포함된 이미지를 사용해주세요."
        }

    # 8. 첫 번째 얼굴의 랜드마크 추출
    face_landmarks = results.multi_face_landmarks[0]

    landmarks = []
    for idx, landmark in enumerate(face_landmarks.landmark):
        x_px = int(landmark.x * width)
        y_px = int(landmark.y * height)
        z = float(landmark.z)

        landmarks.append({
            "index": idx,
            "x": x_px,
            "y": y_px,
            "z": z
        })

    # 9. 주요 랜드마크 추출
    key_landmarks = {
        # 얼굴 세로 기준
        "top_face": landmarks[10],
        "chin": landmarks[152],

        # 얼굴 전체 너비 기준
        "left_face_outer": landmarks[234],
        "right_face_outer": landmarks[454],

        # 이마 폭 추정 기준
        "left_forehead": landmarks[103],
        "right_forehead": landmarks[332],

        # 광대 폭 추정 기준
        "left_cheekbone": landmarks[234],
        "right_cheekbone": landmarks[454],

        # 턱 폭 추정 기준
        "left_jaw": landmarks[172],
        "right_jaw": landmarks[397],

        # 턱선 형태 분석 기준
        "left_jaw_angle": landmarks[172],
        "right_jaw_angle": landmarks[397],
        "left_lower_jaw": landmarks[150],
        "right_lower_jaw": landmarks[379],

        # 눈 기준
        "left_eye_outer": landmarks[33],
        "right_eye_outer": landmarks[263],
        "left_eye_inner": landmarks[133],
        "right_eye_inner": landmarks[362],

        # 코 기준
        "nose_bridge": landmarks[168],
        "nose_tip": landmarks[1],

        # 입 기준
        "mouth_left": landmarks[61],
        "mouth_right": landmarks[291],
        "mouth_top": landmarks[13],
        "mouth_bottom": landmarks[14]
    }

    # 10. 비율 계산
    top_face = key_landmarks["top_face"]
    chin = key_landmarks["chin"]

    left_face_outer = key_landmarks["left_face_outer"]
    right_face_outer = key_landmarks["right_face_outer"]

    left_forehead = key_landmarks["left_forehead"]
    right_forehead = key_landmarks["right_forehead"]

    left_cheekbone = key_landmarks["left_cheekbone"]
    right_cheekbone = key_landmarks["right_cheekbone"]

    left_jaw = key_landmarks["left_jaw"]
    right_jaw = key_landmarks["right_jaw"]

    left_jaw_angle = key_landmarks["left_jaw_angle"]
    right_jaw_angle = key_landmarks["right_jaw_angle"]
    left_lower_jaw = key_landmarks["left_lower_jaw"]
    right_lower_jaw = key_landmarks["right_lower_jaw"]

    left_eye_outer = key_landmarks["left_eye_outer"]
    right_eye_outer = key_landmarks["right_eye_outer"]
    left_eye_inner = key_landmarks["left_eye_inner"]
    right_eye_inner = key_landmarks["right_eye_inner"]

    nose_bridge = key_landmarks["nose_bridge"]
    nose_tip = key_landmarks["nose_tip"]

    mouth_left = key_landmarks["mouth_left"]
    mouth_right = key_landmarks["mouth_right"]
    mouth_top = key_landmarks["mouth_top"]
    mouth_bottom = key_landmarks["mouth_bottom"]

    face_height = distance(top_face, chin)
    face_width = distance(left_face_outer, right_face_outer)
    face_length_width_ratio = safe_ratio(face_height, face_width)

    forehead_width = distance(left_forehead, right_forehead)
    cheekbone_width = distance(left_cheekbone, right_cheekbone)
    jaw_width = distance(left_jaw, right_jaw)

    lower_jaw_width = distance(left_lower_jaw, right_lower_jaw)
    jaw_taper_ratio = safe_ratio(lower_jaw_width, jaw_width)

    jaw_angle_width = distance(left_jaw_angle, right_jaw_angle)

    jaw_angle_center = midpoint(left_jaw_angle, right_jaw_angle)
    lower_jaw_center = midpoint(left_lower_jaw, right_lower_jaw)

    chin_to_jaw_depth = abs(chin["y"] - jaw_angle_center["y"])
    lower_face_height = abs(chin["y"] - nose_tip["y"])

    eye_outer_width = distance(left_eye_outer, right_eye_outer)
    eye_inner_distance = distance(left_eye_inner, right_eye_inner)

    nose_length = distance(nose_bridge, nose_tip)
    mouth_width = distance(mouth_left, mouth_right)
    mouth_height = distance(mouth_top, mouth_bottom)

    face_center = midpoint(left_face_outer, right_face_outer)
    eye_center = midpoint(left_eye_outer, right_eye_outer)
    mouth_center = midpoint(mouth_left, mouth_right)

    center_offset_x = round(abs(face_center["x"] - nose_tip["x"]), 2)

    ratios = {
        "face_height_px": round(face_height, 2),
        "face_width_px": round(face_width, 2),
        "face_length_width_ratio": face_length_width_ratio,

        "forehead_width_px": round(forehead_width, 2),
        "cheekbone_width_px": round(cheekbone_width, 2),
        "jaw_width_px": round(jaw_width, 2),
        "forehead_to_cheekbone_ratio": safe_ratio(forehead_width, cheekbone_width),
        "jaw_to_cheekbone_ratio": safe_ratio(jaw_width, cheekbone_width),

        "jaw_angle_width_px": round(jaw_angle_width, 2),
        "jaw_angle_width_to_cheekbone_ratio": safe_ratio(jaw_angle_width, cheekbone_width),
        "chin_to_jaw_depth_px": round(chin_to_jaw_depth, 2),
        "chin_to_jaw_depth_to_face_height_ratio": safe_ratio(chin_to_jaw_depth, face_height),
        "lower_face_height_px": round(lower_face_height, 2),
        "lower_face_height_to_face_height_ratio": safe_ratio(lower_face_height, face_height),

        "eye_outer_width_px": round(eye_outer_width, 2),
        "eye_inner_distance_px": round(eye_inner_distance, 2),
        "eye_outer_width_to_face_width_ratio": safe_ratio(eye_outer_width, face_width),
        "eye_inner_distance_to_face_width_ratio": safe_ratio(eye_inner_distance, face_width),

        "lower_jaw_width_px": round(lower_jaw_width, 2),
        "lower_jaw_width_to_cheekbone_ratio": safe_ratio(lower_jaw_width, cheekbone_width),
        "jaw_taper_ratio": jaw_taper_ratio,

        "nose_length_px": round(nose_length, 2),
        "nose_length_to_face_height_ratio": safe_ratio(nose_length, face_height),

        "mouth_width_px": round(mouth_width, 2),
        "mouth_height_px": round(mouth_height, 2),
        "mouth_width_to_face_width_ratio": safe_ratio(mouth_width, face_width),

        "nose_center_offset_x_px": center_offset_x
    }

    centers = {
        "face_center": face_center,
        "eye_center": eye_center,
        "mouth_center": mouth_center,
        "jaw_angle_center": jaw_angle_center,
        "lower_jaw_center": lower_jaw_center
    }

    face_shape = classify_face_shape(ratios)

    return {
        "success": True,
        "face_detected": True,
        "face_count": face_count,
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size_bytes": len(file_bytes),
        "image_size": {
            "width": width,
            "height": height,
            "channels": channels
        },
        "landmark_count": len(landmarks),
        "key_landmarks": key_landmarks,
        "ratios": ratios,
        "centers": centers,
        "face_shape": face_shape,
        "message": "얼굴 랜드마크, 비율 계산, 가중치 기반 얼굴형 분류 성공"
    }
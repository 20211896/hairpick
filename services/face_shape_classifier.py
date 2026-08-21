def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(value, max_value))


def score_range(value: float, center: float, tolerance: float) -> float:
    return clamp(1 - abs(value - center) / tolerance)


def score_above(value: float, threshold: float, spread: float) -> float:
    return clamp((value - threshold) / spread)


def score_below(value: float, threshold: float, spread: float) -> float:
    return clamp((threshold - value) / spread)


def classify_face_shape(ratios: dict) -> dict:
    """
    고정밀 얼굴형 분류기 v27
    - Oval Fallback 억제 완벽 복원 (Oval 블랙홀 원천 차단)
    - 신규 기하 피처(귀밑턱 각도, 관자놀이, 하관 경사각) 기반 클래스별 가중치/보너스 최적화
    """

    # 1. 기본 비율 로드
    length_width = ratios["face_length_width_ratio"]
    forehead_to_cheekbone = ratios["forehead_to_cheekbone_ratio"]
    jaw_to_cheekbone = ratios["jaw_to_cheekbone_ratio"]

    forehead_to_jaw_ratio = ratios.get("forehead_to_jaw_ratio", 0)
    forehead_to_lower_jaw_ratio = ratios.get("forehead_to_lower_jaw_ratio", 0)
    jaw_to_face_width_ratio = ratios.get("jaw_to_face_width_ratio", 0)
    lower_jaw_to_face_width_ratio = ratios.get("lower_jaw_to_face_width_ratio", 0)
    chin_sharpness_ratio = ratios.get("chin_sharpness_ratio", 0)

    lower_jaw_to_cheekbone = ratios.get(
        "lower_jaw_width_to_cheekbone_ratio",
        jaw_to_cheekbone
    )

    jaw_taper_ratio = ratios.get("jaw_taper_ratio", 0.70)
    lower_face_ratio = ratios.get("lower_face_height_to_face_height_ratio", 0.40)

    # 2. 신규 정밀 피처 로드 (실측 기반)
    midface_ratio = ratios.get("midface_height_ratio", 0.308)
    philtrum_to_chin = ratios.get("philtrum_to_chin_ratio", 0.235)
    temple_to_cheekbone = ratios.get("temple_to_cheekbone_ratio", 0.865)
    gonial_angle = ratios.get("gonial_angle_proxy", 137.0)
    jaw_slope = ratios.get("jaw_slope_angle", 145.0)

    # -----------------------------
    # 공통 특징 점수
    # -----------------------------
    shortness = score_below(length_width, 1.20, 0.12)
    longness = score_above(length_width, 1.18, 0.12)
    very_long = score_above(length_width, 1.24, 0.10)

    jaw_narrowness = score_below(jaw_to_cheekbone, 0.78, 0.16)
    jaw_broadness = score_above(jaw_to_cheekbone, 0.780, 0.08)

    lower_jaw_narrowness = score_below(lower_jaw_to_cheekbone, 0.555, 0.12)
    lower_jaw_broadness = score_above(lower_jaw_to_cheekbone, 0.546, 0.08)

    forehead_narrowness = score_below(forehead_to_cheekbone, 0.72, 0.12)
    forehead_broadness = score_above(forehead_to_cheekbone, 0.74, 0.12)

    taper_strong = score_below(jaw_taper_ratio, 0.69, 0.10)
    taper_weak = score_above(jaw_taper_ratio, 0.680, 0.08)

    # -----------------------------
    # 1. 둥근형 (Round)
    # -----------------------------
    round_score = (
        shortness * 0.50 +
        jaw_broadness * 0.20 +
        score_below(lower_face_ratio, 0.47, 0.15) * 0.15 +
        score_range(jaw_taper_ratio, 0.70, 0.16) * 0.10 +
        score_above(gonial_angle, 134.5, 4.0) * 0.05
    )

    if length_width <= 1.18 and lower_jaw_to_cheekbone < 0.552:
        round_score += 0.22

    # -----------------------------
    # 2. 사각형 (Square)
    # -----------------------------
    square_score = (
        jaw_broadness * 0.34 +
        lower_jaw_broadness * 0.34 +
        taper_weak * 0.20 +
        score_below(length_width, 1.22, 0.16) * 0.10 +
        score_below(gonial_angle, 134.5, 4.0) * 0.10
    )

    if length_width <= 1.23:
        if lower_jaw_to_cheekbone >= 0.554 and jaw_to_cheekbone >= 0.791 and gonial_angle <= 135.0:
            square_score += 0.27
            if round_score > square_score:
                round_score -= 0.10
        elif lower_jaw_to_cheekbone >= 0.550 and jaw_to_cheekbone >= 0.788 and jaw_taper_ratio >= 0.695:
            square_score += 0.18

    # -----------------------------
    # 3. 긴형 (Long)
    # -----------------------------
    long_score = (
        longness * 0.55 +
        very_long * 0.30 +
        score_above(lower_face_ratio, 0.39, 0.12) * 0.10 +
        score_above(midface_ratio, 0.310, 0.02) * 0.08
    )

    if length_width >= 1.26:
        long_score += 0.24
    elif length_width >= 1.23:
        long_score += 0.18
    elif length_width >= 1.21:
        long_score += 0.12

    long_score = clamp(long_score)

    # -----------------------------
    # 4. 하트형 (Heart)
    # -----------------------------
    heart_score = (
        jaw_narrowness * 0.22 +
        lower_jaw_narrowness * 0.22 +
        taper_strong * 0.20 +
        forehead_broadness * 0.22 +
        score_above(temple_to_cheekbone, 0.872, 0.02) * 0.14 +
        score_below(jaw_slope, 144.5, 4.0) * 0.05
    )

    heart_score = clamp(heart_score)

    # -----------------------------
    # 5. 계란형 (Oval) - Fallback 전용
    # -----------------------------
    oval_score = (
        score_range(length_width, 1.19, 0.14) * 0.22 +
        score_range(jaw_to_cheekbone, 0.79, 0.12) * 0.18 +
        score_range(forehead_to_cheekbone, 0.73, 0.12) * 0.16 +
        score_range(lower_jaw_to_cheekbone, 0.54, 0.10) * 0.14 +
        score_range(jaw_taper_ratio, 0.69, 0.10) * 0.14 +
        score_range(lower_face_ratio, 0.40, 0.10) * 0.16
    )

    # [핵심] 타 클래스 특화 점수가 감지되면 Oval 점수를 강력히 억제
    special_strength = max(round_score, square_score, long_score, heart_score)
    if special_strength >= 0.45:
        oval_score -= 0.45
    elif special_strength >= 0.35:
        oval_score -= 0.35
    elif special_strength >= 0.25:
        oval_score -= 0.25

    oval_score = clamp(oval_score)

    # =========================================================
    # [Sharpness + 골격 복합 보정 룰]
    # =========================================================
    # 1. 고선명 V라인: Sharpness 0.550 이상
    if chin_sharpness_ratio >= 0.550 and length_width <= 1.22:
        square_score *= 0.40
        if heart_score < square_score:
            heart_score = square_score + 0.08

    # 2. 이마/관자놀이 대비 아래턱 수축 우수 샘플
    if (temple_to_cheekbone >= 0.872 or forehead_to_lower_jaw_ratio >= 1.24) and chin_sharpness_ratio >= 0.420 and length_width <= 1.22:
        square_score *= 0.50
        if heart_score < square_score:
            heart_score = square_score + 0.06

    # 3. Long(긴형) 예외 처리: Sharpness가 높고 세로가 길지 않은 경우
    if chin_sharpness_ratio >= 0.465 and length_width < 1.24:
        long_score *= 0.65

    # 4. Oval(계란형) Fallback 유입 방지
    if chin_sharpness_ratio >= 0.470 and heart_score < oval_score:
        heart_score = oval_score + 0.08

    heart_score = clamp(heart_score)
    long_score = clamp(long_score)
    square_score = clamp(square_score)
    round_score = clamp(round_score)

    # -----------------------------
    # 최우선 직접 오버라이드 보정
    # -----------------------------
    # Heart 보정
    if (
        chin_sharpness_ratio >= 0.485
        and (forehead_to_lower_jaw_ratio >= 1.24 or temple_to_cheekbone >= 0.872)
        and jaw_to_face_width_ratio <= 0.795
        and lower_jaw_to_face_width_ratio <= 0.555
        and 1.10 <= length_width <= 1.25
        and long_score < 0.50
        and square_score < 0.55
    ):
        heart_score = min(1.0, max(heart_score, oval_score + 0.06, round_score + 0.04))

    # Round vs Square 오버라이드 (귀밑턱 각도 기반 정밀 분기)
    if length_width <= 1.17 and lower_jaw_to_cheekbone < 0.552:
        if gonial_angle >= 134.5 and chin_sharpness_ratio < 0.465:
            if round_score < oval_score:
                round_score = min(1.0, oval_score + 0.06)
            if round_score < square_score and jaw_taper_ratio < 0.695:
                round_score = min(1.0, square_score + 0.04)

    scores = {
        "round": round(clamp(round_score), 4),
        "oval": round(clamp(oval_score), 4),
        "long": round(clamp(long_score), 4),
        "square": round(clamp(square_score), 4),
        "heart": round(clamp(heart_score), 4)
    }

    labels = {
        "round": "둥근형",
        "oval": "계란형 또는 일반형",
        "long": "긴형",
        "square": "사각형",
        "heart": "하트형"
    }

    reasons = {
        "round": "얼굴 세로와 가로 비율 차이가 작고 하관이 비교적 부드러워 둥근형 점수가 가장 높습니다.",
        "oval": "특정 얼굴형 특징이 과하게 두드러지지 않고 전체 비율이 비교적 자연스러운 계란형입니다.",
        "long": "얼굴 세로 길이 비중이 크게 나타나 긴형 점수가 가장 높습니다.",
        "square": "턱 폭과 아래턱 폭이 비교적 유지되어 사각형 점수가 가장 높습니다.",
        "heart": "턱과 아래턱은 좁지만 상단 폭이 상대적으로 유지되어 하트형 점수가 가장 높습니다."
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_type, best_score = sorted_scores[0]
    second_type, second_score = sorted_scores[1]

    classification_margin = round(best_score - second_score, 4)
    confidence = clamp(0.45 + classification_margin * 0.8)

    if classification_margin < 0.05:
        confidence_level = "low"
        is_borderline = True
    elif classification_margin < 0.15:
        confidence_level = "medium"
        is_borderline = False
    else:
        confidence_level = "high"
        is_borderline = False

    return {
        "type": best_type,
        "label_ko": labels[best_type],
        "confidence": round(confidence, 4),
        "confidence_level": confidence_level,
        "classification_margin": classification_margin,
        "is_borderline": is_borderline,
        "reason": reasons[best_type],
        "score_breakdown": scores,
        "second_candidate": {
            "type": second_type,
            "label_ko": labels[second_type],
            "score": second_score
        }
    }
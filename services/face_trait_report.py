from typing import Any, Dict, List, Mapping

BASE_FRAME_INFO = {
    "heart": {
        "title": "하트형 (Heart Base)",
        "desc": "상안부 폭이 시원하고 턱 끝으로 갈수록 슬림해지는 역삼각형 골격"
    },
    "long": {
        "title": "긴형 (Long Base)",
        "desc": "얼굴의 세로 라인이 시원하게 뻗어 지적이고 우아한 무드를 자아내는 골격"
    },
    "oval": {
        "title": "계란형 (Oval Base)",
        "desc": "가로·세로 비율과 윤곽선이 조화롭고 균형 잡힌 황금 비례 골격"
    },
    "round": {
        "title": "둥근형 (Round Base)",
        "desc": "가로와 세로 비율이 균형을 이루며 볼선과 하관이 부드러운 동안 골격"
    },
    "square": {
        "title": "각진형 (Square Base)",
        "desc": "반듯한 하관과 정돈된 턱 라인으로 세련되고 신뢰감을 주는 골격"
    }
}

CLASS_TRAIT_MAP = {
    "heart": {
        "trait_id": "sharp_chin",
        "name": "날렵한 V라인 턱 끝",
        "category": "하관/턱선",
        "desc": "턱 끝이 갸름하게 모여 세련되고 도회적인 인상",
        "tip": "턱선을 가리지 않고 드러내는 태슬컷 / 굵은 S컬 웨이브"
    },
    "round": {
        "trait_id": "soft_jawline",
        "name": "부드러운 곡선형 볼/하관",
        "category": "페이스라인",
        "desc": "턱선 곡선이 완만하여 친근하고 사랑스러운 분위기",
        "tip": "정수리 볼륨을 살린 레이어드 컷 / 내추럴 가르마"
    },
    "square": {
        "trait_id": "structured_jaw",
        "name": "또렷한 턱 골격 라인",
        "category": "골격/윤곽",
        "desc": "귀밑턱 각이 살아있어 고급스럽고 이지적인 인상",
        "tip": "턱선을 부드럽게 감싸는 소프트 레이어드 C컬"
    },
    "long": {
        "trait_id": "slender_vertical",
        "name": "시원한 세로 비율",
        "category": "비율/밸런스",
        "desc": "세로 길이가 돋보여 차분하고 성숙한 분위기",
        "tip": "사이드 뱅 / 시스루 뱅으로 상안부 면적 조절"
    },
    "oval": {
        "trait_id": "balanced_ratio",
        "name": "조화로운 이목구비 밸런스",
        "category": "전체비율",
        "desc": "특정 부위 치우침 없이 부드러운 연결감을 주는 비례",
        "tip": "과도한 커버 없이 본연의 윤곽선을 살리는 내추럴 스타일링"
    }
}

def verify_class_trait(cls: str, ratios: Mapping[str, float]) -> bool:
    """2위, 3위 클래스에 해당하는 실제 기하 피처 조건 검증"""
    lw = ratios.get("face_length_width_ratio", 1.20)
    sharp = ratios.get("chin_sharpness_ratio", 0.50)
    gonial = ratios.get("gonial_angle_proxy", 135.0)
    jaw_face = ratios.get("jaw_to_cheekbone_ratio", 0.79)
    flj = ratios.get("forehead_to_lower_jaw_ratio", 1.33)
    curvature = ratios.get("jawline_curvature_index", 0.125)

    if cls == "heart":
        return bool(sharp >= 0.490 or flj >= 1.340)
    elif cls == "round":
        return bool(curvature <= 0.126 or gonial >= 134.0)
    elif cls == "square":
        return bool(gonial <= 135.0 or jaw_face >= 0.792)
    elif cls == "long":
        return bool(lw >= 1.195)
    elif cls == "oval":
        return True
    return False

def build_trait_diagnosis_report(classification_result: Mapping[str, Any], raw_ratios: Mapping[str, float]) -> Dict[str, Any]:
    best_type = classification_result.get("type", "oval")
    scores = classification_result.get("score_breakdown", {})
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # 2위와 3위 후보 추출
    sub_candidates = [cls for cls, _ in sorted_scores[1:3]]
    
    # 2위/3위 중 실제 조건을 만족하는 특징만 선별 (미보유는 원천 제외)
    detected_traits = []
    for cand in sub_candidates:
        if verify_class_trait(cand, raw_ratios):
            trait_data = dict(CLASS_TRAIT_MAP[cand])
            trait_data["rank_source"] = f"{scores[cand]*100:.1f}% ({cand.upper()})"
            detected_traits.append(trait_data)

    base_info = BASE_FRAME_INFO.get(best_type, BASE_FRAME_INFO["oval"])
    base_name = base_info["title"].split(" ")[0]

    # 맞춤 총평 문구 조합
    if detected_traits:
        trait_names = [f"'{t['name']}'" for t in detected_traits]
        summary_text = f"기본 베이스는 **{base_name}**이며, 2·3순위 특징인 **{', '.join(trait_names)}** 요소를 함께 지니고 있어 복합적인 매력을 보입니다."
    else:
        summary_text = f"기본 베이스인 **{base_name}** 고유의 전형적인 골격 밸런스가 매우 뚜렷한 타입입니다."

    tips = [CLASS_TRAIT_MAP[best_type]["tip"]]
    for t in detected_traits:
        if t["tip"] not in tips:
            tips.append(t["tip"])

    return {
        "base_frame": {
            "type": best_type,
            "title": base_info["title"],
            "description": base_info["desc"]
        },
        "summary": summary_text,
        "active_traits": detected_traits,
        "active_traits_count": len(detected_traits),
        "styling_recommendations": tips
    }

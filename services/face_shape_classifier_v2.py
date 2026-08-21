from __future__ import annotations

import os
import math
from numbers import Real
from typing import Any, Dict, Mapping, Optional, Tuple
import joblib
import numpy as np

MODEL_VERSION = "v3.3-lgbm-aug19-verified"

FEATURES = ('face_length_width_ratio', 'forehead_to_cheekbone_ratio', 'jaw_to_cheekbone_ratio', 'forehead_to_jaw_ratio', 'forehead_to_lower_jaw_ratio', 'lower_jaw_width_to_cheekbone_ratio', 'chin_sharpness_ratio', 'jaw_taper_ratio', 'lower_face_height_to_face_height_ratio', 'gonial_angle_proxy', 'temple_to_cheekbone_ratio', 'jaw_slope_angle', 'jawline_curvature_index', 'midface_height_ratio')
CLASSES = ("heart", "long", "oval", "round", "square")

LABELS_KO = {
    "heart": "하트형",
    "long": "긴형",
    "oval": "계란형 또는 일반형",
    "round": "둥근형",
    "square": "사각형",
}

REASONS = {
    "heart": "상부 대비 하부 폭의 수축과 날렵한 턱 끝 특징이 하트형 데이터 분포에 가장 가깝습니다.",
    "long": "얼굴 세로/가로 비율과 긴 하관 특징이 긴형 데이터 분포에 가장 가깝습니다.",
    "oval": "전체 비율이 극단적이지 않고 균형 범위에 가까워 계란형 데이터 분포에 가장 가깝습니다.",
    "round": "얼굴 길이가 아담하고 완만한 턱선 조합이 둥근형 데이터 분포에 가장 가깝습니다.",
    "square": "턱 폭과 아래턱 각도가 탄탄하게 유지되어 사각형 데이터 분포에 가장 가깝습니다.",
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pure_lgbm_model.joblib")

_MODEL = None

def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL

def _resolve_feature(ratios: Mapping[str, Any], feature: str) -> Any:
    if feature in ratios:
        return ratios[feature]
    aliases = {
        "jaw_to_cheekbone_ratio": ("jaw_to_face_width_ratio",),
        "lower_jaw_width_to_cheekbone_ratio": ("lower_jaw_to_face_width_ratio",),
    }
    for alias in aliases.get(feature, ()):
        if alias in ratios:
            return ratios[alias]
    raise KeyError(feature)

def validate_ratios(ratios: Mapping[str, Any]) -> list[float]:
    values = []
    for feature in FEATURES:
        try:
            val = _resolve_feature(ratios, feature)
        except KeyError:
            val = 0.0
        if isinstance(val, bool) or not isinstance(val, Real):
            val = 0.0
        val = float(val)
        if not math.isfinite(val) or val <= 0:
            val = 0.001
        values.append(val)
    return values

def classify_face_shape(
    ratios: Mapping[str, Any],
    *,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    model = _get_model()
    raw_vals = np.array([validate_ratios(ratios)])

    probs = model.predict_proba(raw_vals)[0]
    full_scores = {cls_name: float(p) for cls_name, p in zip(CLASSES, probs)}
    sorted_scores = sorted(full_scores.items(), key=lambda x: x[1], reverse=True)

    best_type, best_prob = sorted_scores[0]
    second_type, second_prob = sorted_scores[1]
    margin = best_prob - second_prob

    return {
        "type": best_type,
        "label_ko": LABELS_KO[best_type],
        "confidence": round(best_prob, 4),
        "classification_margin": round(margin, 4),
        "reason": REASONS[best_type],
        "score_breakdown": {c: round(s, 4) for c, s in full_scores.items()},
        "second_candidate": {
            "type": second_type,
            "label_ko": LABELS_KO[second_type],
            "score": round(second_prob, 4),
        },
        "model_version": MODEL_VERSION,
    }

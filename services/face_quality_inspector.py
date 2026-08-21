import cv2
import numpy as np
from typing import Any, Dict, List, Tuple

def calculate_head_pose_and_symmetry(landmarks: np.ndarray, w: int, h: int) -> Dict[str, float]:
    """얼굴 3D 랜드마크 기반 정면 각도(Yaw, Pitch, Roll) 및 좌우 대칭성 산출"""
    # 랜드마크: 1: 코끝, 10: 이마정점, 152: 턱끝, 33: 좌안외각, 263: 우안외각, 234: 좌광대, 454: 우광대
    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    forehead = landmarks[10]
    chin = landmarks[152]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    # 1. Roll (2D 평면 기울기)
    d_y = right_eye[1] - left_eye[1]
    d_x = right_eye[0] - left_eye[0]
    roll_deg = np.degrees(np.arctan2(d_y, d_x))

    # 2. Yaw (좌우 회전각 프록시)
    dist_l = np.linalg.norm(nose[:2] - left_cheek[:2])
    dist_r = np.linalg.norm(nose[:2] - right_cheek[:2])
    yaw_ratio = dist_l / (dist_r + 1e-6)
    yaw_deg = abs(yaw_ratio - 1.0) * 45.0  # 1.0(정면) 기준 편차

    # 3. Pitch (상하 끄덕임 프록시)
    dist_top = np.linalg.norm(nose[:2] - forehead[:2])
    dist_bot = np.linalg.norm(nose[:2] - chin[:2])
    pitch_ratio = dist_top / (dist_bot + 1e-6)
    pitch_deg = abs(pitch_ratio - 0.95) * 40.0

    # 4. 좌우 기하 대칭 지수 (0~100점)
    symmetry_score = max(0.0, 100.0 - (yaw_deg * 2.5 + abs(roll_deg) * 1.5))

    return {
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(abs(roll_deg)),
        "symmetry_score": float(symmetry_score)
    }

def inspect_image_quality(image: np.ndarray, landmarks_px: np.ndarray) -> Dict[str, Any]:
    """
    이미지 글로벌 품질 및 4대 부위별(이마, 광대, 턱선, 조도) 측정 신뢰도 종합 검사
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 1. 글로벌 선명도 (블러링 검사)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var >= 120.0:
        sharpness_grade = "GOOD"
        sharpness_desc = "초점이 선명하고 엣지가 뚜렷합니다."
    elif laplacian_var >= 50.0:
        sharpness_grade = "NORMAL"
        sharpness_desc = "약간의 소프트함이 있으나 분석 가능한 수준입니다."
    else:
        sharpness_grade = "BLURRY"
        sharpness_desc = "초점이 흐리거나 흔들림이 있어 랜드마크 정확도가 떨어질 수 있습니다."

    # 2. 글로벌 조도 및 대비
    mean_brightness = float(np.mean(gray))
    std_brightness = float(np.std(gray))
    
    if 70 <= mean_brightness <= 190 and std_brightness >= 35:
        lighting_grade = "GOOD"
        lighting_desc = "얼굴 전체에 빛이 고르게 분산되어 있습니다."
    elif mean_brightness < 70:
        lighting_grade = "DARK"
        lighting_desc = "이미지가 다소 어두워 하관 경계선 식별에 주의가 필요합니다."
    else:
        lighting_grade = "HARSH"
        lighting_desc = "강한 조명 또는 역광으로 인해 일부 영역에 음영이 존재합니다."

    # 3. 정면 자세 및 대칭도 검사
    pose_info = calculate_head_pose_and_symmetry(landmarks_px, w, h)

    # -------------------------------------------------------------
    # 4. 부위별 측정 신뢰도 진단 (Regional Inspection)
    # -------------------------------------------------------------
    regional_status = []

    # ① 이마 / 상안부 (Forehead)
    # 앞머리 가림 또는 상단 조도 손실 체크
    forehead_p = landmarks_px[10]
    fx, fy = int(forehead_p[0]), int(forehead_p[1])
    fh_box_y1 = max(0, fy - int(h * 0.08))
    fh_box_y2 = min(h, fy + int(h * 0.04))
    fh_box_x1 = max(0, fx - int(w * 0.12))
    fh_box_x2 = min(w, fx + int(w * 0.12))
    
    forehead_roi = gray[fh_box_y1:fh_box_y2, fh_box_x1:fh_box_x2]
    fh_std = float(np.std(forehead_roi)) if forehead_roi.size > 0 else 0.0

    # 앞머리가 있거나 머리카락 결이 있으면 ROI 내 표준편차(에지)가 급증
    if fh_std > 52.0 or pose_info["pitch_deg"] > 14.0:
        forehead_status = "WARNING"
        forehead_msg = "앞머리(뱅) 가림 또는 상단 음영 감지 - 이마 가로폭 및 수축비 측정 신뢰도 낮음"
    elif fh_std > 42.0:
        forehead_status = "CAUTION"
        forehead_msg = "가벼운 시스루 뱅/잔머리 감지 - 이마 경계선 측정 오차 주의"
    else:
        forehead_status = "PASS"
        forehead_msg = "이마 라인과 헤어라인이 깨끗하게 노출되어 정밀 측정 완료"

    regional_status.append({
        "region": "이마 / 상안부 (Forehead)",
        "status": forehead_status,
        "detail": forehead_msg,
        "affected_metrics": ["forehead_to_cheekbone_ratio", "forehead_to_lower_jaw_ratio"]
    })

    # ② 광대 / 볼선 (Cheekbones)
    # 좌우 볼 영역 조도 및 사이드 헤어 가림 검사
    left_cheek_p = landmarks_px[234]
    right_cheek_p = landmarks_px[454]
    
    l_roi = gray[max(0, int(left_cheek_p[1]-20)):min(h, int(left_cheek_p[1]+20)), max(0, int(left_cheek_p[0]-20)):min(w, int(left_cheek_p[0]+20))]
    r_roi = gray[max(0, int(right_cheek_p[1]-20)):min(h, int(right_cheek_p[1]+20)), max(0, int(right_cheek_p[0]-20)):min(w, int(right_cheek_p[0]+20))]
    
    l_mean = float(np.mean(l_roi)) if l_roi.size > 0 else mean_brightness
    r_mean = float(np.mean(r_roi)) if r_roi.size > 0 else mean_brightness
    cheek_diff = abs(l_mean - r_mean)

    if pose_info["yaw_deg"] > 12.0 or cheek_diff > 45.0:
        cheek_status = "WARNING"
        cheek_msg = "고개 회전(Yaw) 또는 한쪽 볼 강한 그림자 감지 - 광대 최대폭 측정값 왜곡 가능성"
    elif pose_info["yaw_deg"] > 7.0 or cheek_diff > 25.0:
        cheek_status = "CAUTION"
        cheek_msg = "옆머리(사이드뱅)에 의한 일부 가림 또는 미세 측면광 감지"
    else:
        cheek_status = "PASS"
        cheek_msg = "좌우 광대뼈 외곽선이 대칭적이며 선명하게 노출됨"

    regional_status.append({
        "region": "광대 / 중안부 (Cheekbones)",
        "status": cheek_status,
        "detail": cheek_msg,
        "affected_metrics": ["face_length_width_ratio", "jaw_to_cheekbone_ratio"]
    })

    # ③ 턱선 / 하관 (Jawline & Chin)
    # 턱 끝 주변 영역 랜드마크 깊이 및 하악각 대칭성
    chin_p = landmarks_px[152]
    cx, cy = int(chin_p[0]), int(chin_p[1])
    chin_roi = gray[max(0, cy - int(h * 0.05)):min(h, cy + int(h * 0.05)), max(0, cx - int(w * 0.1)):min(w, cx + int(w * 0.1))]
    chin_mean = float(np.mean(chin_roi)) if chin_roi.size > 0 else mean_brightness

    if chin_mean < 45.0 or pose_info["pitch_deg"] > 15.0:
        jaw_status = "WARNING"
        jaw_msg = "턱 하단 강한 그림자 또는 고개 숙임으로 턱 끝 정점(V라인) 경계 모호"
    elif pose_info["roll_deg"] > 8.0:
        jaw_status = "CAUTION"
        jaw_msg = "고개 기울임(Roll)으로 인해 좌우 턱 각도 비대칭 보정 적용"
    else:
        jaw_status = "PASS"
        jaw_msg = "귀밑턱 하악각과 턱 끝 라인이 뚜렷하게 식별됨"

    regional_status.append({
        "region": "턱선 / 하관 (Jawline & Chin)",
        "status": jaw_status,
        "detail": jaw_msg,
        "affected_metrics": ["chin_sharpness_ratio", "gonial_angle_proxy", "jawline_curvature_index"]
    })

    # -------------------------------------------------------------
    # 5. 종합 신뢰도 점수 및 최종 판정 (Overall Health)
    # -------------------------------------------------------------
    warning_count = sum(1 for r in regional_status if r["status"] == "WARNING")
    caution_count = sum(1 for r in regional_status if r["status"] == "CAUTION")

    if warning_count >= 2 or pose_info["yaw_deg"] > 18.0 or laplacian_var < 35.0:
        overall_grade = "RETRY_RECOMMENDED"
        overall_score = max(40, 75 - warning_count * 15)
        overall_guide = "정면을 똑바로 응시하고 이마와 턱선이 잘 드러나도록 재촬영을 권장합니다."
    elif warning_count == 1 or caution_count >= 1:
        overall_grade = "ACCEPTABLE"
        overall_score = max(65, 90 - warning_count * 12 - caution_count * 5)
        overall_guide = "일부 부위에 경미한 가림/음영이 있으나 기하 알고리즘 보정으로 분석을 완료했습니다."
    else:
        overall_grade = "OPTIMAL"
        overall_score = 98.0
        overall_guide = "조명, 각도, 이목구비 노출도가 최상이며 측정 신뢰도가 매우 높습니다."

    return {
        "overall_grade": overall_grade,
        "overall_score": float(round(overall_score, 1)),
        "overall_guide": overall_guide,
        "global_quality": {
            "sharpness": {"grade": sharpness_grade, "score": float(round(laplacian_var, 1)), "desc": sharpness_desc},
            "lighting": {"grade": lighting_grade, "brightness": float(round(mean_brightness, 1)), "desc": lighting_desc},
            "pose": {
                "yaw_deg": float(round(pose_info["yaw_deg"], 1)),
                "pitch_deg": float(round(pose_info["pitch_deg"], 1)),
                "roll_deg": float(round(pose_info["roll_deg"], 1)),
                "symmetry_score": float(round(pose_info["symmetry_score"], 1))
            }
        },
        "regional_status": regional_status
    }

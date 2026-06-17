from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import mediapipe as mp
import math
from services.face_shape_classifier import classify_face_shape

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

    overlay_points = key_landmarks

    overlay_lines = [
        {
            "name": "face_height",
            "label_ko": "얼굴 세로 길이",
            "from": "top_face",
            "to": "chin"
        },
        {
            "name": "face_width",
            "label_ko": "얼굴 전체 너비",
            "from": "left_face_outer",
            "to": "right_face_outer"
        },
        {
            "name": "forehead_width",
            "label_ko": "이마 폭",
            "from": "left_forehead",
            "to": "right_forehead"
        },
        {
            "name": "cheekbone_width",
            "label_ko": "광대 폭",
            "from": "left_cheekbone",
            "to": "right_cheekbone"
        },
        {
            "name": "jaw_width",
            "label_ko": "턱 폭",
            "from": "left_jaw",
            "to": "right_jaw"
        },
        {
            "name": "lower_jaw_width",
            "label_ko": "아래턱 폭",
            "from": "left_lower_jaw",
            "to": "right_lower_jaw"
        },
        {
            "name": "lower_face_height",
            "label_ko": "하관 길이",
            "from": "nose_tip",
            "to": "chin"
        }
    ]

    analysis = {
        "face_shape": face_shape,
        "ratios": ratios,
        "centers": centers
    }

    overlay = {
        "points": overlay_points,
        "lines": overlay_lines
    }

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

        # 기존 응답 구조: 하위 호환용
        "key_landmarks": key_landmarks,
        "ratios": ratios,
        "centers": centers,
        "face_shape": face_shape,

        # 신규 응답 구조: 프론트엔드 연동용
        "analysis": analysis,
        "overlay": overlay,

        "message": "얼굴 랜드마크, 비율 계산, 가중치 기반 얼굴형 분류 성공"
    }
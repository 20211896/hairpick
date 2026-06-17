from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
import mediapipe as mp
from services.face_shape_classifier import classify_face_shape
from services.ratio_calculator import build_face_analysis_data

app = FastAPI(title="Face Analysis AI Backend")

mp_face_mesh = mp.solutions.face_mesh






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

    analysis_data = build_face_analysis_data(landmarks)

    key_landmarks = analysis_data["key_landmarks"]
    ratios = analysis_data["ratios"]
    centers = analysis_data["centers"]
    overlay = analysis_data["overlay"]

    face_shape = classify_face_shape(ratios)

    analysis = {
        "face_shape": face_shape,
        "ratios": ratios,
        "centers": centers
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
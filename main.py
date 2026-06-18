from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.face_shape_classifier import classify_face_shape
from services.ratio_calculator import build_face_analysis_data
from services.landmark_detector import detect_face_landmarks
from schemas import AnalyzeFaceResponse

app = FastAPI(title="Face Analysis AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 단계에서는 전체 허용. 배포 시 프론트 도메인으로 제한 필요.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)






@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "Face Analysis AI Backend is running"
    }


@app.post("/api/analyze-face", response_model=AnalyzeFaceResponse)
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

    detection_result = detect_face_landmarks(file_bytes)

    if not detection_result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=detection_result.get("message", "이미지를 읽을 수 없습니다.")
        )

    image_size = detection_result["image_size"]
    face_count = detection_result["face_count"]
    landmarks = detection_result["landmarks"]

    if not detection_result["face_detected"]:
        return {
            "success": False,
            "face_detected": False,
            "face_count": 0,
            "filename": file.filename,
            "error_code": "NO_FACE_DETECTED",
            "image_size": image_size,
            "message": "이미지에서 얼굴을 찾을 수 없습니다."
        }

    if face_count > 1:
        return {
            "success": False,
            "face_detected": True,
            "face_count": face_count,
            "filename": file.filename,
            "error_code": "MULTIPLE_FACES_DETECTED",
            "image_size": image_size,
            "message": "여러 명의 얼굴이 감지되었습니다. 한 명의 얼굴만 포함된 이미지를 사용해주세요."
        }

    analysis_data = build_face_analysis_data(landmarks)

    key_landmarks = analysis_data["key_landmarks"]
    ratios = analysis_data["ratios"]
    centers = analysis_data["centers"]
    quality = analysis_data["quality"]
    overlay = analysis_data["overlay"]

    face_shape = classify_face_shape(ratios)

    analysis = {
        "face_shape": face_shape,
        "ratios": ratios,
        "centers": centers,
        "quality": quality
    }

    return {
        "success": True,
        "face_detected": True,
        "face_count": face_count,
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size_bytes": len(file_bytes),
        "image_size": image_size,
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
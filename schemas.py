from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ImageSize(BaseModel):
    width: int
    height: int


class FaceShapeResult(BaseModel):
    type: str
    label_ko: str
    confidence: float
    reason: str
    score_breakdown: Dict[str, float]
    second_candidate: Optional[Dict[str, Any]] = None


class AnalysisResult(BaseModel):
    face_shape: FaceShapeResult
    ratios: Dict[str, float]
    centers: Dict[str, Any]


class AnalyzeFaceResponse(BaseModel):
    success: bool
    face_detected: bool
    face_count: int
    filename: Optional[str] = None
    content_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    image_size: Optional[ImageSize] = None
    landmark_count: Optional[int] = None

    # 기존 응답 구조: 하위 호환용
    key_landmarks: Optional[Dict[str, Any]] = None
    ratios: Optional[Dict[str, float]] = None
    centers: Optional[Dict[str, Any]] = None
    face_shape: Optional[FaceShapeResult] = None

    # 신규 응답 구조: 프론트엔드 연동용
    analysis: Optional[AnalysisResult] = None
    overlay: Optional[Dict[str, Any]] = None

    message: str

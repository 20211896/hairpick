import cv2
import numpy as np
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh


def decode_image(file_bytes: bytes):
    """
    업로드된 이미지 bytes를 OpenCV BGR 이미지로 디코딩한다.
    """
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return image_bgr


def detect_face_landmarks(file_bytes: bytes) -> dict:
    """
    이미지 bytes에서 얼굴을 감지하고 MediaPipe Face Mesh 랜드마크를 추출한다.
    """
    image_bgr = decode_image(file_bytes)

    if image_bgr is None:
        return {
            "success": False,
            "error_code": "INVALID_IMAGE",
            "message": "이미지를 읽을 수 없습니다."
        }

    height, width, channels = image_bgr.shape

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    image_size = {
        "width": width,
        "height": height,
        "channels": channels
    }

    if not results.multi_face_landmarks:
        return {
            "success": True,
            "face_detected": False,
            "face_count": 0,
            "image_size": image_size,
            "landmarks": []
        }

    face_count = len(results.multi_face_landmarks)

    if face_count > 1:
        return {
            "success": True,
            "face_detected": True,
            "face_count": face_count,
            "image_size": image_size,
            "landmarks": []
        }

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

    return {
        "success": True,
        "face_detected": True,
        "face_count": face_count,
        "image_size": image_size,
        "landmarks": landmarks
    }

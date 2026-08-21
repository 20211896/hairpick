import sys
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from services.face_analyzer import extract_ratios_from_landmarks, generate_face_ti_code
from services.face_shape_classifier_v2 import classify_face_shape

def main():
    sample_dir = Path("sampleimg")
    if not sample_dir.exists():
        print(f"[!] {sample_dir} 디렉토리를 찾을 수 없습니다.")
        sys.exit(1)

    folder_map = {
        "Heart": "heart",
        "Oblong": "long",
        "Oval": "oval",
        "Round": "round",
        "Square": "square"
    }

    mp_face_mesh = mp.solutions.face_mesh
    detector = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )

    results_list = []
    total_evaluated = 0
    total_correct = 0

    print("=" * 100)
    print(" [HairPick AI] 5대 얼굴형 500장 전수 벤치마크 평가 시작 (폴더당 100장)")
    print("=" * 100)

    for folder_name, expected_label in folder_map.items():
        curr_folder = sample_dir / folder_name
        if not curr_folder.exists():
            continue

        img_files = sorted(list(curr_folder.glob("*.jpg")) + list(curr_folder.glob("*.png")))[:100]
        folder_correct = 0
        folder_detected = 0

        print(f"\n[*] [{folder_name}] 분석 중... (대상: {len(img_files)}장)")
        print("-" * 100)

        for idx, img_path in enumerate(img_files, 1):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mesh_res = detector.process(img_rgb)

            if not mesh_res.multi_face_landmarks:
                continue

            folder_detected += 1
            landmarks_px = np.array([[lm.x * w, lm.y * h] for lm in mesh_res.multi_face_landmarks[0].landmark])
            ratios = extract_ratios_from_landmarks(landmarks_px, w, h)
            
            clf_res = classify_face_shape(ratios)
            pred = clf_res["type"]
            conf = clf_res["confidence"]
            scores = clf_res["score_breakdown"]

            face_ti = generate_face_ti_code(ratios, scores, pred)
            is_correct = (pred == expected_label)
            if is_correct:
                folder_correct += 1

            if idx <= 5 or idx % 25 == 0 or idx == len(img_files):
                mark = "O" if is_correct else "X"
                score_str = f"H:{scores['heart']:.2f} L:{scores['long']:.2f} O:{scores['oval']:.2f} R:{scores['round']:.2f} S:{scores['square']:.2f}"
                print(f"  [{folder_name} {idx:03d}/{len(img_files):03d}] 정답: {expected_label:<6} | 예측: {pred:<6} [{mark}] ({conf*100:5.1f}%) | TI: {face_ti['code']} | {score_str}")

            row = {
                "folder": folder_name,
                "filename": img_path.name,
                "expected": expected_label,
                "predicted": pred,
                "correct": is_correct,
                "confidence": conf,
                "face_ti": face_ti["code"],
                "score_heart": scores["heart"],
                "score_long": scores["long"],
                "score_oval": scores["oval"],
                "score_round": scores["round"],
                "score_square": scores["square"]
            }
            row.update(ratios)
            results_list.append(row)

        acc = (folder_correct / folder_detected * 100) if folder_detected > 0 else 0.0
        print(f"  => [{folder_name}] 검출: {folder_detected}/{len(img_files)} | 일치: {folder_correct} | 정확도: {acc:.1f}%")
        total_evaluated += folder_detected
        total_correct += folder_correct

    detector.close()

    if not results_list:
        print("\n[!] 평가된 이미지가 없습니다.")
        return

    df_res = pd.DataFrame(results_list)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "benchmark_500_results.csv"
    df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 100)
    print(" [최종 500장 평가 결과 요약]")
    print("=" * 100)

    matrix = pd.crosstab(df_res["expected"], df_res["predicted"], margins=True)
    print("\n[Confusion Matrix (정답행 x 예측열)]")
    print(matrix.to_string())

    print("\n[클래스별 정답률 (Recall)]")
    for shape in ["heart", "long", "oval", "round", "square"]:
        sub = df_res[df_res["expected"] == shape]
        if len(sub) > 0:
            c_acc = (sub["correct"].sum() / len(sub)) * 100
            print(f"  - {shape.upper():<8}: {sub['correct'].sum():3d} / {len(sub):3d} ({c_acc:5.1f}%)")

    overall_acc = (total_correct / total_evaluated * 100) if total_evaluated > 0 else 0.0
    print("-" * 100)
    print(f"[*] 전체 평가 샘플 : {total_evaluated}장")
    print(f"[*] 전체 일치 건수 : {total_correct}장")
    print(f"[*] 최종 전체 정확도 : {overall_acc:.2f}%")
    print(f"[*] 상세 CSV 저장   : {csv_path}")
    print("=" * 100 + "\n")

if __name__ == "__main__":
    main()

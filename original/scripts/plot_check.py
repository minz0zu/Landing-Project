import cv2
import os
import random
import glob
from pathlib import Path

def plot_random_samples(image_dir, label_dir, output_dir, num_samples=10):
    # 결과 저장 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # 이미지 파일 목록 가져오기
    image_files = glob.glob(os.path.join(image_dir, "*.*"))

    if len(image_files) == 0:
        print(f"이미지가 없습니다: {image_dir}")
        return

    # 랜덤 샘플
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    for img_path in samples:
        # 이미지 읽기
        img = cv2.imread(img_path)
        if img is None:
            print(f"이미지 읽기 실패: {img_path}")
            continue

        h, w = img.shape[:2]

        # 대응 라벨 경로
        file_stem = Path(img_path).stem
        label_path = os.path.join(label_dir, f"{file_stem}.txt")

        if not os.path.exists(label_path):
            print(f"라벨 없음: {label_path}")
            continue

        # 라벨 읽기 및 그리기
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(f"라벨 형식 이상 ({file_stem}): {line}")
                continue

            # YOLO 포맷
            cls, cx, cy, bw, bh = map(float, parts)

            # 픽셀 좌표 복원
            x_c_px = cx * w
            y_c_px = cy * h
            w_px = bw * w
            h_px = bh * h

            x1 = int(x_c_px - w_px / 2)
            y1 = int(y_c_px - h_px / 2)
            x2 = int(x_c_px + w_px / 2)
            y2 = int(y_c_px + h_px / 2)

            # 이미지 범위로 클램핑(자르기)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            # 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 텍스트가 이미지 밖으로 나가지 않게
            text_y = max(0, y1 - 10)
            cv2.putText(img, f"Class: {int(cls)}", (x1, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 결과 저장
        save_path = os.path.join(output_dir, f"{file_stem}.png")
        cv2.imwrite(save_path, img)
        print(f"Saved plot: {save_path}")

# 실행
processed_img_dir = "/home/minji/landing_project/processed_original/Train/images"
processed_label_dir = "/home/minji/landing_project/processed_original/Train/labels"
output_check_dir = f"/home/minji/landing_project/processed_original/Train/plotted_image"
plot_random_samples(processed_img_dir, processed_label_dir, output_check_dir)

processed_img_dir = "/home/minji/landing_project/processed_original/Validation/images"
processed_label_dir = "/home/minji/landing_project/processed_original/Validation/labels"
output_check_dir = f"/home/minji/landing_project/processed_original/Validation/plotted_image"
plot_random_samples(processed_img_dir, processed_label_dir, output_check_dir)
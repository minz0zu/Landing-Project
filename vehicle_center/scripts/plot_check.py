import cv2
import os
import random
import glob
from pathlib import Path

def plot_validation_samples(image_dir, label_dir, output_dir, num_samples=10):
    os.makedirs(output_dir, exist_ok=True)

    image_files = glob.glob(os.path.join(image_dir, "*.*"))
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg'))]

    if len(image_files) == 0:
        print(f"이미지를 찾을 수 없습니다: {image_dir}")
        return

    # 랜덤하게 10개 선택
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    print(f"{len(samples)}개의 샘플 시각화를 시작합니다.")

    for img_path in samples:
        img = cv2.imread(img_path)
        if img is None: continue

        h, w = img.shape[:2]
        file_stem = Path(img_path).stem
        label_path = os.path.join(label_dir, f"{file_stem}.txt")

        if not os.path.exists(label_path):
            print(f"라벨 없음: {file_stem}")
            continue

        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5: continue

                cls, cx, cy, bw, bh = map(float, parts)

                # YOLO (Normalize) -> Pixel 좌표 변환
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                # Class 0 (Plate): 빨간색 
                # Class 1 (Car): 파란색
                if int(cls) == 0:
                    color = (0, 0, 255) 
                    label_text = "Plate"
                else:
                    color = (255, 0, 0)
                    label_text = "Car"

                # 박스 및 라벨 그리기
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        save_path = os.path.join(output_dir, f"plotted_{file_stem}.png")
        cv2.imwrite(save_path, img)

    print(f"시각화 완료: {output_dir}")

if __name__ == "__main__":
    BASE_PATH = "/mnt/hdd_6tb/minji/processed_vehicle_center"
    
    for sub_set in ["Train", "Validation"]:
        IMG_DIR = os.path.join(BASE_PATH, sub_set, "images")
        LBL_DIR = os.path.join(BASE_PATH, sub_set, "labels")
        OUT_DIR = os.path.join(BASE_PATH, sub_set, "plotted_image")
        
        plot_validation_samples(IMG_DIR, LBL_DIR, OUT_DIR)
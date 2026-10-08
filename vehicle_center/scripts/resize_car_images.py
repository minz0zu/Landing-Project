import cv2
import os
import glob

# 경로 설정
IMG_PATH = "/mnt/hdd_6tb/minji/P2_Dhaka_Dataset-29/train/images"
LBL_PATH = "/mnt/hdd_6tb/minji/P2_Dhaka_Dataset-29/train/labels"
OUT_PATH = "/mnt/hdd_6tb/minji/processed_car/train"

os.makedirs(OUT_PATH, exist_ok=True)

# 차량으로 간주할 클래스 리스트 수정 (1, 2, 3, 4, 5, 7 만 포함)
target_car_classes = ["1", "2", "3", "4", "5", "7"] 

img_files = glob.glob(os.path.join(IMG_PATH, "*"))

for img_path in img_files:
    img_name = os.path.basename(img_path)
    base_name = os.path.splitext(img_name)[0]
    lbl_path = os.path.join(LBL_PATH, base_name + ".txt")
    
    if not os.path.exists(lbl_path):
        continue
    
    img = cv2.imread(img_path)
    if img is None: continue
    h, w, _ = img.shape
    
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
        
    car_count = 0
    for line in lines:
        parts = line.split()
        if not parts: continue
        
        # 수정 포인트: 클래스 번호가 [1, 2, 3, 4, 5, 7] 중 하나인지 확인
        if parts[0] in target_car_classes:
            
            # YOLO 정규화 좌표 -> 픽셀 좌표 복원
            cx, cy, cw, ch = map(float, parts[1:])
            x1 = int((cx - cw/2) * w)
            y1 = int((cy - ch/2) * h)
            x2 = int((cx + cw/2) * w)
            y2 = int((cy + ch/2) * h)
            
            # 이미지 경계 처리 (이미지 범위를 벗어나지 않도록 제어)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop 수행
            crop = img[y1:y2, x1:x2]
            
            if crop.size > 0:
                # 640x640 크기로 Resize
                resized_crop = cv2.resize(crop, (640, 640))
                
                # 파일명 형식 준수: {image_name}_{n}.png
                save_filename = f"{base_name}_{car_count}.png"
                cv2.imwrite(os.path.join(OUT_PATH, save_filename), resized_crop)
                car_count += 1

print(f"수정된 클래스 필터로 전처리 완료!")
print(f"생성된 조각 이미지 수: {len(glob.glob(OUT_PATH+'/*.png'))}개")
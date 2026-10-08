import cv2
import os
import glob

def resize_images(input_path, output_path):

    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    
    # 이미지 파일 목록
    image_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff"):
        image_files += glob.glob(os.path.join(input_path, "**", ext), recursive=True)
    print(f"Total {len(image_files)} images")

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]

        # 긴 변을 640으로 맞추는 비율
        scale = 640 / max(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 저장
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_path, f"{stem}.png")

        cv2.imwrite(out_path, resized_img)

# 실행
original_dir = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/Validation/01.원천데이터"
image_dir = "/home/minji/landing_project/processed_original/Validation/image"
resize_images(original_dir, image_dir)
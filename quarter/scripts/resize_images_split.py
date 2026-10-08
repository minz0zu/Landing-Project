import cv2
import os
import glob

def resize_images_split(input_path, output_path):

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
        half_h, half_w = h // 2, w // 2
        
        # 4분할 영역 좌표
        tiles = [
            img[0:half_h, 0:half_w],       # 0: 좌측 상단
            img[0:half_h, half_w:w],       # 1: 우측 상단
            img[half_h:h, 0:half_w],       # 2: 좌측 하단
            img[half_h:h, half_w:w]        # 3: 우측 하단
        ]

        stem = os.path.splitext(os.path.basename(img_path))[0]

        for i, tile in enumerate(tiles):
            th, tw = tile.shape[:2]

            scale = 640 / max(th, tw)
            new_w = int(round(tw * scale))
            new_h = int(round(th * scale))
            
            # 리사이즈 수행
            resized_tile = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 저장명
            out_path = os.path.join(output_path, f"{stem}_{i}.png")
            cv2.imwrite(out_path, resized_tile)

# 실행
original_dir = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/01.원천데이터"
image_dir = "/mnt/hdd_6tb/minji/processed_quarter/Test/images"
resize_images_split(original_dir, image_dir)
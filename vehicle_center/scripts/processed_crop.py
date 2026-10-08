import cv2
import os
import numpy as np
from ultralytics import YOLO
import multiprocessing as mp

def process_chunk(gpu_id, file_list, sub_set, BASE_PATH, OUT_PATH, MODEL_PATH):
    # 각 프로세스별로 지정된 GPU에 모델 할당
    device = f'cuda:{gpu_id}'
    vehicle_model = YOLO(MODEL_PATH).to(device)

    img_dir = os.path.join(BASE_PATH, sub_set, "images")
    label_dir = os.path.join(BASE_PATH, sub_set, "labels")
    save_img_dir = os.path.join(OUT_PATH, sub_set, "images")
    save_label_dir = os.path.join(OUT_PATH, sub_set, "labels")

    for img_file in file_list:
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        txt_path = os.path.join(label_dir, f"{img_name}.txt")

        if not os.path.exists(txt_path):
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]

        # 정답 번호판 정보 읽기
        gt_plates = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, n_cx, n_cy, n_w, n_h = map(float, parts)
                    pw, ph = n_w * img_w, n_h * img_h
                    px, py = (n_cx * img_w) - (pw / 2), (n_cy * img_h) - (ph / 2)
                    gt_plates.append({'cls': cls, 'bbox': [px, py, pw, ph]})

        # 차량 검출 (verbose=False로 설정하여 터미널 혼선 방지)
        results = vehicle_model(img, conf=0.4, verbose=False)
        detected_vehicles = results[0].boxes.xyxy.cpu().numpy()

        for v_idx, v_box in enumerate(detected_vehicles):
            vx1, vy1, vx2, vy2 = map(int, v_box)
            for p_idx, plate in enumerate(gt_plates):
                px, py, pw, ph = plate['bbox']
                pcx, pcy = px + pw/2, py + ph/2
                
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    vx1, vy1 = max(0, vx1), max(0, vy1)
                    vx2, vy2 = min(img_w, vx2), min(img_h, vy2)
                    crop_img = img[vy1:vy2, vx1:vx2]
                    if crop_img.size == 0: continue
                    
                    crop_h, crop_w = crop_img.shape[:2]
                    resized_img = cv2.resize(crop_img, (640, 640))
                    
                    save_name = f"{img_name}_{v_idx}"
                    cv2.imwrite(os.path.join(save_img_dir, f"{save_name}.png"), resized_img)

                    new_lx, new_ly = px - vx1, py - vy1
                    new_cx = (new_lx + pw/2) / crop_w
                    new_cy = (new_ly + ph/2) / crop_h
                    new_w, new_h = pw / crop_w, ph / crop_h

                    with open(os.path.join(save_label_dir, f"{save_name}.txt"), 'w') as f_label:
                        f_label.write(f"{int(plate['cls'])} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}\n")
                    break 

if __name__ == '__main__':
    # 설정값
    BASE_PATH = "/mnt/hdd_6tb/minji/processed_original"
    OUT_PATH = "/mnt/hdd_6tb/minji/processed_vehicle_center"
    MODEL_PATH = "/home/minji/landing_project/vehicle_center/train_results/Car/weights/best.pt"
    sub_sets = ["Train", "Validation"]

    for sub_set in sub_sets:
        img_dir = os.path.join(BASE_PATH, sub_set, "images")
        
        # 미리 저장 폴더 생성
        os.makedirs(os.path.join(OUT_PATH, sub_set, "images"), exist_ok=True)
        os.makedirs(os.path.join(OUT_PATH, sub_set, "labels"), exist_ok=True)

        # 전체 이미지 리스트 확보 후 2등분
        all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]
        mid = len(all_files) // 2
        
        chunk1 = all_files[:mid]
        chunk2 = all_files[mid:]

        print(f"{sub_set} 처리 시작: GPU 1({len(chunk1)}장), GPU 2({len(chunk2)}장)")

        # 멀티프로세싱 시작
        p1 = mp.Process(target=process_chunk, args=(1, chunk1, sub_set, BASE_PATH, OUT_PATH, MODEL_PATH))
        p2 = mp.Process(target=process_chunk, args=(2, chunk2, sub_set, BASE_PATH, OUT_PATH, MODEL_PATH))

        p1.start()
        p2.start()

        # 작업 완료 대기
        p1.join()
        p2.join()
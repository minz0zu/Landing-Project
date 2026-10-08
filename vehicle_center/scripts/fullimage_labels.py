import cv2
import os
import numpy as np
from ultralytics import YOLO
import multiprocessing as mp

def process_integration(gpu_id, file_list, sub_set, BASE_PATH, OUT_PATH, MODEL_PATH):
    # 각 프로세스별 GPU 할당 및 모델 로드
    device = f'cuda:{gpu_id}'
    vehicle_model = YOLO(MODEL_PATH).to(device)

    img_dir = os.path.join(BASE_PATH, sub_set, "images")
    origin_label_dir = os.path.join(BASE_PATH, sub_set, "labels")
    save_label_dir = os.path.join(OUT_PATH, sub_set, "labels")

    for img_file in file_list:
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        txt_path = os.path.join(origin_label_dir, f"{img_name}.txt")

        if not os.path.exists(txt_path):
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]

        # 기존 번호판 라벨 읽기
        plate_labels = []
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = map(float, parts)
                    plate_labels.append([cls, cx, cy, w, h])

        # 차량 모델로 자동차 검출
        results = vehicle_model(img, conf=0.4, verbose=False)
        detected_vehicles = results[0].boxes.xywhn.cpu().numpy()

        final_labels = []
        # 번호판 라벨들을 먼저 추가
        for pl in plate_labels:
            final_labels.append(f"{int(pl[0])} {pl[1]:.6f} {pl[2]:.6f} {pl[3]:.6f} {pl[4]:.6f}")

        # 번호판을 포함한 차량만 Class 1로 추가
        for v_box in detected_vehicles:
            vcx, vcy, vw, vh = v_box
            
            is_target_car = False
            for pl in plate_labels:
                pcx, pcy = pl[1], pl[2]
                
                # 번호판 중심점이 차량 박스 안에 있는지 확인
                if (vcx - vw/2 <= pcx <= vcx + vw/2) and (vcy - vh/2 <= pcy <= vcy + vh/2):
                    is_target_car = True
                    break
            
            if is_target_car:
                # Class 1로 추가
                final_labels.append(f"1 {vcx:.6f} {vcy:.6f} {vw:.6f} {vh:.6f}")

        # 최종 통합 라벨 저장
        save_txt_path = os.path.join(save_label_dir, f"{img_name}.txt")
        with open(save_txt_path, 'w') as f_out:
            f_out.write("\n".join(final_labels) + "\n")

if __name__ == '__main__':
    # 경로 설정
    BASE_PATH = "/mnt/hdd_6tb/minji/processed_original"
    OUT_PATH = "/mnt/hdd_6tb/minji/processed_vehicle_center"
    MODEL_PATH = "/home/minji/landing_project/vehicle_center/train_results/Car/weights/best.pt"
    sub_sets = ["Train", "Validation"]

    for sub_set in sub_sets:
        img_dir = os.path.join(BASE_PATH, sub_set, "images")
        
        # 저장 폴더 생성 확인
        os.makedirs(os.path.join(OUT_PATH, sub_set, "labels"), exist_ok=True)

        all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg'))]
        mid = len(all_files) // 2
        
        chunk1 = all_files[:mid]
        chunk2 = all_files[mid:]

        print(f"{sub_set} 통합 라벨링 시작: GPU 1({len(chunk1)}장), GPU 2({len(chunk2)}장)")

        p1 = mp.Process(target=process_integration, args=(1, chunk1, sub_set, BASE_PATH, OUT_PATH, MODEL_PATH))
        p2 = mp.Process(target=process_integration, args=(2, chunk2, sub_set, BASE_PATH, OUT_PATH, MODEL_PATH))

        p1.start()
        p2.start()

        p1.join()
        p2.join()
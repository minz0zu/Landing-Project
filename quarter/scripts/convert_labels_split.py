import json
import os
import glob

def convert_labels_split(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    
    json_files = glob.glob(os.path.join(input_path, "**", "*.json"), recursive=True)
    print(f"Total {len(json_files)} label files")
    
    for j_path in json_files:
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 원본 해상도 및 분할 기준 (half_w, half_h)
        res = data['Raw_Data_Info']['resolution']
        w_orig, h_orig = (map(int, res) if isinstance(res, list) else map(int, res.replace("x", ",").split(",")))

        hw, hh = w_orig // 2, h_orig // 2

        file_basename = os.path.splitext(os.path.basename(j_path))[0]

        # 4분할 영역별 라벨을 담을 리스트 (0:좌상, 1:우상, 2:좌하, 3:우하)
        split_labels = [[] for _ in range(4)]

        annotations = data.get('Learning_Data_Info', {}).get('annotations', [])
        for ann in annotations:
            for plate in ann.get('license_plate', []):
                x, y, w, h = plate['bbox']
                cx, cy = x + w/2.0, y + h/2.0 # 원본 중심점

                # 중심점 위치에 따라 어느 타일(0~3)에 속할지 결정
                # 0:좌상(x<hw, y<hh), 1:우상(x>=hw, y<hh), 2:좌하(x<hw, y>=hh), 3:우하(x>=hw, y>=hh)
                tile_idx = (1 if cx >= hw else 0) + (2 if cy >= hh else 0)
                
                # 해당 타일 좌표계로 변환
                new_cx = cx - (hw if cx >= hw else 0)
                new_cy = cy - (hh if cy >= hh else 0)

                # 타일 크기 기준으로 정규화
                norm_cx = new_cx / hw
                norm_cy = new_cy / hh
                norm_w = w / hw
                norm_h = h / hh

                split_labels[tile_idx].append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

        # 결과 저장
        for i in range(4):
            output_txt_path = os.path.join(output_path, f"{file_basename}_{i}.txt")
            with open(output_txt_path, 'w', encoding='utf-8') as f_txt:
                f_txt.write("\n".join(split_labels[i]))

# 실행 경로
#json_dir = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/data/Validation/02.라벨링데이터"
#label_dir = "/mnt/hdd_6tb/minji/processed_quarter/Validation/labels"
json_dir = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/02.라벨링데이터"
label_dir = "/mnt/hdd_6tb/minji/processed_quarter/Test/labels"
convert_labels_split(json_dir, label_dir)
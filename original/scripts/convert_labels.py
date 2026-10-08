import json
import os
import glob

def convert_labels(input_path, output_path):

    # 출력 디렉토리 생성
    os.makedirs(output_path, exist_ok=True)
    
    # 모든 JSON 파일 목록 가져오기
    json_files = glob.glob(os.path.join(input_path, "**", "*.json"), recursive=True)
    print(f"Total {len(json_files)} label files")
    
    for j_path in json_files:
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 이미지 원본 해상도 정보 추출
        # JSON의 "resolution": "3840, 2160" 부분을 잘라서 숫자로 변환
        res_val = data['Raw_Data_Info']['resolution']

        if isinstance(res_val, list):
            w_orig, h_orig = map(int, res_val)
        else:
            w_orig, h_orig = map(int, [s.strip() for s in res_val.split(",")])

        # 파일명 추출 (확장자 제외)
        file_basename = os.path.splitext(os.path.basename(j_path))[0]
        output_txt_path = os.path.join(output_path, file_basename + ".txt")

        with open(output_txt_path, 'w', encoding='utf-8') as f_txt:
            # Learning_Data_Info -> annotations -> license_plate 리스트 확인
            annotations = data.get('Learning_Data_Info', {}).get('annotations', [])
            
            for ann in annotations:
                # license_plate 내부에 여러 번호판이 있을 수 있음
                plates = ann.get('license_plate', [])
                for plate in plates:
                    # class = 0
                    cls_id = 0
                    
                    # bbox 좌표 추출
                    x, y, w, h = plate['bbox']
                    
                    # YOLO 포맷 정규화 (0~1 사이)
                    x_center = (x + w / 2.0) / w_orig
                    y_center = (y + h / 2.0) / h_orig
                    norm_w = w / w_orig
                    norm_h = h / h_orig

                    f_txt.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

json_dir = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/labels"
label_dir = "/mnt/hdd_6tb/minji/processed/Test/labels"
convert_labels(json_dir, label_dir)
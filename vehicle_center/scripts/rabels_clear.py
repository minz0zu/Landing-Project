import os
import glob

# 경로 설정
# 리사이즈 완료된 이미지가 있는 경로 (파일명 매칭용)
IMG_PATH = "/mnt/hdd_6tb/minji/processed_car/Validation/images" 
# 원본 뒤죽박죽 라벨이 있는 경로
LBL_PATH = "/mnt/hdd_6tb/minji/P2_Dhaka_Dataset-29/valid/labels"
# 수정된 라벨이 저장될 경로
OUT_PATH = "/mnt/hdd_6tb/minji/processed_car/Validation/labels"

os.makedirs(OUT_PATH, exist_ok=True)

# 차량으로 통합할 클래스 리스트 (1, 2, 3, 4, 5, 7)
target_car_classes = ["1", "2", "3", "4", "5", "7"] 

# 리사이즈된 이미지 목록을 가져옵니다 (확장자 상관없이 이름만 추출)
img_files = glob.glob(os.path.join(IMG_PATH, "*.png")) # .png로 저장하셨다고 해서 설정
print(f"이미지 개수: {len(img_files)}개에 대한 라벨 처리를 시작합니다.")

processed_count = 0

for img_path in img_files:
    # 파일 이름만 추출 (예: image123)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    lbl_path = os.path.join(LBL_PATH, base_name + ".txt")
    
    # 해당 이미지의 원본 라벨 파일이 존재하는지 확인
    if not os.path.exists(lbl_path):
        continue
    
    with open(lbl_path, 'r') as f:
        lines = f.readlines()
    
    new_labels = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        
        # 클래스 번호(parts[0])가 타겟 리스트에 있는지 확인
        if parts[0] in target_car_classes:
            # 클래스 번호를 '1'로 변경
            parts[0] = "1"
            new_labels.append(" ".join(parts))
    
    # 필터링된 라벨이 있을 경우에만 새 txt 파일 생성
    if new_labels:
        out_path = os.path.join(OUT_PATH, base_name + ".txt")
        with open(out_path, 'w') as f:
            f.write("\n".join(new_labels))
        processed_count += 1

print(f"생성된 라벨 파일 수: {processed_count}개")
print(f"저장 위치: {OUT_PATH}")
from ultralytics import YOLO

def train_yolo():
    model = YOLO("yolo11n.pt") 

    model.train(
        data="/home/minji/landing_project/vehicle_center/configs/data_plate.yaml",
        epochs=50,
        imgsz=640,
        batch=72,
        device=[3],
        optimizer='SGD',
        lr0=0.01,
        project="/home/minji/landing_project/vehicle_center/train_results",
        name='SGD_aug',
        exist_ok=True, # 폴더가 이미 있어도 덮어쓰기

        # --- [날씨 및 조명 대응 증강 설정] ---
        hsv_v=0.4,           # Brightness (0.0 ~ 1.0): 어두운 날/밤 대응 (명도 변화)
        hsv_s=0.7,           # Saturation (0.0 ~ 1.0): 비 오는 날의 채도 저하 대응
        hsv_h=0.015,         # Hue (0.0 ~ 1.0): 조명 색온도 변화 대응
        
        # --- [노이즈 및 가림 대응] ---
        mixup=0.1,           # 이미지 합성으로 복잡한 배경 학습
        mosaic=1.0,          # 4장의 이미지를 합쳐 객체 크기 변화 대응
        
        # --- [기하학적 변형 - 번호판 각도 대응] ---
        degrees=10.0,        # 회전 (주차된 차량이나 커브길 대응)
        translate=0.1,       # 이동
        scale=0.5,           # 크기 변화
        shear=2.0,           # 비틀기 (번호판이 사선으로 보일 때)
        perspective=0.0001,  # 원근감 (카메라 각도 대응)
        
        # --- 마지막 10 에포크는 증강을 끄고 정교하게 학습 ---
        close_mosaic=10,
        
        workers=8,      # CPU가 데이터를 미리 읽어오게 해서 GPU 대기 시간을 줄임
        cache=True
    )

if __name__ == "__main__":
    train_yolo()
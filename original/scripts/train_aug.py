# from ultralytics import YOLO

# def train_yolo():
#     model = YOLO("yolo11n.pt") 

#     # 모델 학습
#     model.train(
#         data="/home/minji/landing_project/configs/data_plate.yaml",
#         epochs=50,
#         imgsz=640,
#         batch=36,
#         device=3,
#         optimizer='SGD',
#         lr0       = 0.01,
#         project= "/home/minji/landing_project/train_results_split",
#         name='SGD_aug',폴더가 이미 있어도 덮어쓰기

#         exist_ok=True,
#         # --- [날씨 및 조명 대응 증강 설정] ---
#         hsv_v=0.4,           # Brightness (0.0 ~ 1.0): 어두운 날/밤 대응 (명도 변화)
#         hsv_s=0.7,           # Saturation (0.0 ~ 1.0): 비 오는 날의 채도 저하 대응
#         hsv_h=0.015,         # Hue (0.0 ~ 1.0): 조명 색온도 변화 대응
        
#         # --- [노이즈 및 가림 대응] ---
#         mixup=0.1,           # 이미지 합성으로 복잡한 배경 학습
#         mosaic=1.0,          # 4장의 이미지를 합쳐 객체 크기 변화 대응
        
#         # --- [기하학적 변형 - 번호판 각도 대응] ---
#         degrees=10.0,        # 회전 (주차된 차량이나 커브길 대응)
#         translate=0.1,       # 이동
#         scale=0.5,           # 크기 변화
#         shear=2.0,           # 비틀기 (번호판이 사선으로 보일 때)
#         perspective=0.0001,  # 원근감 (카메라 각도 대응)
        
#         # --- 마지막 10 에포크는 증강을 끄고 정교하게 학습 ---
#         close_mosaic=10
#     )

# if __name__ == "__main__":
#     train_yolo()
# 위 코드를 멀티로 재개함.
import os
from ultralytics import YOLO

def train_yolo():
    model = YOLO("/home/minji/landing_project/train_results_split/SGD_aug/weights/last.pt") 

    # CUDA_VISIBLE_DEVICES로 GPU를 지정하면, device는 자동으로 [0, 1, ...]로 매핑됨
    # 예: CUDA_VISIBLE_DEVICES=1,3 이면, device=[0,1] 또는 "0,1"로 사용
    # 이렇게 하면 resume할 때도 device 설정이 일관되게 유지됨
    
    # CUDA_VISIBLE_DEVICES가 설정되어 있으면 그걸 사용, 없으면 "1,3" 사용
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        # CUDA_VISIBLE_DEVICES=1,3 이면, 보이는 GPU는 0,1이 됨
        num_gpus = len(cuda_visible.split(","))
        if num_gpus > 1:
            device = list(range(num_gpus))  # [0, 1] 또는 [0, 1, 2, ...]
        else:
            device = 0
    else:
        # CUDA_VISIBLE_DEVICES가 없으면 직접 지정
        device = "1,3"
    
    model.train(
        resume=True,
        device=device,
        batch=36,      # 총 배치 크기 (GPU당 자동 분배)
        imgsz=640,
    )

if __name__ == "__main__":
    train_yolo()
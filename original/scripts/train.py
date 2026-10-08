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
#         name='SGD',
#         exist_ok=True, # 폴더가 이미 있어도 덮어쓰기
#     )

# if __name__ == "__main__":
#     train_yolo()

from ultralytics import YOLO

def train_yolo():
    model = YOLO("/home/minji/landing_project/train_results_split/SGD/weights/last.pt")

    # 모델 학습
    model.train(
        resume=True
    )

if __name__ == "__main__":
    train_yolo()
from ultralytics import YOLO

def train_yolo():
    model = YOLO("yolo11n.pt") 

    model.train(
        data="/home/minji/landing_project/vehicle_center/configs/data_plate.yaml",
        epochs=50,
        imgsz=640,
        batch=36,
        device='1,2',
        optimizer='SGD',
        lr0=0.01,
        project="/home/minji/landing_project/vehicle_center/train_results",
        name='SGD',
        exist_ok=True,
        workers=8,
        save=True
    )

if __name__ == "__main__":
    train_yolo()
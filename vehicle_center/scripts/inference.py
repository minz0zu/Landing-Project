import os
import cv2
import torch
import numpy as np
import glob
from ultralytics import YOLO
from torchvision.ops import nms

class YOLO2StageSlicer:
    def __init__(self, car_model_path, plate_model_path, device=0, margin_rate=0.2):
        self.car_model = YOLO(car_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
        self.margin_rate = margin_rate
        self.car_conf = 0.45
        self.plate_conf = 0.48
        self.iou_thres = 0.3

    def forward_process(self, frame):
        h, w, _ = frame.shape
        mid_h, mid_w = h // 2, w // 2
        m_h, m_w = int(mid_h * self.margin_rate), int(mid_w * self.margin_rate)

        slices = [
            frame[0 : mid_h + m_h, 0 : mid_w + m_w],
            frame[0 : mid_h + m_h, mid_w - m_w : w],
            frame[mid_h - m_h : h, 0 : mid_w + m_w],
            frame[mid_h - m_h : h, mid_w - m_w : w]
        ]
        
        offsets = [
            (0, 0), (mid_w - m_w, 0), (0, mid_h - m_h), (mid_w - m_w, mid_h - m_h)
        ]

        car_results = self.car_model.predict(source=slices, conf=self.car_conf, device=self.device, imgsz=640, verbose=False)
        
        all_plates_list = []
        for i, res in enumerate(car_results):
            if len(res.boxes) == 0: continue
            car_boxes = res.boxes.xyxy.cpu().numpy()
            off_x, off_y = offsets[i]

            for c_box in car_boxes:
                cx1, cy1, cx2, cy2 = map(int, c_box)
                car_crop = slices[i][cy1:cy2, cx1:cx2]
                if car_crop.size == 0: continue

                plate_results = self.plate_model.predict(source=car_crop, conf=self.plate_conf, device=self.device, imgsz=320, verbose=False)

                for p_res in plate_results:
                    if len(p_res.boxes) == 0: continue
                    p_boxes = p_res.boxes.xyxy.cpu().numpy()
                    p_confs = p_res.boxes.conf.cpu().numpy()
                    for pb, pc in zip(p_boxes, p_confs):
                        all_plates_list.append([pb[0] + cx1 + off_x, pb[1] + cy1 + off_y, pb[2] + cx1 + off_x, pb[3] + cy1 + off_y, pc])

        if not all_plates_list: return frame

        plates_arr = np.array(all_plates_list, dtype=np.float32)
        boxes_t = torch.from_numpy(plates_arr[:, :4])
        scores_t = torch.from_numpy(plates_arr[:, 4])
        keep = nms(boxes_t, scores_t, iou_threshold=self.iou_thres)
        
        for idx in keep:
            x1, y1, x2, y2, conf = plates_arr[idx]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(frame, f"plate {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return frame

def run_test_set_inference(car_model_path, plate_model_path, test_dir, save_dir):
    if not os.path.exists(test_dir):
        return

    print(f"폴더 내부 확인: {os.listdir(test_dir)[:3]}")
    
    os.makedirs(save_dir, exist_ok=True)
    slicer = YOLO2StageSlicer(car_model_path, plate_model_path)

    img_list = glob.glob(os.path.join(test_dir, "*.jpg"))

    for i, img_path in enumerate(img_list):
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        result_img = slicer.forward_process(frame)
        cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), result_img)
        
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{len(img_list)}] 완료...")

if __name__ == "__main__":
    CAR_MODEL = "/home/minji/landing_project/vehicle_center/train_results/Car/weights/best.pt"
    PLATE_MODEL = "/home/minji/landing_project/vehicle_center/train_results/SGD/weights/best.pt"
    TEST_SET_DIR = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/images"
    RESULT_SAVE_DIR = "/mnt/hdd_6tb/minji/inference_results/SGD_vehicle_center"

    run_test_set_inference(CAR_MODEL, PLATE_MODEL, TEST_SET_DIR, RESULT_SAVE_DIR)
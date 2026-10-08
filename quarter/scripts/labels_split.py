import os
import json
import glob
from copy import deepcopy

def parse_resolution(res_val):
    """
    지원 형태:
    1) "1920, 1080" / "1920,1080" / "1920x1080"
    2) ["1920", "1080"] / [1920, 1080]
    3) {"width":1920,"height":1080} 같은 dict(있으면)
    """
    # case 2) list/tuple
    if isinstance(res_val, (list, tuple)) and len(res_val) >= 2:
        try:
            w = int(res_val[0])
            h = int(res_val[1])
            return w, h
        except:
            return None

    # case 3) dict
    if isinstance(res_val, dict):
        for k_w, k_h in [("width", "height"), ("w", "h")]:
            if k_w in res_val and k_h in res_val:
                try:
                    return int(res_val[k_w]), int(res_val[k_h])
                except:
                    return None
        return None

    # case 1) string
    if isinstance(res_val, str):
        s = res_val.replace("X", "x").replace("x", ",").replace(" ", "")
        parts = s.split(",")
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except:
                return None

    return None


def xywh_to_xyxy(b):
    x, y, w, h = b
    return [x, y, x + w, y + h]

def xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return [x1, y1, x2 - x1, y2 - y1]

def intersect_xyxy(a, b):
    """
    a, b: [x1,y1,x2,y2]
    return: intersection [x1,y1,x2,y2] or None
    """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def round_int(v):
    # 이미지 리사이즈에서 int(round())를 썼으므로 bbox도 동일한 라운딩으로 맞춤
    return int(round(v))

def process_one_json(json_path: str, out_dir: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_info = data.get("Raw_Data_Info", {})
    res = parse_resolution(raw_info.get("resolution", ""))
    if res is None:
        raise ValueError(f"resolution 파싱 실패: {json_path} / Raw_Data_Info.resolution={raw_info.get('resolution')}")

    W, H = res
    half_w, half_h = W // 2, H // 2

    # 4분할 타일 영역(원본 이미지 좌표 기준) — 질문 이미지 코드와 동일
    # 0: 좌상, 1: 우상, 2: 좌하, 3: 우하
    tiles = [
        (0,      0,      half_w,  half_h),  # (x0,y0,x1,y1)
        (half_w, 0,      W,       half_h),
        (0,      half_h, half_w,  H),
        (half_w, half_h, W,       H),
    ]

    base_stem = os.path.splitext(os.path.basename(json_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    learning_info = data.get("Learning_Data_Info", {})
    annotations = learning_info.get("annotations", [])

    # annotations 구조가 예시처럼 [ {license_plate:[...], license_plate_number:[...]} ] 인 형태를 가정
    # 그래도 최대한 유연하게 처리: ann dict마다 해당 키를 있으면 처리
    for tile_idx, (tx0, ty0, tx1, ty1) in enumerate(tiles):
        tile_w = tx1 - tx0
        tile_h = ty1 - ty0

        # 이미지 저장 코드와 동일한 스케일
        scale = 640 / max(tile_h, tile_w)

        out_data = deepcopy(data)

        # 필요하면 식별자 갱신(선택 사항이지만, 파일명과 맞춰주는 게 관리에 유리)
        # out_data["Learning_Data_Info"]["json_data_ID"] 는 원본에 있을 때만 갱신
        out_learning = out_data.setdefault("Learning_Data_Info", {})
        if "json_data_ID" in out_learning:
            out_learning["json_data_ID"] = f"{base_stem}_{tile_idx}"

        # 타일 리사이즈 후 해상도 정보도 갱신(선택 사항)
        new_w = round_int(tile_w * scale)
        new_h = round_int(tile_h * scale)
        out_data.setdefault("Raw_Data_Info", {})
        out_data["Raw_Data_Info"]["resolution"] = f"{new_w}, {new_h}"

        new_annotations = []
        for ann in annotations:
            if not isinstance(ann, dict):
                continue

            new_ann = deepcopy(ann)

            # 처리 대상 키들
            for key in ["license_plate", "license_plate_number"]:
                items = ann.get(key, [])
                if not isinstance(items, list):
                    items = []

                new_items = []
                for obj in items:
                    if not isinstance(obj, dict):
                        continue
                    if "bbox" not in obj:
                        continue

                    bbox_xywh = obj["bbox"]
                    if not (isinstance(bbox_xywh, list) and len(bbox_xywh) == 4):
                        continue

                    x1, y1, x2, y2 = xywh_to_xyxy(bbox_xywh)

                    # 타일 영역과 교집합
                    inter = intersect_xyxy([x1, y1, x2, y2], [tx0, ty0, tx1, ty1])
                    if inter is None:
                        continue

                    # 타일 좌표계로 shift
                    inter[0] -= tx0
                    inter[1] -= ty0
                    inter[2] -= tx0
                    inter[3] -= ty0

                    # 리사이즈 스케일 적용
                    inter = [v * scale for v in inter]

                    # 최종 정수화 + 경계 클램프
                    ix1 = max(0, min(new_w, round_int(inter[0])))
                    iy1 = max(0, min(new_h, round_int(inter[1])))
                    ix2 = max(0, min(new_w, round_int(inter[2])))
                    iy2 = max(0, min(new_h, round_int(inter[3])))

                    # 너무 작거나 뒤집힌 박스 제거
                    if ix2 <= ix1 or iy2 <= iy1:
                        continue

                    new_obj = deepcopy(obj)
                    new_obj["bbox"] = xyxy_to_xywh([ix1, iy1, ix2, iy2])
                    new_items.append(new_obj)

                new_ann[key] = new_items

            new_annotations.append(new_ann)

        out_learning["annotations"] = new_annotations

        out_path = os.path.join(out_dir, f"{base_stem}_{tile_idx}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=4)

def split_labels_4tiles(original_label_dir: str, out_label_dir: str):
    json_files = glob.glob(os.path.join(original_label_dir, "**", "*.json"), recursive=True)
    print(f"Total {len(json_files)} json labels")

    os.makedirs(out_label_dir, exist_ok=True)

    for jp in json_files:
        try:
            process_one_json(jp, out_label_dir)
        except Exception as e:
            print(f"[WARN] fail: {jp} -> {e}")

if __name__ == "__main__":
    original_label_dir = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/labels"
    out_label_dir = "/mnt/hdd_6tb/minji/processed_quarter/Test/labels_quarter"

    split_labels_4tiles(original_label_dir, out_label_dir)
    print("Done.")
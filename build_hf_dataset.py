import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, Features, Value, Sequence, Image as HFImage
from huggingface_hub import login

DATASET_DIR = "/mnt/c/Users/bence/Downloads/m55m1data"
OUTPUT_DIR = "/home/bence/prepare-overhead-person-detection"
HF_TOKEN = "insert_here"
DATASET_NAME = "bdanko/overhead-person-detection"

folders = [
    "lift overhead detection.yolov8",
    "Overhead.yolov8",
    "overhead.yolov8 (1)",
    "People Detection Overhead V2.yolov8",
    "Top down people.yolov8"
]

def letterbox(img, new_shape=(192, 192), color=114):
    shape = img.shape[:2]  # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def adjust_box_letterbox(x_center, y_center, w, h, img_w, img_h, target_size=(192, 192)):
    x_abs, y_abs = x_center * img_w, y_center * img_h
    w_abs, h_abs = w * img_w, h * img_h
    
    r = min(target_size[0] / img_h, target_size[1] / img_w)
    new_w, new_h = int(round(img_w * r)), int(round(img_h * r))
    dw = (target_size[1] - new_w) / 2
    dh = (target_size[0] - new_h) / 2
    
    x_abs_new = x_abs * r + dw
    y_abs_new = y_abs * r + dh
    w_abs_new = w_abs * r
    h_abs_new = h_abs * r
    
    xmin = x_abs_new - w_abs_new / 2
    ymin = y_abs_new - h_abs_new / 2
    
    return [xmin, ymin, w_abs_new, h_abs_new]

def parse_label(filepath, img_w, img_h, class_map, dataset_folder):
    objects = []
    if not os.path.exists(filepath):
        return objects
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            
            # Use strict class mapping
            mapped_class = class_map.get(class_id, None)
            if mapped_class is None:
                continue
            
            if len(coords) == 4:
                # Bounding Box
                x_center, y_center, w, h = coords
            else:
                # Polygon/Segment
                xs = coords[::2]
                ys = coords[1::2]
                if not xs or not ys:
                    continue
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                w = xmax - xmin
                h = ymax - ymin
                x_center = (xmin + xmax) / 2
                y_center = (ymin + ymax) / 2
                
            bbox = adjust_box_letterbox(x_center, y_center, w, h, img_w, img_h)
            
            # BBox format: [xmin, ymin, width, height]
            objects.append({
                "bbox": bbox,
                "category": mapped_class
            })
    return objects

# Define Class Mappings
class_maps = {
    "lift overhead detection.yolov8": {0: 0},
    "Overhead.yolov8": {0: 0},
    "overhead.yolov8 (1)": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},  # Collapse All 5: 0, Man, Person, Woman, ero
    "People Detection Overhead V2.yolov8": {1: 0}, 
    "Top down people.yolov8": {0: 0}
}

samples_collected = 0
sample_images = []

login(HF_TOKEN)

data_items = []

for folder in folders:
    print(f"Processing folder: {folder}")
    path = os.path.join(DATASET_DIR, folder)
    class_map = class_maps.get(folder, {0: 0})
    
    img_dir = os.path.join(path, "train", "images")
    lbl_dir = os.path.join(path, "train", "labels")
    
    if not os.path.isdir(img_dir):
        print(f"No train/images found in {folder}, skipping.")
        continue
        
    img_files = os.listdir(img_dir)
    for img_name in tqdm(img_files):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        
        img_path = os.path.join(img_dir, img_name)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)
        
        # Load image via OpenCV
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Convert to Monochrome (Grayscale)
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Letterbox to 192x192
        # Use color=0 for black padding or 114 for gray. User specified "monochrome conversion", scalar 0 is fine
        img_lb, r, (dw, dh) = letterbox(gray, new_shape=(192, 192), color=0)
        
        # Parse labels
        objects = parse_label(lbl_path, w, h, class_map, folder)
        
        # Structure box fields for list Sequence
        bboxes = [obj["bbox"] for obj in objects]
        categories = [obj["category"] for obj in objects]
        
        # Convert to PIL Image for Datasets library
        pil_img = Image.fromarray(img_lb)
        
        # Collect Sample
        if samples_collected < 3 and len(bboxes) > 0 and folder in ["lift overhead detection.yolov8", "People Detection Overhead V2.yolov8", "Top down people.yolov8"]:
            samples_collected += 1
            # Save drawable image (BGR -> Gray -> BGR for boxes)
            img_with_boxes = cv2.cvtColor(img_lb, cv2.COLOR_GRAY2BGR)
            for bbox in bboxes:
                xmin, ymin, wb, hb = map(int, bbox)
                cv2.rectangle(img_with_boxes, (xmin, ymin), (xmin + wb, ymin + hb), (0, 255, 0), 1)
            
            sample_path = os.path.join(OUTPUT_DIR, "samples", f"sample_{samples_collected}.png")
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            cv2.imwrite(sample_path, img_with_boxes)
            print(f"Saved sample image to {sample_path}")
            
        data_items.append({
            "image": pil_img,
            "objects": {
                "bbox": bboxes,
                "category": categories
            }
        })
        
        # Log Progress
        if len(data_items) % 100 == 0:
            with open("/tmp/hf_dataset_progress.txt", "w") as f_log:
                f_log.write(f"Processed {len(data_items)} items")

print(f"Total items collected: {len(data_items)}")

# Create Features Schema
features = Features({
    "image": HFImage(),
    "objects": Sequence({
        "bbox": Sequence(Value("float32"), length=4),  # [xmin, ymin, width, height]
        "category": Value("int32")
    })
})

print("Creating Hugging Face Dataset...")
dataset = Dataset.from_list(data_items, features=features)

print(f"Uploading to Hugging Face Hub: {DATASET_NAME}...")
dataset.push_to_hub(DATASET_NAME, private=False)

print("Upload complete!")

with open("/tmp/hf_dataset_upload_complete.txt", "w") as f_done:
    f_done.write("DONE")

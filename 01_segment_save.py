import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import requests
from tqdm import tqdm
from ultralytics import YOLO

def load_sam_model(checkpoint_path, model_type="vit_h"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    return SamAutomaticMaskGenerator(sam)

def load_yolo_model(model_name="yolov8n.pt"):
    return YOLO(model_name)

def extract_human_bbox(yolo_results):
    """Extract bounding box for person class (class_id=0 in COCO)"""
    for result in yolo_results:
        for box in result.boxes:
            if int(box.cls) == 0:  # person class
                return box.xyxy[0].cpu().numpy().astype(int)
    return None

def process_images_sam(
    input_dir, 
    output_dir,
    sam_checkpoint_pth="./pretrained_models/sam_vit_h_4b8939.pth",
    model_type="vit_h",
    yolo_model_name="yolov8n.pt"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
    
    def download_sam_model(out_path="sam_vit_h_4b8939.pth"):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        if not os.path.exists(out_path):
            print("Downloading SAM model...")
            r = requests.get(url, stream=True)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download complete!")
        else:
            print("SAM Model already present.")
    
    download_sam_model(sam_checkpoint_pth)
    
    # Load models
    yolo = load_yolo_model(yolo_model_name)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_pth)
    sam.to(device="cuda")
    predictor = SamPredictor(sam)
    
    image_list = sorted([f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for img_name in tqdm(image_list):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect human with YOLO
        yolo_results = yolo(image_rgb)
        bbox = extract_human_bbox(yolo_results)
        
        if bbox is None:
            print(f"No person detected in {img_name}, skipping...")
            continue
        
        # Segment within human bbox using SAM
        predictor.set_image(image_rgb)
        x1, y1, x2, y2 = bbox
        box = np.array([x1, y1, x2, y2])
        
        masks, scores, _ = predictor.predict(box=box)
        best_mask = masks[np.argmax(scores)]
        
        h, w = image_rgb.shape[:2]
        fg_mask = best_mask.astype(np.uint8)
        
        foreground = image_rgb

        output_rgb_path = os.path.join(output_dir, 'images', img_name)
        Image.fromarray(foreground).save(output_rgb_path)
        
        alpha_path = os.path.join(output_dir, 'masks', img_name.replace(".", "_mask."))
        Image.fromarray((fg_mask * 255).astype(np.uint8)).save(alpha_path)

        print(f"Saved human segmentation for: {img_name}")

    print("All images processed!")

if __name__ == "__main__":
    process_images_sam(
        input_dir="./video_frames/8613",
        output_dir="./processed_images/8613",
        sam_checkpoint_pth="./pretrained_models/sam_vit_h_4b8939.pth",
        model_type="vit_h",
        yolo_model_name="yolov8n.pt"
    )

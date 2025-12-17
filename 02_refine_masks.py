import os
import cv2
import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
import requests
from tqdm import tqdm

def load_sam_model(checkpoint_path, model_type="vit_h"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    return SamAutomaticMaskGenerator(sam)

def enhance_contrast(image_rgb):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def extract_center_mask(masks, h, w, min_area_ratio=0.01):
   
    if len(masks) == 0:
        return np.ones((h, w), dtype=np.uint8)

    total_pixels = h * w
    min_area = total_pixels * min_area_ratio  # ignore masks smaller than this
    cy, cx = h // 2, w // 2

    best_idx = None
    best_score = 1e18

    for i, m in enumerate(masks):
        seg = m["segmentation"]

        area = seg.sum()
        if area < min_area:
            continue   # skip small useless masks

        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue

        mx, my = xs.mean(), ys.mean()  # mask centroid

        dist = (mx - cx)**2 + (my - cy)**2
        if dist < best_score:
            best_score = dist
            best_idx = i

    # If no mask passed the threshold, fall back to largest mask
    if best_idx is None:
        areas = [m["segmentation"].sum() for m in masks]
        best_idx = int(np.argmax(areas))

    return masks[best_idx]["segmentation"].astype(np.uint8)

def extract_largest_mask(masks, h, w):
    # Pick the mask with the largest area
    if len(masks) == 0:
        return np.ones((h, w), dtype=np.uint8)

    areas = [np.sum(m["segmentation"]) for m in masks]
    idx = np.argmax(areas)
    return masks[idx]["segmentation"].astype(np.uint8)

def extract_center_weighted_mask(masks, h, w, alpha=0.7):
    cx, cy = w // 2, h // 2
    best_idx = None
    best_score = -1e18

    for i, m in enumerate(masks):
        seg = m["segmentation"]
        area = seg.sum()

        ys, xs = np.where(seg)
        mx, my = xs.mean(), ys.mean()
        
        # compute closeness to center (smaller distance = better)
        dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
        center_score = 1.0 / (dist + 1e-6)

        # combine: prefer masks that are both big & centered
        score = alpha * area + (1 - alpha) * center_score

        if score > best_score:
            best_score = score
            best_idx = i

    return masks[best_idx]["segmentation"].astype(np.uint8)

def extract_center_region_mask(masks, h, w, box_ratio=0.5):
    cx, cy = w // 2, h // 2
    bw, bh = int(w * box_ratio), int(h * box_ratio)

    x1, x2 = cx - bw // 2, cx + bw // 2
    y1, y2 = cy - bh // 2, cy + bh // 2

    merged = np.zeros((h, w), dtype=np.uint8)

    for m in masks:
        seg = m["segmentation"]
        # If the mask overlaps our central box → keep it
        if seg[y1:y2, x1:x2].any():
            merged = merged | seg

    return merged.astype(np.uint8)
def process_images_sam(
    input_dir, 
    output_dir,
    sam_checkpoint_pth="./pretrained_models/sam_vit_h_4b8939.pth",
    model_type="vit_h",
    mask_numbers=None
):
    os.makedirs(output_dir, exist_ok=True)
    
    # download SAM to pretrainedmodel repo
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
      
    # mask_generator = load_sam_model(sam_checkpoint, model_type)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_pth)
    sam.to(device="cuda")
    
    predictor = SamPredictor(sam)
    
    mask_generator = SamAutomaticMaskGenerator(sam)


    image_list = sorted([os.path.join("frame_000"+str(x)+".png") for x in mask_numbers])
   
    for img_name in tqdm(image_list):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # predictor.set_image(image_rgb)
        
        h, w = image_rgb.shape[:2]
        cx, cy = w // 2, h // 2
        bw, bh = w // 4, h // 4
        box = np.array([cx - bw, cy - bh, cx + bw, cy + bh])
        
        x1, y1, x2, y2 = box
        
        
        # Run SAM
        masks = mask_generator.generate(image_rgb)
        # masks, scores, _ = predictor.predict(box=box)
        # best_mask = masks[np.argmax(scores)-1]
        
        # Extract largest connected object
        # fg_mask = extract_largest_mask(masks, h, w)
        # fg_mask = extract_center_region_mask(masks, h, w)
        fg_mask = extract_center_mask(masks, h, w, min_area_ratio=0.08)
        # fg_mask = best_mask
        # Convert to 3-channel mask
        fg_mask_3c = np.repeat(fg_mask[:, :, None], 3, axis=2)

        # Apply mask
        foreground = image_rgb * fg_mask_3c

        # Save masked image
        output_rgb_path = os.path.join(output_dir, img_name)
        Image.fromarray(foreground).save(output_rgb_path)
        
        # Draw rectangle on the image
        img_box = foreground.copy()
        cv2.rectangle(
            img_box,
            (x1, y1),        # top-left
            (x2, y2),        # bottom-right
            (0, 255, 0),     # green color
            3                # thickness
        )

        # Save or view
        cv2.imwrite(os.path.join(output_dir, img_name.split(".")[0]+"_bbox.png"), img_box)

        # Optional: save alpha mask
        alpha_path = os.path.join(output_dir, img_name.replace(".", "_mask."))
        Image.fromarray((fg_mask * 255).astype(np.uint8)).save(alpha_path)

        print(f"Saved foreground for: {img_name}")

    print("All images processed!")

if __name__ == "__main__":
    process_images_sam(
        input_dir="./video_frames/8573",
        output_dir="./masked_images/8573",
        sam_checkpoint_pth="./pretrained_models/sam_vit_h_4b8939.pth",
        model_type="vit_h",
        mask_numbers=[40, 56, 58]
    )
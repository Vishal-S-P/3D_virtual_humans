import os
import pickle
import numpy as np
import torch
import smplx
import trimesh
from sklearn.neighbors import NearestNeighbors

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PATHS (Update these if they changed)
SIMPLX_FITTED_PARAMS = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/smplx_fitted_params.pt"
SCAN_PATH            = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/mesh_poisson_rotated_and_dir.ply"
SIMPLX_VERTEX_COLORS = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/smplx_color_map.npy"
SIMPLX_MODEL_PATH    = "/depot/qqiu/data/vishal/3d_proj/pretrained_models/models/"

# MOTION FILE
MOYO_PKL = (
    "/depot/qqiu/data/vishal/3d_proj/pretrained_models/moyo/mosh/val/221004_yogi_nexus_body_hands_03596_Seated_Forward_Bend_pose_or_Paschimottanasana_-a_stageii.pkl"
)

OUT_DIR = "moyo_anim"
os.makedirs(OUT_DIR, exist_ok=True)

GENDER = "male"

# -------------------------------------------------
# 1. LOAD FITTED IDENTITY
# -------------------------------------------------
print(f"Loading identity from {SIMPLX_FITTED_PARAMS}...")
params = torch.load(SIMPLX_FITTED_PARAMS, map_location=device)

# Create Model
smplx_model = smplx.create(
    model_path=SIMPLX_MODEL_PATH,
    model_type="smplx",
    gender=GENDER,
    use_pca=False,
    flat_hand_mean=True,
    num_betas=10, 
    num_expression_coeffs=10
).to(device)

# A. Apply Shape (Betas)
betas = params["betas"].to(device).float().detach()


# B. Apply Scale
if "scale" in params:
    scale = params["scale"].to(device).float()
    print(f"-> Found scale: {scale.item():.4f}")
else:
    scale = torch.tensor(1.0, device=device)

# -------------------------------------------------
# 2. LOAD/EXTRACT COLORS
# -------------------------------------------------
print("Checking for texture...")
if os.path.exists(SIMPLX_VERTEX_COLORS):
    print(f"-> Loading existing color map: {SIMPLX_VERTEX_COLORS}")
    colors = np.load(SIMPLX_VERTEX_COLORS)
else:
    print("-> Color map NOT found. Extracting colors from Scan now...")
    if os.path.exists(SCAN_PATH):
        scan_mesh = trimesh.load(SCAN_PATH, process=False)
        # Normalize colors
        if scan_mesh.visual.vertex_colors.shape[1] == 4:
            scan_colors = scan_mesh.visual.vertex_colors[:, :3]
        else:
            scan_colors = scan_mesh.visual.vertex_colors
            
        if scan_colors.max() > 1.0:
            scan_colors = scan_colors / 255.0

        # Get Model Vertices (apply scale)
        with torch.no_grad():
            output = smplx_model(
                betas=betas, 
                global_orient=params.get('global_orient', torch.zeros(1,3).to(device)),
                body_pose=params.get('body_pose', torch.zeros(1,63).to(device)),
                transl=params.get('transl', torch.zeros(1,3).to(device)),
                return_verts=True
            )
            model_verts = (output.vertices[0] * scale).cpu().numpy()
        
        # Nearest Neighbor Transfer
        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(scan_mesh.vertices)
        _, indices = knn.kneighbors(model_verts)
        colors = scan_colors[indices[:, 0]]
        
        np.save(SIMPLX_VERTEX_COLORS, colors)
        print(f"   Saved new color map to: {SIMPLX_VERTEX_COLORS}")
    else:
        print("!! WARNING: Scan not found. Mesh will be white.")
        colors = np.ones((10475, 3)) * 0.8

# Convert to uint8 for Trimesh
if colors.max() <= 1.0:
    vertex_colors = (colors * 255).astype(np.uint8)
else:
    vertex_colors = colors.astype(np.uint8)

# -------------------------------------------------
# 3. LOAD MOTION (CORRECTED)
# -------------------------------------------------
print(f"Loading motion from {MOYO_PKL}...")
with open(MOYO_PKL, "rb") as f:
    moyo_data = pickle.load(f)

def prepare_tensor(data_dict, key, expected_dim):
    if key in data_dict:
        val = data_dict[key]
        tensor = torch.tensor(val, dtype=torch.float32).to(device)
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        return tensor
    else:
        # Debug warning if key is missing
        print(f"Warning: Key '{key}' not found. Using Zeros.")
        return torch.zeros((1, expected_dim), dtype=torch.float32).to(device)

# --- KEY FIX: Use 'fullpose' instead of 'poses' ---
full_pose = prepare_tensor(moyo_data, 'fullpose', 165)
bs = full_pose.shape[0]

print(f"DEBUG: Found {bs} frames of motion.")

# Slicing (Standard SMPL-X packing)
moyo_global_orient = full_pose[:, :3]
moyo_body_pose     = full_pose[:, 3:66]
moyo_jaw_pose      = full_pose[:, 66:69]
# indices 69-75 are usually eyes, we skip them or use them if needed
moyo_hand_pose     = full_pose[:, 75:] 

moyo_transl = prepare_tensor(moyo_data, 'trans', 3)

# Floor Alignment
print("Aligning to floor...")
with torch.no_grad():
    initial_output = smplx_model(
        betas=betas,
        body_pose=moyo_body_pose[0:1],
        global_orient=moyo_global_orient[0:1],
        transl=moyo_transl[0:1]
    )
    verts = initial_output.vertices * scale
    min_y = torch.min(verts[:, :, 1])
    floor_offset = -min_y

moyo_transl[:, 1] += floor_offset

# -------------------------------------------------
# 4. GENERATE AND SAVE OBJ FRAMES
# -------------------------------------------------
print(f"Processing {bs} frames and saving to '{OUT_DIR}'...")

for i in range(bs):
    
    with torch.no_grad():
        output = smplx_model(
            betas=betas,                   
            body_pose=moyo_body_pose[i:i+1],
            global_orient=moyo_global_orient[i:i+1],
            transl=moyo_transl[i:i+1],
            jaw_pose=moyo_jaw_pose[i:i+1],
            left_hand_pose=moyo_hand_pose[i:i+1, :45],
            right_hand_pose=moyo_hand_pose[i:i+1, 45:],
            return_verts=True
        )
    
    # Apply Scale
    frame_verts = (output.vertices.detach().cpu().numpy().squeeze()) * scale.cpu().numpy()
    
    # Save as OBJ
    filename = os.path.join(OUT_DIR, f'frame_{str(i).zfill(4)}.obj')
    
    # Indented correctly!
    trimesh.Trimesh(
        vertices=frame_verts, 
        faces=smplx_model.faces,
        vertex_colors=vertex_colors
    ).export(filename)
    
    if i % 100 == 0:
        print(f"Saved {i}/{bs} frames...")

print(f"Done. All frames saved in {os.path.abspath(OUT_DIR)}")
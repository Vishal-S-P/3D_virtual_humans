import torch
import numpy as np
import smplx
import trimesh
from sklearn.neighbors import NearestNeighbors
import os

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SCAN_PATH       = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/mesh_poisson_rotated_and_dir_normals_fixed.ply"
PARAMS_PATH     = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/smplx_fitted_params.pt"
MODEL_FOLDER    = "/depot/qqiu/data/vishal/3d_proj/pretrained_models/models/"
COLOR_MAP_OUT   = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/smplx_color_map.npy"

GENDER = "male"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# ROBUST COLOR EXTRACTION
# -----------------------------------------------------------------------------
print(f"Loading Scan: {SCAN_PATH}")
scan_mesh = trimesh.load(SCAN_PATH, process=False)

# 1. FORCE CENTER SCAN (Move to 0,0,0)
scan_min = scan_mesh.vertices.min(axis=0)
scan_max = scan_mesh.vertices.max(axis=0)
scan_center = (scan_min + scan_max) / 2.0
scan_verts_centered = scan_mesh.vertices - scan_center

# Normalize scan colors to 0-1
if scan_mesh.visual.vertex_colors.max() > 1.0:
    scan_colors = scan_mesh.visual.vertex_colors[:, :3] / 255.0
else:
    scan_colors = scan_mesh.visual.vertex_colors[:, :3]

print(f"Loading Params: {PARAMS_PATH}")
params = torch.load(PARAMS_PATH, map_location=DEVICE)

print("Creating SMPL-X Model...")
smplx_model = smplx.create(
    MODEL_FOLDER, model_type='smplx', gender=GENDER, use_pca=False, 
    num_betas=10, num_expression_coeffs=10
).to(DEVICE)

# 2. GENERATE MODEL MESH
with torch.no_grad():
    output = smplx_model(
        betas=params['betas'],
        global_orient=params['global_orient'],
        body_pose=params['body_pose'],
        transl=params['transl'],
        return_verts=True
    )
    # Apply scale
    scale = params.get('scale', torch.tensor([1.0], device=DEVICE))
    model_verts = output.vertices[0] * scale
    model_verts_np = model_verts.cpu().numpy()

# 3. FORCE CENTER MODEL (Move to 0,0,0)
# We ignore the 'transl' parameter mismatch by re-centering geometrically
model_min = model_verts_np.min(axis=0)
model_max = model_verts_np.max(axis=0)
model_center = (model_min + model_max) / 2.0
model_verts_centered = model_verts_np - model_center

print("--- Alignment Check ---")
print(f"Scan  Center (Local): {np.mean(scan_verts_centered, axis=0)}")
print(f"Model Center (Local): {np.mean(model_verts_centered, axis=0)}")
print("-> Both should be effectively (0,0,0). Proceeding with projection...")

# 4. NEAREST NEIGHBOR (On Centered Meshes)
print("Projecting textures...")
knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(scan_verts_centered)
_, indices = knn.kneighbors(model_verts_centered)

# 5. SAVE
smplx_colors = scan_colors[indices[:, 0]]
np.save(COLOR_MAP_OUT, smplx_colors)
print(f"[SUCCESS] Saved robust color map to {COLOR_MAP_OUT}")
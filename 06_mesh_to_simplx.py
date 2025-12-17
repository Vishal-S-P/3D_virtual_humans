# import torch
# import numpy as np
# import smplx
# import trimesh
# from sklearn.neighbors import NearestNeighbors
# import os

# # -----------------------------------------------------------------------------
# # CONFIGURATION
# # -----------------------------------------------------------------------------
# # SCAN_PATH       = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/meshed-poisson_preprocessed.ply"
# SCAN_PATH       = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/scan_smplx_aligned.ply"
# MODEL_FOLDER    = "/depot/qqiu/data/vishal/3d_proj/pretrained_models/models/"
# OUTPUT_PATH     = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/smplx_fitted_params.pt"

# GENDER = "male"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # -----------------------------------------------------------------------------
# # FIXED LOSS FUNCTION (CONNECTS GRADIENTS)
# # -----------------------------------------------------------------------------
# def get_chamfer_loss(scan_verts, model_verts):
#     """
#     Computes bidirectional Chamfer distance while maintaining gradient flow.
#     """
#     # 1. Scan -> Model (find closest model vertex for each scan vertex)
#     # We use sklearn for the SEARCH (non-differentiable), but PyTorch for the DISTANCE (differentiable)
    
#     # Detach for sklearn search
#     scan_np = scan_verts.detach().cpu().numpy()
#     model_np = model_verts.detach().cpu().numpy()

#     # S2M: Find nearest model vert for every scan vert
#     nbrs_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(model_np)
#     _, idx_s2m = nbrs_model.kneighbors(scan_np)
#     idx_s2m = torch.tensor(idx_s2m.flatten(), device=DEVICE).long()
    
#     # CRITICAL: Gather the differentiable model vertices using the indices
#     nearest_model_verts = model_verts[idx_s2m] 
#     # Compute distance in PyTorch (Gradients flow here!)
#     s2m_loss = torch.mean(torch.sum((scan_verts - nearest_model_verts) ** 2, dim=1))

#     # M2S: Find nearest scan vert for every model vert
#     nbrs_scan = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(scan_np)
#     _, idx_m2s = nbrs_scan.kneighbors(model_np)
#     idx_m2s = torch.tensor(idx_m2s.flatten(), device=DEVICE).long()
    
#     nearest_scan_verts = scan_verts[idx_m2s]
#     # Compute distance in PyTorch
#     m2s_loss = torch.mean(torch.sum((model_verts - nearest_scan_verts) ** 2, dim=1))

#     return s2m_loss + m2s_loss

# # -----------------------------------------------------------------------------
# # INITIALIZATION
# # -----------------------------------------------------------------------------
# print(f"--- Loading Scan: {SCAN_PATH} ---")
# scan_mesh = trimesh.load(SCAN_PATH, process=False)
# scan_center = scan_mesh.vertices.mean(axis=0)
# scan_mesh.vertices -= scan_center
# scan_verts = torch.tensor(scan_mesh.vertices, dtype=torch.float32, device=DEVICE)

# print("--- Initializing SMPL-X ---")
# smplx_model = smplx.create(
#     MODEL_FOLDER, 
#     model_type='smplx',
#     gender=GENDER, 
#     use_pca=False, 
#     flat_hand_mean=True,
#     num_betas=10, 
#     num_expression_coeffs=10
# ).to(DEVICE)

# # --- PARAMETERS ---
# betas = torch.zeros(1, 10, device=DEVICE, requires_grad=True)
# body_pose       = torch.zeros(1, 63, device=DEVICE, requires_grad=True)
# left_hand_pose  = torch.zeros(1, 45, device=DEVICE, requires_grad=True)
# right_hand_pose = torch.zeros(1, 45, device=DEVICE, requires_grad=True)
# jaw_pose   = torch.zeros(1, 3, device=DEVICE, requires_grad=True)
# expression = torch.zeros(1, 10, device=DEVICE, requires_grad=True)
# global_orient = torch.zeros(1, 3, device=DEVICE, requires_grad=True)
# transl        = torch.zeros(1, 3, device=DEVICE, requires_grad=True)
# scale = torch.tensor([1.0], device=DEVICE, requires_grad=True)

# # -----------------------------------------------------------------------------
# # OPTIMIZATION LOOP
# # -----------------------------------------------------------------------------
# def run_stage(stage_name, params_list, lr, iterations, lambda_pose=0.0):
#     print(f"\n[Stage: {stage_name}] Iter: {iterations} | LR: {lr} | Lambda_Pose: {lambda_pose}")
    
#     optimizer = torch.optim.Adam(params_list, lr=lr)
    
#     for i in range(iterations):
#         optimizer.zero_grad()
        
#         # Forward Pass
#         output = smplx_model(
#             betas=betas, 
#             global_orient=global_orient,
#             body_pose=body_pose,
#             left_hand_pose=left_hand_pose,
#             right_hand_pose=right_hand_pose,
#             jaw_pose=jaw_pose,
#             expression=expression,
#             transl=transl,
#             return_verts=True
#         )
#         model_verts = output.vertices[0] * scale
        
#         # 1. Data Term (Chamfer) - NOW WITH GRADIENTS
#         loss_chamfer = get_chamfer_loss(scan_verts, model_verts)
        
#         # 2. Regularizers
#         loss_shape = (torch.mean(betas**2) + torch.mean(expression**2)) * 0.01 
#         loss_body = torch.mean(body_pose**2) * lambda_pose
#         hand_reg_weight = lambda_pose if lambda_pose > 1 else 1.0
#         loss_hands_jaw = (torch.mean(left_hand_pose**2) + 
#                           torch.mean(right_hand_pose**2) + 
#                           torch.mean(jaw_pose**2)) * hand_reg_weight

#         total_loss = loss_chamfer + loss_shape + loss_body + loss_hands_jaw
        
#         total_loss.backward()
#         optimizer.step()
        
#         if i % 20 == 0: # Print more often
#             print(f"  Step {i:03d}: Loss={total_loss.item():.5f} (Chamfer={loss_chamfer.item():.5f})")

# # Stage 1: Global 
# run_stage("1 Global", [global_orient, transl, scale], lr=0.01, iterations=150, lambda_pose=0.0)

# # Stage 2: Shape 
# run_stage("2 Shape", [betas, scale], lr=0.01, iterations=250, lambda_pose=0.0)

# # Stage 3: Pose Up 
# run_stage("3 Pose(High)", [body_pose], lr=0.005, iterations=300, lambda_pose=100.0)

# # Stage 4: Pose Down 
# run_stage("4 Pose(Low)", [body_pose, global_orient], lr=0.002, iterations=400, lambda_pose=10.0)

# # Stage 5: All 
# run_stage("5 All", 
#           [betas, body_pose, global_orient, transl, scale, 
#            left_hand_pose, right_hand_pose, jaw_pose, expression], 
#           lr=0.001, iterations=200, lambda_pose=1.0)

# # -----------------------------------------------------------------------------
# # SAVE OUTPUT
# # -----------------------------------------------------------------------------
# final_transl = transl.detach().cpu() + torch.tensor(scan_center, dtype=torch.float32)

# final_params = {
#     "betas": betas.detach().cpu(),
#     "global_orient": global_orient.detach().cpu(),
#     "body_pose": body_pose.detach().cpu(),
#     "left_hand_pose": left_hand_pose.detach().cpu(),
#     "right_hand_pose": right_hand_pose.detach().cpu(),
#     "jaw_pose": jaw_pose.detach().cpu(),
#     "expression": expression.detach().cpu(),
#     "transl": final_transl,
#     "scale": scale.detach().cpu()
# }

# torch.save(final_params, OUTPUT_PATH)
# print(f"\n[SUCCESS] Saved fitted params to: {OUTPUT_PATH}")

# print("Exporting debug_fit.ply (visual check)...")
# with torch.no_grad():
#     smplx_model.cpu()
#     # Re-run forward pass on CPU to get verts
#     output = smplx_model(
#         betas=final_params["betas"],
#         global_orient=final_params["global_orient"],
#         body_pose=final_params["body_pose"],
#         left_hand_pose=final_params["left_hand_pose"],
#         right_hand_pose=final_params["right_hand_pose"],
#         jaw_pose=final_params["jaw_pose"],
#         expression=final_params["expression"],
#         transl=final_params["transl"] - torch.tensor(scan_center), 
#         return_verts=True
#     )
#     # Apply Scale & Trans
#     verts = (output.vertices[0] * final_params["scale"]) + torch.tensor(scan_center)
    
#     # Save
#     mesh = trimesh.Trimesh(verts.numpy(), smplx_model.faces, process=False)
#     mesh.export("debug_fit.ply")

# print("Done. Open 'debug_fit.ply' in MeshLab to verify alignment.")

import torch
import numpy as np
import smplx
import trimesh
from sklearn.neighbors import NearestNeighbors
import os

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Use the UPRIGHT scan
SCAN_PATH       = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/mesh_poisson_rotated_and_dir.ply"
MODEL_FOLDER    = "/depot/qqiu/data/vishal/3d_proj/pretrained_models/models/"
OUTPUT_PATH     = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/smplx_fitted_params.pt"

GENDER = "male"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# HELPER: LOSS
# -----------------------------------------------------------------------------
def get_chamfer_loss(scan_verts, model_verts):
    scan_np = scan_verts.detach().cpu().numpy()
    model_np = model_verts.detach().cpu().numpy()

    # S2M
    nbrs_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(model_np)
    _, idx_s2m = nbrs_model.kneighbors(scan_np)
    idx_s2m = torch.tensor(idx_s2m.flatten(), device=DEVICE).long()
    nearest_model = model_verts[idx_s2m] 
    s2m_loss = torch.mean(torch.sum((scan_verts - nearest_model) ** 2, dim=1))

    # M2S
    nbrs_scan = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(scan_np)
    _, idx_m2s = nbrs_scan.kneighbors(model_np)
    idx_m2s = torch.tensor(idx_m2s.flatten(), device=DEVICE).long()
    nearest_scan = scan_verts[idx_m2s]
    m2s_loss = torch.mean(torch.sum((model_verts - nearest_scan) ** 2, dim=1))

    return s2m_loss + m2s_loss

# -----------------------------------------------------------------------------
# INITIALIZATION
# -----------------------------------------------------------------------------
print(f"--- Loading Scan: {SCAN_PATH} ---")
scan_mesh = trimesh.load(SCAN_PATH, process=False)

# --- FIX: BOUNDING BOX CENTERING ---
# Calculating Mean (old way) causes vertical offsets if point density varies.
# Calculating Bounding Box Center (new way) snaps the middles together.
min_xyz = scan_mesh.vertices.min(axis=0)
max_xyz = scan_mesh.vertices.max(axis=0)
scan_center = (min_xyz + max_xyz) / 2.0

scan_mesh.vertices -= scan_center # Center scan at 0,0,0
scan_verts = torch.tensor(scan_mesh.vertices, dtype=torch.float32, device=DEVICE)
print(f"Centered Scan. Shifted by {-scan_center}")

print("--- Initializing SMPL-X ---")
smplx_model = smplx.create(
    MODEL_FOLDER, model_type='smplx', gender=GENDER, use_pca=False, 
    flat_hand_mean=True, num_betas=10, num_expression_coeffs=10
).to(DEVICE)

# --- PARAMETERS (Back to T-Pose Init) ---
betas = torch.zeros(1, 10, device=DEVICE, requires_grad=True)
global_orient = torch.zeros(1, 3, device=DEVICE, requires_grad=True)
transl        = torch.zeros(1, 3, device=DEVICE, requires_grad=True)
scale         = torch.tensor([1.0], device=DEVICE, requires_grad=True)
body_pose     = torch.zeros(1, 63, device=DEVICE, requires_grad=True) # T-Pose default

left_hand_pose  = torch.zeros(1, 45, device=DEVICE, requires_grad=True)
right_hand_pose = torch.zeros(1, 45, device=DEVICE, requires_grad=True)
jaw_pose        = torch.zeros(1, 3, device=DEVICE, requires_grad=True)
expression      = torch.zeros(1, 10, device=DEVICE, requires_grad=True)

# -----------------------------------------------------------------------------
# OPTIMIZATION LOOP
# -----------------------------------------------------------------------------
def run_stage(stage_name, params_list, lr, iterations, lambda_pose=0.0):
    print(f"\n[Stage: {stage_name}] Iter: {iterations} | LR: {lr} | Lambda_Pose: {lambda_pose}")
    optimizer = torch.optim.Adam(params_list, lr=lr)
    
    for i in range(iterations):
        optimizer.zero_grad()
        
        output = smplx_model(
            betas=betas, global_orient=global_orient, body_pose=body_pose,
            left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
            jaw_pose=jaw_pose, expression=expression, transl=transl,
            return_verts=True
        )
        model_verts = output.vertices[0] * scale
        
        loss_chamfer = get_chamfer_loss(scan_verts, model_verts)
        loss_betas = torch.mean(betas**2) * 0.01 
        loss_body  = torch.mean(body_pose**2) * lambda_pose
        loss_hands = (torch.mean(left_hand_pose**2) + torch.mean(right_hand_pose**2)) * 1.0
        
        total_loss = loss_chamfer + loss_betas + loss_body + loss_hands
        
        total_loss.backward()
        optimizer.step()
        
        if i % 10 == 0: 
            print(f"  Step {i:03d}: Loss={total_loss.item():.5f} (Chamfer={loss_chamfer.item():.5f})")

# Stage 1: Global (Orient, Trans, Scale)
run_stage("1 Global", [global_orient, transl, scale], lr=0.01, iterations=150, lambda_pose=0.0)

# Stage 2: Shape
run_stage("2 Shape", [betas, scale], lr=0.01, iterations=250, lambda_pose=0.0)

# Stage 3: Pose (High Reg - Forces T-Pose compliance initially)
run_stage("3 Pose(High)", [body_pose], lr=0.005, iterations=300, lambda_pose=100.0)

# Stage 4: Pose (Low Reg - Allows arms to drop to A-pose)
run_stage("4 Pose(Low)", [body_pose, global_orient], lr=0.002, iterations=400, lambda_pose=10.0)

# Stage 5: All
run_stage("5 All", [betas, body_pose, global_orient, transl, scale], lr=0.001, iterations=200, lambda_pose=0.01)

# -----------------------------------------------------------------------------
# SAVE
# -----------------------------------------------------------------------------
final_transl = transl.detach().cpu() + torch.tensor(scan_center, dtype=torch.float32)

final_params = {
    "betas": betas.detach().cpu(),
    "global_orient": global_orient.detach().cpu(),
    "body_pose": body_pose.detach().cpu(),
    "left_hand_pose": left_hand_pose.detach().cpu(),
    "right_hand_pose": right_hand_pose.detach().cpu(),
    "jaw_pose": jaw_pose.detach().cpu(),
    "expression": expression.detach().cpu(),
    "transl": final_transl,
    "scale": scale.detach().cpu()
}

torch.save(final_params, OUTPUT_PATH)
print(f"\n[SUCCESS] Fitted params saved to: {OUTPUT_PATH}")

# DEBUG MESH
print("Exporting debug_fit.ply...")
with torch.no_grad():
    smplx_model.cpu()
    output = smplx_model(
        betas=final_params["betas"],
        global_orient=final_params["global_orient"],
        body_pose=final_params["body_pose"],
        left_hand_pose=final_params["left_hand_pose"],
        right_hand_pose=final_params["right_hand_pose"],
        transl=final_params["transl"] - torch.tensor(scan_center), 
        return_verts=True
    )
    verts = (output.vertices[0] * final_params["scale"]) + torch.tensor(scan_center)
    mesh = trimesh.Trimesh(verts.numpy(), smplx_model.faces, process=False)
    mesh.export("debug_fit.ply")
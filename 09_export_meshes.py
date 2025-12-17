import os
# FORCE EGL (Fixes 'NoSuchDisplayException')
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import trimesh
import pyrender
import numpy as np
import glob

# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------
OBJ_FOLDER = "moyo_anim"        
OUTPUT_VIDEO = "final_animation_fixed.mp4"
FPS = 30                        
RESOLUTION = (800, 800)         

# -------------------------------------------------
# 1. LOAD DATA & CALCULATE CENTROID
# -------------------------------------------------
files = sorted(glob.glob(os.path.join(OBJ_FOLDER, "*.obj")))
if not files:
    print("No OBJ files found!")
    exit()

print(f"Found {len(files)} frames. Analyzing geometry...")

# Load first frame to find location
temp_mesh = trimesh.load(files[0], process=False)

# --- FIX 1: ROTATE UPRIGHT ---
# Rotate -90 degrees around X-axis to stand the character up
# (From Z-up to Y-up)
rot_matrix = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
temp_mesh.apply_transform(rot_matrix)

# Calculate new center after rotation
centroid = temp_mesh.centroid 
print(f"DEBUG: Character center is at: {centroid}")

# -------------------------------------------------
# 2. SETUP SCENE
# -------------------------------------------------
scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0], ambient_light=[0.3, 0.3, 0.3])

# --- FIX 2: AUTO-CAMERA PLACEMENT ---
# Place camera at character's X, Y (plus slight height offset), and back 2.5m in Z
camera_x = centroid[0]
camera_y = centroid[1] + 0.2  # Look slightly above waist
camera_z = centroid[2] + 2.5  # Distance from character

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
camera_pose = np.eye(4)
camera_pose[:3, 3] = [camera_x, camera_y, camera_z] 

print(f"DEBUG: Camera placed at: {camera_pose[:3, 3]}")
camera_node = scene.add(camera, pose=camera_pose)

# Light Setup (Attached to camera position so character is always lit)
light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
light_pose = np.eye(4)
light_pose[:3, 3] = [camera_x, camera_y + 2.0, camera_z + 2.0]
scene.add(light, pose=light_pose)

# Renderer
r = pyrender.OffscreenRenderer(viewport_width=RESOLUTION[0], viewport_height=RESOLUTION[1])
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, RESOLUTION)

# -------------------------------------------------
# 3. RENDER LOOP
# -------------------------------------------------
mesh_node = None

for i, filename in enumerate(files):
    
    # Load Mesh
    tm_mesh = trimesh.load(filename, process=False) 
    
    # APPLY THE SAME ROTATION TO EVERY FRAME
    tm_mesh.apply_transform(rot_matrix)

    # Convert to Render Mesh
    mesh = pyrender.Mesh.from_trimesh(tm_mesh)

    # Update Scene
    if mesh_node is not None:
        scene.remove_node(mesh_node)
    mesh_node = scene.add(mesh)

    # Render & Save
    color, depth = r.render(scene)
    frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    video.write(frame)

    if i % 50 == 0:
        print(f"Rendered frame {i}/{len(files)}")

video.release()
r.delete()
print(f"Video saved as: {OUTPUT_VIDEO}")
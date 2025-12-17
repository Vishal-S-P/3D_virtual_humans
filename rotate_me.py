import trimesh
import numpy as np

SCAN_PATH = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/meshed-poisson_preprocessed_new.ply"
OUT_PATH  = "/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/mesh_poisson_rotated_and_dir_normals_fixed.ply"

# mesh = trimesh.load(SCAN_PATH, process=False)


# # COLMAP→SMPL-X coords
# verts = mesh.vertices.copy()
# verts[:, 1] *= -1  # Y-flip
# mesh.vertices = verts
# mesh.fix_normals()

# mesh.export(OUT_PATH)

# print("Saved SMPL-X aligned scan:")
# print(OUT_PATH)
# print("Verify in MeshLab:")
# print("- Standing upright")
# print("- Facing +Z")
# print("- Feet on Y=0")

# Load mesh
mesh = trimesh.load(SCAN_PATH, process=False)

# 1. Flip Y Axis (Geometry)
verts = mesh.vertices.copy()
verts[:, 1] *= -1 
mesh.vertices = verts

# 2. Fix Winding Order
# Because we mirrored the geometry, we must flip the triangle winding 
# so the faces point "out" instead of "in".
mesh.faces = np.fliplr(mesh.faces)

# 3. CRITICAL STEP: Merge Vertices for Smooth Shading
# COLMAP meshes often have disconnected triangles. 
# We must merge them to calculate smooth vertex normals.
mesh.merge_vertices()

# 4. Recalculate Normals
# Now that vertices are merged, this will generate smooth normals
# instead of flat/faceted normals.
mesh.fix_normals()

# 5. Export
mesh.export(OUT_PATH, file_type='ply')

print("Saved with Merged Vertices & Smooth Normals:")
print(OUT_PATH)
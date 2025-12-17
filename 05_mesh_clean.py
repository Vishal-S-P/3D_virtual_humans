import open3d as o3d
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time

def load_and_clean_geometry(mesh_path, target_vertices=150000):
    """
    Loads mesh, cleans, DECIMATES, and preserves colors.
    """
    print(f"\n[1/5] Loading and Cleaning: {mesh_path}")
    t_start = time.time()
    
    # 1. Load via Trimesh (Slow but robust for parsing)
    tm_raw = trimesh.load(mesh_path, process=False)
    print(f"   -> Raw mesh loaded: {len(tm_raw.vertices)} vertices. ({time.time() - t_start:.2f}s)")
    
    # Check colors
    if hasattr(tm_raw.visual, 'vertex_colors') and len(tm_raw.visual.vertex_colors) > 0:
        raw_colors = np.asarray(tm_raw.visual.vertex_colors)[:, :3] 
        has_colors = True
    else:
        raw_colors = None
        has_colors = False
        print("   -> Warning: No vertex colors found.")

    raw_vertices = np.asarray(tm_raw.vertices)

    # 2. Convert to Open3D
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(raw_vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tm_raw.faces))

    # ---- Cleaning Operations ----
    # t_clean = time.time()
    # o3d_mesh.remove_duplicated_vertices()
    # o3d_mesh.remove_duplicated_triangles()
    # o3d_mesh.remove_degenerate_triangles()
    
    # # Outlier Removal
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d_mesh.vertices
    # _, ind = pcd.remove_radius_outlier(nb_points=4, radius=0.05)
    # o3d_mesh = o3d_mesh.select_by_index(ind)
    
    # ---- NEW: Decimation (Simplification) ----
    current_verts = len(o3d_mesh.vertices)
    if current_verts > target_vertices:
        print(f"   -> Decimating from {current_verts} to ~{target_vertices} vertices...")
        # Target triangles approx 2x vertices
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_vertices * 2)
        print(f"   -> Decimation complete: {len(o3d_mesh.vertices)} vertices.")

    # print(f"   -> Geometry cleaned & reduced. ({time.time() - t_clean:.2f}s)")

    # 3. Transfer Colors (KNN)
    # Note: We fit on the HUGE raw mesh, but query with the SMALL clean mesh.
    # This is still heavy RAM usage but much faster than querying 5M points.
    if has_colors:
        t_knn = time.time()
        print("   -> Starting KNN color transfer (building tree on original high-res mesh)...")
        
        # Fit on 5.5M points (Heavy RAM, take care)
        # n_jobs=-1 uses all CPU cores
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1).fit(raw_vertices)
        
        # Query with the NEW (smaller) vertices
        new_vertices = np.asarray(o3d_mesh.vertices)
        if len(new_vertices) > 0:
            _, indices = nbrs.kneighbors(new_vertices)
            new_colors = raw_colors[indices[:, 0]]
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors / 255.0)
        
        print(f"   -> Color transfer finished. ({time.time() - t_knn:.2f}s)")

    # 4. Convert back to Trimesh
    clean_mesh = trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        vertex_colors=(np.asarray(o3d_mesh.vertex_colors) * 255).astype(np.uint8) if has_colors else None,
        process=False
    )
    
    return clean_mesh

def keep_largest_component(mesh):
    """
    Keeps only the largest connected mesh chunk.
    """
    print("\n[2/5] Filtering largest component...")
    t_start = time.time()
    
    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        return mesh
    
    largest = max(components, key=lambda m: len(m.faces))
    print(f"   -> Kept largest of {len(components)} components. ({time.time() - t_start:.2f}s)")
    return largest

def fix_normals(mesh):
    """
    Fixes winding order and normals.
    """
    print("\n[3/5] Fixing normals...")
    mesh.rezero() 
    mesh.fix_normals()
    return mesh

def taubin_smooth(mesh, iters=5, lam=0.5, mu=-0.53):
    """
    Smooths geometry without shrinking.
    """
    print(f"\n[4/5] Smoothing (Taubin, {iters} iters)...")
    t_start = time.time()
    
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )
    
    o3d_mesh = o3d_mesh.filter_smooth_taubin(
        number_of_iterations=iters,
        lambda_filter=lam,
        mu=mu
    )
    
    print(f"   -> Smoothing done. ({time.time() - t_start:.2f}s)")
    
    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        vertex_colors=mesh.visual.vertex_colors,
        process=False
    )
    

def normalize_scale(mesh, target_height=1.72):
    print("\n[5/5] Normalizing Scale & Orientation...")
    bounds = mesh.bounds 
    extents = bounds[1] - bounds[0] 
    current_height = extents[1] 
    
    print(f"   -> Original Height (Y-axis): {current_height:.3f}m")
    
    if current_height < 0.1:
        print("   -> Warning: Mesh is extremely small or flattened.")

    scale = target_height / current_height
    mesh.apply_scale(scale)
    print(f"   -> Applied Scale: {scale:.3f}")
    return mesh

def preprocess_mesh(mesh_path, out_path):
    total_start = time.time()
    
    # 1. Load and clean
    mesh = load_and_clean_geometry(mesh_path, target_vertices=100000)
    
    # 2. Filtering
    mesh = keep_largest_component(mesh)
    
    # 3. Normals
    mesh = fix_normals(mesh)
    
    # 4. Smoothing
    mesh = taubin_smooth(mesh)
    
    # 5. Scale/Rotate (Combined for logging clarity)
    mesh = normalize_scale(mesh, target_height=1.68)
    
    
    print(f"\nSaving to {out_path}...")
    mesh.export(out_path)
    
    print(f"\n--- Total Processing Time: {time.time() - total_start:.2f}s ---")
    return mesh

if __name__ == "__main__":
    preprocess_mesh(
        mesh_path="./colmap_py/8613/mvs/meshed-poisson_cleaned.ply",
        out_path="./colmap_py/8613/mvs/meshed-poisson_preprocessed_new.ply"
    )


import os
import pycolmap
import subprocess
import trimesh
import numpy as np



def run_pmesh_reconstruction(poisson_input_mesh_path, cleaned_mesh_path):
    
    print("Running Poisson meshing...")
    # You can optionally tune options, e.g. depth, trim, etc.
    opts = pycolmap.PoissonMeshingOptions()
    opts.depth = 12         # try 10–11; 8 is coarse, 12+ is heavy
    # opts.point_weight = 8.0  # pulls surface closer to points; 2–8 is typical
    opts.trim = 4        # 5–10; higher trims more low-density regions

    pycolmap.poisson_meshing(poisson_input_mesh_path, 
                             cleaned_mesh_path, 
                             opts)

    pycolmap.poisson_meshing(
        input_path=poisson_input_mesh_path,
        output_path=cleaned_mesh_path,
        options=opts
    )

    print(f"Poisson mesh saved to: {cleaned_mesh_path}")
    

if __name__ == "__main__":
    run_pmesh_reconstruction(
    poisson_input_mesh_path="/depot/qqiu/data/vishal/3d_proj/colmap_py/8613/mvs/dense_me_cleaned.ply",
    cleaned_mesh_path="./colmap_py/8613/mvs/meshed-poisson_cleaned.ply"
    )
    
    
    
    
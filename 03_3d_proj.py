import os
import pycolmap
import subprocess
import trimesh
import numpy as np


def run_pycolmap_reconstruction(image_dir, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    database_path = os.path.join(output_dir, "database.db")
    sparse_output = os.path.join(output_dir, "sparse")
    mvs_output =  os.path.join(output_dir, "mvs")
    
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.mask_path = mask_dir
    
    # 1. Feature Extraction
    print("Extracting features...")
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.AUTO,
        reader_options=reader_opts,
        camera_model="OPENCV"
    )

    # 2. Feature Matching (older PyCOLMAP style)
    print("Matching features... (sequential)")
    # matcher = pycolmap.SequentialMatcher(
    #     database_path=database_path,
    #     match_options=pycolmap.FeatureMatchingOptions()
    # )
    # matcher.match()
    
    # pycolmap.match_exhaustive(database_path)
    pycolmap.match_sequential(
        database_path,
        )
    
    # 3. Incremental Mapping
    print("Running SfM reconstruction...")
    reconstruction = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_output,
    )
   
    print(f"Reconstruction saved to: {sparse_output}")
  
    pycolmap.undistort_images(
        output_path=mvs_output, 
        input_path=os.path.join(sparse_output, "0"),
        image_path=image_dir)
    
    pycolmap.patch_match_stereo(mvs_output)
    
    sf_opts = pycolmap.StereoFusionOptions()
    sf_opts.mask_path = mask_dir
    # Require more consistent depth observations per voxel
    print("Fusing depth maps into a dense point cloud...")
    pycolmap.stereo_fusion(output_path=os.path.join(mvs_output, "dense.ply"), 
                           workspace_path=mvs_output,
                           workspace_format="COLMAP",
                           input_type="geometric",
                           options=sf_opts
                           )
    
    print("Running Poisson meshing...")
    poisson_mesh_path = os.path.join(mvs_output, "meshed-poisson.ply")

    # You can optionally tune options, e.g. depth, trim, etc.
    opts = pycolmap.PoissonMeshingOptions()
    opts.depth = 16          # try 10–11; 8 is coarse, 12+ is heavy
    # opts.point_weight = 4.0  # pulls surface closer to points; 2–8 is typical
    # opts.trim = 12        # 5–10; higher trims more low-density regions

    pycolmap.poisson_meshing(os.path.join(mvs_output, "dense.ply"), 
                             poisson_mesh_path, 
                             opts)

    pycolmap.poisson_meshing(
        input_path=os.path.join(mvs_output, "dense.ply"),
        output_path=poisson_mesh_path,
        options=opts
    )

    print(f"Poisson mesh saved to: {poisson_mesh_path}")
    

if __name__ == "__main__":
    # run_pycolmap_reconstruction(
    # image_dir="processed_images/8613/images",
    # mask_dir="processed_images/8613/masks",
    # output_dir="./colmap_py/8613"
    # )
    
   
    
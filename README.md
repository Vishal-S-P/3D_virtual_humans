Human Motion Retargeting from RGB Images

This repository presents an end-to-end pipeline for human motion retargeting, starting from raw RGB images and producing a fully animated, identity-preserving 3D human avatar. The system integrates multi-view 3D reconstruction, parametric human modeling with SMPL-X, appearance transfer, and motion retargeting using real motion data.

The project was developed as part of STAT 59800: 3D Computer Vision & Virtual Human Models.

⸻

✨ Highlights
	•	📸 Image-based 3D reconstruction using COLMAP + PyCOLMAP (CUDA)
	•	🧍 Parametric human fitting with SMPL-X via multi-stage optimization
	•	🎨 Robust color transfer from reconstructed scans to SMPL-X
	•	🧹 Point cloud & mesh refinement (SOR, noise filtering, Poisson reconstruction)
	•	🏃 Motion retargeting using MOYO motion sequences
	•	🎬 Final animated video output with preserved identity and appearance

⸻

🧠 Pipeline Overview

RGB Images
   ↓
Foreground Masking (YOLO + SAM)
   ↓
Multi-view Reconstruction (COLMAP / PyCOLMAP)
   ↓
Point Cloud Refinement (SOR + Noise Filtering)
   ↓
Poisson Surface Reconstruction
   ↓
Mesh Refinement & Triangle Decimation
   ↓
SMPL-X Fitting (Multi-stage Optimization)
   ↓
Color Transfer (Nearest Neighbor Projection)
   ↓
Motion Retargeting (MOYO)
   ↓
Final Animated Video


⸻

📁 Repository Structure

.
├── reconstruction/
│   ├── colmap_pipeline.py
│   └── point_cloud_refinement.py
│
├── mesh_processing/
│   ├── poisson_reconstruction.py
│   ├── mesh_refinement.py
│   └── mesh_decimation.py
│
├── smplx_fitting/
│   ├── fit_smplx.py
│   ├── loss_plot.py
│   └── debug_visualization.py
│
├── appearance_transfer/
│   └── color_transfer.py
│
├── motion_retargeting/
│   └── retarget_moyo_motion.py
│
├── assets/
│   ├── images/
│   ├── videos/
│   └── figures/
│
├── README.md
└── requirements.txt


⸻

🔧 Installation

1. Environment Setup

conda create -n human_motion python=3.9
conda activate human_motion

2. Install Dependencies

pip install -r requirements.txt

Key dependencies include:
	•	torch
	•	smplx
	•	trimesh
	•	pycolmap
	•	open3d
	•	scikit-learn

Note: PyCOLMAP with CUDA support is required for dense multi-view stereo.

⸻

📸 Foreground Masking

Foreground human segmentation is achieved via a two-stage learning-based pipeline:
	•	YOLO for coarse human detection (bounding boxes)
	•	SAM (Segment Anything Model) for precise pixel-level masks

This step removes background clutter and significantly improves reconstruction quality.

⸻

☁️ 3D Reconstruction
	•	Multi-view images are reconstructed using COLMAP / PyCOLMAP (CUDA)
	•	Outputs include camera poses, sparse points, and dense point clouds
	•	Typical scale:
	•	Initial points: ~8.3M
	•	Refined points: ~1.2M

⸻

🧹 Point Cloud & Mesh Processing

Point Cloud Refinement
	•	Statistical Outlier Removal (SOR)
	•	Noise filtering (CloudCompare-style)

Mesh Construction
	•	Poisson Surface Reconstruction (depth ≈ 9)
	•	Produces watertight, manifold meshes

Mesh Optimization
	•	Laplacian smoothing
	•	Removal of disconnected components
	•	Triangle decimation (Quadric Edge Collapse)
	•	~840K → ~200K faces

⸻

🧍 SMPL-X Fitting

We fit an SMPL-X model to the refined mesh using multi-stage gradient-based optimization.

Optimization Stages

Stage	Parameters	Learning Rate	Iterations
1	Global orient, translation, scale	0.01	150
2	Shape (β)	0.01	250
3	Pose (high reg)	0.005	300
4	Pose refinement	0.002	400
5	All parameters	0.001	200

	•	Loss: Bidirectional Chamfer distance
	•	Regularization: Shape, pose, and hand priors

⸻

🎨 Appearance Transfer

Color is transferred from the reconstructed scan to the SMPL-X mesh via nearest-neighbor projection:
	•	Scan and SMPL-X meshes are independently centered
	•	Vertex colors are normalized to [0, 1]
	•	k-NN projection (k = 1) in 3D space

This avoids UV mapping and remains robust to alignment offsets.

⸻

🏃 Motion Retargeting (MOYO)

Motion is retargeted using sequences from the MOYO dataset:
	•	Pose representation: 165D fullpose
	•	Body, hand, and jaw poses extracted explicitly
	•	Shape (β) and scale fixed across all frames
	•	Floor alignment via minimum vertex height
	•	Frame-wise export as OBJ

⸻

🎬 Final Output
	•	Colored SMPL-X avatar animated with retargeted motion
	•	Rendered as a video (MP4) from exported frame sequence

<p align="center">
  <img src="assets/videos/final_animation.gif" width="600" />
</p>



⸻

⚠️ Limitations
	•	Fine hand and finger geometry may be incomplete due to reconstruction sparsity
	•	Facial expressions are limited by scan quality
	•	No temporal smoothing across motion frames

⸻

🚀 Future Work
	•	Temporal consistency constraints during fitting
	•	Improved hand reconstruction
	•	Texture atlas generation
	•	Real-time rendering via Vertex Animation Textures (VAT)

⸻

📚 Acknowledgements
	•	SMPL-X: AMASS / MPI
	•	COLMAP / PyCOLMAP
	•	MOYO Dataset
	•	Segment Anything (SAM)

⸻

👤 Author

Vishal Purohit
PhD Student — 3D Vision & Generative Models

⸻

📬 Contact

For questions or collaborations, feel free to reach out.
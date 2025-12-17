# Human Motion Retargeting from RGB Images

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Stable-EE4C2C?logo=pytorch&logoColor=white)
![SMPL-X](https://img.shields.io/badge/Model-SMPL--X-green)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?logo=nvidia&logoColor=white)

This repository presents an end-to-end pipeline for human motion retargeting, starting from raw RGB images and producing a fully animated, identity-preserving 3D human avatar. The system integrates multi-view 3D reconstruction, parametric human modeling with SMPL-X, appearance transfer, and motion retargeting using real motion data.

**Context:** This project was developed as part of **STAT 59800: 3D Computer Vision & Virtual Human Models**.

<p align="center">
  <img src="assets/videos/final_animation.gif" alt="Final Animation" width="600" />
</p>

---

## ✨ Highlights

* 📸 **Image-based 3D reconstruction** using COLMAP + PyCOLMAP (CUDA-accelerated).
* 🧍 **Parametric human fitting** with SMPL-X via robust multi-stage optimization.
* 🎨 **Appearance transfer** from reconstructed scans to SMPL-X using nearest-neighbor projection.
* 🧹 **Mesh refinement** including SOR, noise filtering, and Poisson reconstruction.
* 🏃 **Motion retargeting** utilizing MOYO motion sequences.
* 🎬 **Identity preservation** in final animated video outputs.

---

## 🧠 Pipeline Overview

```mermaid
graph TD;
    A[RGB Images] --> B[Foreground Masking<br/>YOLO + SAM];
    B --> C[Multi-view Reconstruction<br/>COLMAP / PyCOLMAP];
    C --> D[Point Cloud Refinement<br/>SOR + Noise Filtering];
    D --> E[Poisson Surface Reconstruction];
    E --> F[Mesh Refinement &<br/>Triangle Decimation];
    F --> G[SMPL-X Fitting<br/>Multi-stage Optimization];
    G --> H[Color Transfer<br/>NN Projection];
    H --> I[Motion Retargeting<br/>MOYO];
    I --> J[Final Animated Video];
````
-----

## 📁 Repository Structure

```text
```text
3d_virtual_humans/
├── 00_extract_frames.py         # Extract frames from input video
├── 01_segment_save.py           # Generate foreground masks (YOLO + SAM)
├── 02_refine_masks.py           # Refine and clean binary masks
├── 03_3d_proj.py                # Run COLMAP/PyCOLMAP reconstruction
├── 04_create_posson_mesh.py     # Poisson surface reconstruction
├── 05_mesh_clean.py             # Mesh refinement, smoothing, decimation
├── 06_mesh_to_simplx.py         # Fit SMPL-X model to the clean mesh
├── 07_extract_colors_robust.py  # Transfer colors from scan to SMPL-X
├── 08_motion_retarget.py        # Apply MOYO motion sequences
├── 09_export_meshes.py          # Export final animated meshes/video
├── download_moyo_data.py        # Helper to download specific sequences
├── download_moyo.sh             # Shell script for MOYO dataset
└── README.md
```

-----

## 🔧 Installation

### 1\. Dependencies

  * `torch`
  * `smplx`
  * `trimesh`
  * `pycolmap` (Requires CUDA for dense reconstruction)
  * `open3d`
  * `scikit-learn`

-----

## 📸 Foreground Masking

Foreground human segmentation is achieved via a two-stage learning-based pipeline to remove background clutter and improve reconstruction quality:

1.  **YOLO:** Used for coarse human detection (bounding boxes).
2.  **SAM (Segment Anything Model):** Used for precise pixel-level masks within the bounding box.

## ☁️ 3D Reconstruction

  * **Engine:** COLMAP / PyCOLMAP (CUDA).
  * **Outputs:** Camera poses, sparse point clouds, and dense point clouds.
  * **Scale:**
      * Initial points: \~8.3M
      * Refined points: \~1.2M

## 🧹 Point Cloud & Mesh Processing

**Point Cloud Refinement:**

  * Statistical Outlier Removal (SOR).
  * Noise filtering (CloudCompare-style).

**Mesh Construction:**

  * **Method:** Poisson Surface Reconstruction (depth ≈ 9).
  * **Result:** Watertight, manifold meshes.

**Mesh Optimization:**

  * Laplacian smoothing.
  * Removal of disconnected components.
  * Triangle decimation (Quadric Edge Collapse) reducing \~840K to \~200K faces.

-----

## 🧍 SMPL-X Fitting

We fit an SMPL-X model to the refined mesh using multi-stage gradient-based optimization.

### Optimization Schedule

| Stage | Parameters Target | Learning Rate | Iterations |
| :---: | :--- | :---: | :---: |
| **1** | Global orient, translation, scale | 0.01 | 150 |
| **2** | Shape ($\beta$) | 0.01 | 250 |
| **3** | Pose (High Regularization) | 0.005 | 300 |
| **4** | Pose Refinement | 0.002 | 400 |
| **5** | All Parameters | 0.001 | 200 |

  * **Loss Function:** Bidirectional Chamfer distance.
  * **Regularization:** Priors applied to Shape, Pose, and Hands.

-----

## 🎨 Appearance Transfer & Motion Retargeting

### Appearance Transfer

Color is transferred from the reconstructed scan to the SMPL-X mesh via **Nearest-Neighbor Projection**:

1.  Scan and SMPL-X meshes are independently centered.
2.  Vertex colors are normalized to `[0, 1]`.
3.  $k$-NN projection ($k=1$) is performed in 3D space.

<!-- end list -->

  * *Advantage:* Avoids complex UV mapping and remains robust to slight alignment offsets.

### Motion Retargeting (MOYO)

Motion is retargeted using sequences from the **MOYO dataset**:

  * **Pose Representation:** 165D fullpose.
  * **Explicit Extraction:** Body, hand, and jaw poses are extracted explicitly.
  * **Constraints:** Shape ($\beta$) and scale are fixed across all frames; floor alignment is enforced via minimum vertex height.

-----

## ⚠️ Limitations

  * **Hand Geometry:** Fine hand and finger geometry may be incomplete due to reconstruction sparsity.
  * **Facial Expressions:** Limited by the quality of the initial scan.

-----

## 📚 Acknowledgements

  * [SMPL-X / AMASS / MPI](https://smpl-x.is.tue.mpg.de/)
  * [COLMAP](https://colmap.github.io/) / [PyCOLMAP](https://github.com/colmap/pycolmap)
  * [MOYO Dataset](https://moyo.is.tue.mpg.de/)
  * [Segment Anything (SAM)](https://segment-anything.com/)



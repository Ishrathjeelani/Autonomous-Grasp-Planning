# 🤖 Vision-to-Grasp Robotic Pipeline

This repository implements a **Autonomous robotic grasping pipeline** that starts from **Perception** and ends with **executing a grasp in simulation**.  
It combines **modern generative AI for 3D vision** with **motion and grasp planning** in MuJoCo.

---

## 🚀 Pipeline Overview

1. **Simulated Image Capture**
   - Robots: Franka Emika (MuJoCo).
   - Mounted eye-in-hand camera captures RGB-D images of unknown objects on a table.

2. **Object Pose Estimation & Segmentation**
   - Segment object using **SAM**.
   - Estimate coarse 6-DoF pose with **FoundationPose**.

3. **3D Object Reconstruction**
   - Generate meshes from images:
     - **Shap-E** (image → mesh).
     - **Instant-NGP / NeRFStudio** for NeRF reconstruction.
     - Diffusion-based Image-to-3D models for category-level reconstruction.

4. **Mesh & Point Cloud Processing**
   - Extract **watertight meshes** or **point clouds**.
   - Compute **oriented bounding boxes** with **Trimesh**.
   - Normalize scale + align meshes for simulation.

5. **Grasp Pose Generation**
   - Grasp candidates from:
     - **GraspNet** API (point cloud based).

6. **Motion Planning & Execution**
   - Plan collision-free trajectory using **OMPL (RRTConnect)**.
   - Solve 6D IK with **null-space projection** for redundancy.
   - Execute trajectory on **MuJoCo** robots (Franka).
   - Simulated gripper executes the grasp.

---

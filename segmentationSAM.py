import mujoco
from mujoco import Renderer
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt

# === Load MuJoCo model and data ===
model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

# === Setup renderer ===
renderer = Renderer(model, height=480, width=640)
camera_name = "ee_camera"  # Change if needed

def colorize_and_label_masks(masks, labels, original_image):
    # HSV color ranges
    blue_lower = np.array([100, 100, 50])
    blue_upper = np.array([130, 255, 255])
    yellow_lower = np.array([20, 100, 50])
    yellow_upper = np.array([35, 255, 255])

    output = np.zeros_like(original_image)

    for i, mask in enumerate(masks):
        mask_uint8 = (mask.astype(np.uint8) * 255)  # (H, W) with 0 or 255

        # Extract masked image
        masked_img = cv2.bitwise_and(original_image, original_image, mask=mask_uint8)
        hsv_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)

        # Check for blue/yellow
        blue_mask = cv2.inRange(hsv_img, blue_lower, blue_upper)
        yellow_mask = cv2.inRange(hsv_img, yellow_lower, yellow_upper)

        if cv2.countNonZero(blue_mask) > 50:
            color = (255, 0, 0)  # Blue in BGR
        elif cv2.countNonZero(yellow_mask) > 50:
            color = (0, 255, 255)  # Yellow in BGR
        else:
            continue

        # Apply color only where mask is True
        output[mask] = color

        # Draw label
        ys, xs = np.where(mask)
        if len(xs) > 0:
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            cv2.putText(output, labels[i], (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return output

def show_sam_masks(image, masks, scores):
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, len(masks)+1, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")
    
    # Each mask overlay
    for i, mask in enumerate(masks):
        plt.subplot(1, len(masks)+1, i+2)
        
        # Overlay the mask in semi-transparent red
        overlay = image.copy()
        overlay[mask] = (255, 0, 0)  # Red for mask
        plt.imshow(overlay)
        plt.title(f"Mask {i} (score={scores[i]:.2f})")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()

# === Step simulation and render image ===
kf_id = 0
mujoco.mj_resetDataKeyframe(model, data, kf_id)
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera=camera_name)
rgb_image = renderer.render()
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# === Load SAM ===
checkpoint_path = "sam_vit_b_01ec64.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path).to(device)

# === Predict segmentation ===
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=10,        # higher → more dense sampling
    pred_iou_thresh=0.88,      # quality threshold
    stability_score_thresh=0.95,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500   # ignore tiny masks
)

masks = mask_generator.generate(rgb_image)

# ---- Visualize Masks ----
overlay = rgb_image.copy()

for i, mask in enumerate(masks):
    mask_img = mask["segmentation"]  # boolean mask
    
    # Convert to a visible RGB mask
    colored_mask = np.zeros((mask_img.shape[0], mask_img.shape[1], 3), dtype=np.uint8)
    colored_mask[mask_img] = [255, 0, 0]  # red for this example
    
    plt.figure(figsize=(5,5))
    plt.imshow(colored_mask)
    plt.axis('off')
    plt.title(f"Mask {i+1}")
    plt.show()

for mask in masks:
    segmentation = mask["segmentation"]  # Boolean mask
    color = np.random.randint(0, 255, (1, 1, 3), dtype=np.uint8)

    # Create a full-size color mask
    colored_mask = np.zeros_like(rgb_image, dtype=np.uint8)
    colored_mask[segmentation] = color

    # Blend the colored mask into the overlay
    overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

cv2.imshow("SAM Auto Masks", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

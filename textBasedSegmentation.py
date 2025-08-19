import mujoco
import mujoco.viewer
import numpy as np
import torch
import cv2
from PIL import Image
import time

# HuggingFace + SAM
from transformers import CLIPProcessor, CLIPModel
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ========================
# 1. CONFIGURATION
# ========================
MODEL_XML_PATH = "franka_emika_panda/scene.xml"
CLIP_MODEL_PATH = "./clip-vit-base-patch32"
SAM_CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
CAMERA_NAME = "ee_camera"
IMG_WIDTH, IMG_HEIGHT = 640, 480
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========================
# 2. INITIALIZATION
# ========================
def init_mujoco():
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print("[INFO] Robot at home position.")
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    return model, data, renderer


def init_clip():
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
    print("[INFO] CLIP model loaded.")
    return clip_model, clip_processor


def init_sam():
    sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
    mask_gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=10,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=500
    )
    print("[INFO] SAM model loaded.")
    return mask_gen


# ========================
# 3. CAPTURE IMAGE
# ========================
def capture_image(renderer, data):
    renderer.update_scene(data, camera=CAMERA_NAME)
    img = renderer.render()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAMERA_NAME)
    depth = renderer.render()
    # Shift nearest values to the origin.
    depth -= depth.min()
    # Scale by 2 mean distances of near rays.
    depth /= 2*depth[depth <= 1].mean()
    pixels = 255*np.clip(depth, 0, 1)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), pixels.astype(np.uint8)  # return BGR for OpenCV


# ========================
# 4. CLIP MASK FILTERING
# ========================
def filter_masks_with_clip(image, masks, text_prompt, clip_model, clip_processor, threshold=25):
    filtered_masks = []
    scores = []

    for idx, m in enumerate(masks):
        seg_mask = m["segmentation"]
        masked_img = image.copy()
        masked_img[~seg_mask] = 0

        pil_img = Image.fromarray(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
        inputs = clip_processor(text=[text_prompt], images=pil_img,
                                return_tensors="pt", padding=True).to(DEVICE)

        outputs = clip_model(**inputs)
        score = outputs.logits_per_image[0][0].item()

        if score >= threshold:
            filtered_masks.append(seg_mask)
            scores.append(score)

    return filtered_masks, scores


# ========================
# 5. MAIN PIPELINE
# ========================
def main():
    model, data, renderer = init_mujoco()
    clip_model, clip_processor = init_clip()
    mask_generator = init_sam()

    with mujoco.viewer.launch_passive(model, data):
        img_bgr, img_depth = capture_image(renderer, data)
        cv2.imwrite("depth_raw.png", img_depth)

        # SAM segmentation
        masks = mask_generator.generate(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        print(f"[INFO] SAM generated {len(masks)} masks.")

        # User text prompt
        text_prompt = input("Enter object description (e.g., 'yellow capsule'): ")

        # CLIP filtering
        filtered_masks, scores = filter_masks_with_clip(
            img_bgr, masks, text_prompt, clip_model, clip_processor, threshold=25
        )

        if filtered_masks:
            print(f"[INFO] Found {len(filtered_masks)} mask(s) for '{text_prompt}'.")
            for i, mask in enumerate(filtered_masks):
                output_img = img_bgr.copy()
                color_mask = np.zeros_like(output_img, dtype=np.uint8)
                color_mask[mask] = (0, 255, 0)  # Green overlay
                blended = cv2.addWeighted(output_img, 0.5, color_mask, 0.5, 0)
                cv2.putText(blended, f"{text_prompt} ({scores[i]:.2f})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(f"Match {i+1}", blended)
                # === Save Overlay Image ===
                cv2.imwrite(f"filtered_mask_overlay_{i+1}.png", blended)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("[WARN] No matching object found above threshold.")


if __name__ == "__main__":
    main()

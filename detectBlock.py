import mujoco
import mujoco.viewer
import numpy as np
import cv2

# Load MJCF model
model_path = "franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Camera parameters
cam_width, cam_height = 640, 480
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_camera")
fovy = model.cam_fovy[cam_id]
focal_length = (cam_height / 2) / np.tan(np.deg2rad(fovy / 2))
cx, cy = cam_width / 2, cam_height / 2
z_table = 0.47  # Known table height in world frame (adjust as per your scene)

renderer = mujoco.Renderer(model, height=cam_height, width=cam_width)

# Detect center of black object
def detect_yellow_object_center(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

# Compute 3D point from pixel assuming table height
def pixel_to_world_and_base(u, v):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_camera")
    cam_pos = data.cam_xpos[cam_id]
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    cx = cam_width / 2
    cy = cam_height / 2

    x_cam = (u - cx) / focal_length
    y_cam = (v - cy) / focal_length
    ray_cam = np.array([x_cam, y_cam, 1.0])
    ray_world = cam_mat @ ray_cam
    ray_world /= np.linalg.norm(ray_world)

    t = (z_table - cam_pos[2]) / ray_world[2]
    point_world = cam_pos + t * ray_world

    # Transform to base frame
    base_pos = data.body("franka1").xpos
    base_rot = data.body("franka1").xmat.reshape(3, 3)
    point_base = base_rot.T @ (point_world - base_pos)

    return point_world, point_base

# Camera capture
def capture_camera_view():
    renderer.update_scene(data, camera="ee_camera")
    return renderer.render()

# Main loop
if __name__ == "__main__":
    viewer = mujoco.viewer.launch_passive(model, data)
    print("[DEBUG] Viewer launched.")

    try:
        while viewer.is_running():
            mujoco.mj_step(model, data)

            img = capture_camera_view()
            center = detect_yellow_object_center(img)

            if center:
                u, v = center
                print(f"[Image] Black object center: (u={u}, v={v})")
                # Draw circle on the image for visual debugging
                debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
                cv2.circle(debug_img, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
                cv2.imshow("EE Camera View with Detection", debug_img)
                cv2.waitKey(1)

                pos_world, pos_base = pixel_to_world_and_base(u, v)
                print(f"[World Frame] Estimated 3D position: {pos_world}")
                print(f"[Base Frame] Estimated 3D position: {pos_base}")
            else:
                print("[Detection] No black object found.")
                debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
                cv2.imshow("EE Camera View with Detection", debug_img)
                cv2.waitKey(1)

            viewer.sync()
    except KeyboardInterrupt:
        print("Stopped by user.")

    renderer.close()

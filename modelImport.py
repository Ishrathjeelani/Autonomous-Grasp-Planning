import mujoco
import mujoco.viewer
import numpy as np
import cv2

# ---- MJCF Scene ----
model_path = "franka_emika_panda/scene.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# ---- Camera Setup ----
cam_width, cam_height = 640, 480
renderer = mujoco.Renderer(model, height=cam_height, width=cam_width)

def detect_black_object_center(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    return None

def detect_blue_object_center(img):
    # Convert image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Define blue color range in HSV
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Create a mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow("Black Mask", mask)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
    
    return None

def pixel_to_camera_ray(u, v, width, height, fovy_deg):
    fovy = np.deg2rad(fovy_deg)
    fy = height / (2 * np.tan(fovy / 2))
    fx = fy  # assuming square pixels and fx=fy
    cx = width / 2
    cy = height / 2

    x = (u - cx) / fx
    y = (v - cy) / fy
    z = 1.0
    ray = np.array([x, y, z])
    return ray / np.linalg.norm(ray)


def get_object_position_in_base(u, v):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_camera")
    cam_pos = model.cam_pos[cam_id]
    cam_quat = model.cam_quat[cam_id]
    cam_mat = np.zeros(9)
    mujoco.mju_quat2Mat(cam_mat, cam_quat)
    cam_mat = cam_mat.reshape(3, 3)

    fovy_deg = model.cam_fovy[cam_id]
    ray_cam = pixel_to_camera_ray(u, v, cam_width, cam_height, fovy_deg)

    ray_world = cam_mat @ ray_cam

    pnt = cam_pos.astype(np.float64)
    vec = (10 * ray_world).astype(np.float64)
    geomid = np.zeros(1, dtype=np.int32)
    geomgroup = np.ones(6, dtype=np.uint8)
    flg_static = 0
    bodyexclude = -1
    # Perform raycast
    dist = mujoco.mj_ray(model, data, pnt, vec, geomgroup, flg_static, bodyexclude, geomid)

    if geomid[0] == -1:
        return None

    # Compute 3D hit point
    point = pnt + dist * vec / np.linalg.norm(vec)

    base_pos = data.body("franka1").xpos
    base_rot = data.body("franka1").xmat.reshape(3, 3)
    point_base = base_rot.T @ (point - base_pos)
    return point_base

def capture_camera_view():
    renderer.update_scene(data, camera="ee_camera")
    return renderer.render()

# ---- Main ----
if __name__ == "__main__":
    viewer = mujoco.viewer.launch_passive(model, data)
    print("Running simulation...")

    while viewer.is_running():
        mujoco.mj_step(model, data)

        # Capture and process image from end-effector camera
        img = capture_camera_view()
        center = detect_blue_object_center(img)

        if center:
            u, v = center
            print(f"[Image] Black object center: (u={u}, v={v})")
            pos = get_object_position_in_base(u, v)
            if pos is not None:
                print(f"[Base Frame] Estimated 3D position: {pos}")
            else:
                print("[Raycast] Ray hit nothing.")
        else:
            print("[Detection] No black object found.")

        viewer.sync()

import mujoco
import numpy as np
from mujoco import viewer
import ompl.base as ob
import ompl.geometric as og
import time

# === Load model and initialize data ===
model = mujoco.MjModel.from_xml_path("franka_emika_panda/scene.xml")
data = mujoco.MjData(model)

# === Define joint and end-effector names ===
joint_names = [f"joint{i+1}" for i in range(7)]
ee_body_name = "hand"

# === Target pose (position + quaternion) ===
target_position = np.array([-0.2105, -0.2237, 0.9])
target_orientation = np.array([0, 1, 0, 0])  # 90° around Y-axis

# === Joint limits ===
lower_limits, upper_limits = [], []
for name in joint_names:
    j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    lower_limits.append(model.jnt_range[j_id][0])
    upper_limits.append(model.jnt_range[j_id][1])
lower_limits = np.array(lower_limits)
upper_limits = np.array(upper_limits)

# === Start state from keyframe "home" ===
kf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(model, data, kf_id)

start_state = data.qpos[:7].copy()
print("Start qpos:", start_state)

# === OMPL Setup ===
space = ob.RealVectorStateSpace(7)
bounds = ob.RealVectorBounds(7)
for i in range(7):
    bounds.setLow(i, float(lower_limits[i]))
    bounds.setHigh(i, float(upper_limits[i]))
space.setBounds(bounds)
ss = og.SimpleSetup(space)
print("Lower bounds:", lower_limits)
print("Upper bounds:", upper_limits)
# === State validity checker ===
def is_valid(state): return True  # Replace with collision check if needed
ss.setStateValidityChecker(ob.StateValidityCheckerFn(is_valid))
si = ss.getSpaceInformation()

# === Convert to OMPL state ===
def to_ompl_state(vector):
    state = ob.State(space)
    s = state()
    for i in range(7):
        s[i] = float(vector[i])
    return state

start = to_ompl_state(start_state)
if not si.satisfiesBounds(start.get()):
    raise ValueError("[ERROR] Start state is out of bounds.")
print("[INFO] Start state OK.")

# === Helper: Set robot joint positions ===
def set_robot_qpos(q):
    data.qpos[:7] = q
    mujoco.mj_forward(model, data)

# === Pose and orientation error ===
def orientation_error(q_target, q_current):
    q_conj = q_current.copy()
    q_conj[1:] *= -1  # conjugate
    q_rel = np.zeros(4)
    mujoco.mju_mulQuat(q_rel, q_target, q_conj)
    return 2.0 * q_rel[1:]  # Small-angle approx (axis-angle)

def pose_error(pos_des, quat_des, pos_curr, quat_curr):
    e_pos = pos_des - pos_curr
    e_rot = orientation_error(quat_des, quat_curr)
    return np.concatenate([e_pos, e_rot])  # 6D error

# === 6D Inverse Kinematics with Nullspace Optimization ===
def inverse_kinematics(target_pos, target_quat, q_init, rest_pose=None):
    q = q_init.copy()
    if rest_pose is None:
        rest_pose = np.zeros(7)

    for _ in range(300):
        set_robot_qpos(q)
        ee_pos = data.body(ee_body_name).xpos.copy()
        ee_quat = data.body(ee_body_name).xquat.copy()

        err = pose_error(target_pos, target_quat, ee_pos, ee_quat)
        if np.linalg.norm(err) < 1e-3:
            break

        # Jacobians
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        mujoco.mj_jacBody(model, data, Jp, Jr, body_id)
        J_full = np.vstack([Jp[:, :7], Jr[:, :7]])  # 6x7

        # Damped pseudoinverse for stability
        J_pinv = np.linalg.pinv(J_full)

        # Task-space joint update
        dq_task = J_pinv @ err

        # Nullspace projection for secondary objective (joint effort minimization)
        null_proj = np.eye(7) - J_pinv @ J_full
        dq_null = null_proj @ (rest_pose - q[:7])

        # Combine motions
        q[:7] += dq_task + 0.01 * dq_null
        q[:7] = np.clip(q[:7], lower_limits, upper_limits)

    return q

# === Solve IK for goal config ===
goal_config = inverse_kinematics(target_position, target_orientation, start_state)
goal = to_ompl_state(goal_config)
if not si.satisfiesBounds(goal.get()):
    raise ValueError("[ERROR] Goal state is out of bounds.")
print("[INFO] Goal state OK.")

# === Plan motion with OMPL ===
ss.setStartAndGoalStates(start, goal)
planner = og.RRTConnect(si)
planner.setRange(0.1) 
ss.setPlanner(planner)

if ss.solve(2.0):
    print("[INFO] Path found. Visualizing...")
    path = ss.getSolutionPath()
    path.interpolate(100)
    trajectory = [np.array([path.getState(i)[j] for j in range(7)]) for i in range(path.getStateCount())]
    
    # Reset robot to the start of the trajectory before launching viewer
    data.qpos[:7] = start_state
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    with viewer.launch_passive(model, data) as v:
        time.sleep(2)  # let viewer render initial state
        for q in trajectory:
            set_robot_qpos(q)
            mujoco.mj_step(model, data)
            v.sync()
            time.sleep(0.05)
        # Add this delay before closing
        print("[INFO] Trajectory visualization complete. Closing viewer.")
        time.sleep(1)
else:
    print("[ERROR] No path found.")

del model
del data
import gc
gc.collect()

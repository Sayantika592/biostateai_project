"""SMPL motion helpers: convert EDGE output into viewer JSON and BVH.

EDGE saves a pickle per generated dance with:
  smpl_poses: (T, 72) axis-angle local rotations for the 24 SMPL joints
  smpl_trans: (T, 3) root translation
  full_pose:  (T, 24, 3) global joint positions from forward kinematics
Output is 30 fps.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

FPS = 30

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
CHILDREN = {j: [c for c, p in enumerate(PARENTS) if p == j] for j in range(24)}

# Approximate SMPL neutral rest-pose joint locations (y-up, metres). We don't
# ship the SMPL body model, so forward kinematics uses this template.
SMPL_REST_JOINTS = np.array([
    [-0.0018, -0.2233, 0.0282], [0.0695, -0.3141, 0.0239], [-0.0677, -0.3143, 0.0215],
    [-0.0025, -0.1035, 0.0119], [0.1040, -0.6990, 0.0300], [-0.1060, -0.7045, 0.0289],
    [0.0049, 0.0352, 0.0010], [0.0886, -1.1074, -0.0097], [-0.0920, -1.1173, -0.0097],
    [0.0030, 0.0887, 0.0267], [0.1182, -1.1612, 0.1053], [-0.1182, -1.1612, 0.1077],
    [-0.0013, 0.2995, -0.0145], [0.0742, 0.2063, -0.0074], [-0.0757, 0.2049, -0.0108],
    [0.0047, 0.3646, 0.0418], [0.1975, 0.2301, -0.0181], [-0.1914, 0.2324, -0.0194],
    [0.4499, 0.2216, -0.0397], [-0.4475, 0.2213, -0.0438], [0.7155, 0.2367, -0.0436],
    [-0.7121, 0.2348, -0.0472], [0.7976, 0.2272, -0.0532], [-0.7971, 0.2294, -0.0576],
], dtype=np.float64)
REST_OFFSETS = SMPL_REST_JOINTS - SMPL_REST_JOINTS[[max(p, 0) for p in PARENTS]]
REST_OFFSETS[0] = 0.0


def forward_kinematics(local_rotvec: np.ndarray, root_pos: np.ndarray, offsets: np.ndarray):
    """local_rotvec (T,24,3), root_pos (T,3), offsets (24,3) -> positions (T,24,3), global mats (T,24,3,3)."""
    T = local_rotvec.shape[0]
    local = R.from_rotvec(local_rotvec.reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3)
    glob = np.zeros_like(local)
    pos = np.zeros((T, 24, 3))
    for j in range(24):
        p = PARENTS[j]
        if p < 0:
            glob[:, j] = local[:, j]
            pos[:, j] = root_pos
        else:
            glob[:, j] = glob[:, p] @ local[:, j]
            pos[:, j] = pos[:, p] + np.einsum("tij,j->ti", glob[:, p], offsets[j])
    return pos, glob


def _up_fix(full_pose: np.ndarray) -> np.ndarray:
    """Rotation that maps the motion's up axis (pelvis->head) onto +Y for three.js / BVH."""
    up = (full_pose[:, 15] - full_pose[:, 0]).mean(axis=0)
    axis = int(np.argmax(np.abs(up)))
    sign = np.sign(up[axis])
    if axis == 1:
        return np.eye(3) if sign > 0 else R.from_euler("x", 180, degrees=True).as_matrix()
    if axis == 2:  # z-up (EDGE convention) -> y-up
        return R.from_euler("x", -90 * sign, degrees=True).as_matrix()
    return R.from_euler("z", 90 * sign, degrees=True).as_matrix()


def normalize(sample: dict) -> dict:
    """Return y-up copies of rotations and positions, with the dancer starting at the origin on the floor."""
    rotvec = np.asarray(sample["smpl_poses"], dtype=np.float64).reshape(-1, 24, 3)
    full = np.asarray(sample["full_pose"], dtype=np.float64).reshape(-1, 24, 3)
    C = _up_fix(full)
    full = full @ C.T
    root = R.from_matrix(C) * R.from_rotvec(rotvec[:, 0])
    rotvec = rotvec.copy()
    rotvec[:, 0] = root.as_rotvec()
    shift = np.array([full[0, 0, 0], full[:, :, 1].min(), full[0, 0, 2]])
    full = full - shift
    return {"rotvec": rotvec, "positions": full}


def to_viewer_json(sample: dict) -> dict:
    n = normalize(sample)
    return {
        "fps": FPS,
        "names": JOINT_NAMES,
        "parents": PARENTS,
        "frames": np.round(n["positions"], 3).tolist(),
    }


def _offsets_from_data(rotvec: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Recover bone offsets in each parent's local frame, averaged over frames."""
    T = rotvec.shape[0]
    local = R.from_rotvec(rotvec.reshape(-1, 3)).as_matrix().reshape(T, 24, 3, 3)
    glob = np.zeros_like(local)
    offsets = np.zeros((24, 3))
    for j in range(24):
        p = PARENTS[j]
        if p < 0:
            glob[:, j] = local[:, j]
            continue
        glob[:, j] = glob[:, p] @ local[:, j]
        d = positions[:, j] - positions[:, p]
        offsets[j] = np.einsum("tji,tj->ti", glob[:, p], d).mean(axis=0)
    return offsets


def to_bvh(sample: dict) -> str:
    n = normalize(sample)
    rotvec, pos = n["rotvec"], n["positions"]
    offsets = _offsets_from_data(rotvec, pos)
    lines: list[str] = ["HIERARCHY"]
    order: list[int] = []

    def emit(j: int, depth: int):
        ind = "  " * depth
        o = offsets[j] * 100  # centimetres, the usual BVH unit
        if PARENTS[j] < 0:
            lines.append(f"{ind}ROOT {JOINT_NAMES[j]}")
            lines.append(f"{ind}{{")
            lines.append(f"{ind}  OFFSET 0.0 0.0 0.0")
            lines.append(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{ind}JOINT {JOINT_NAMES[j]}")
            lines.append(f"{ind}{{")
            lines.append(f"{ind}  OFFSET {o[0]:.4f} {o[1]:.4f} {o[2]:.4f}")
            lines.append(f"{ind}  CHANNELS 3 Zrotation Xrotation Yrotation")
        order.append(j)
        if CHILDREN[j]:
            for c in CHILDREN[j]:
                emit(c, depth + 1)
        else:
            lines.append(f"{ind}  End Site")
            lines.append(f"{ind}  {{")
            e = o / max(np.linalg.norm(o), 1e-6) * 5
            lines.append(f"{ind}    OFFSET {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}")
            lines.append(f"{ind}  }}")
        lines.append(f"{ind}}}")

    emit(0, 0)
    T = rotvec.shape[0]
    eul = R.from_rotvec(rotvec.reshape(-1, 3)).as_euler("ZXY", degrees=True).reshape(T, 24, 3)
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / FPS:.6f}"]
    for t in range(T):
        vals = list(pos[t, 0] * 100)
        for j in order:
            vals += list(eul[t, j])
        lines.append(" ".join(f"{v:.4f}" for v in vals))
    return "\n".join(lines) + "\n"


def mock_sample(seconds: float, bpm: float = 120.0, seed: int = 0) -> dict:
    """Procedural beat-synced motion so the app can be tried without a GPU."""
    rng = np.random.default_rng(seed)
    T = int(seconds * FPS)
    t = np.arange(T) / FPS
    beat = 2 * np.pi * bpm / 60.0
    ph = rng.uniform(0, 2 * np.pi, 8)
    rot = np.zeros((T, 24, 3))
    bounce = np.sin(beat * t) ** 2
    rot[:, 0, 1] = 0.4 * np.sin(beat * t / 4 + ph[0])            # pelvis twist
    rot[:, 1, 0] = -0.5 * bounce; rot[:, 2, 0] = -0.5 * bounce   # hips bend
    rot[:, 4, 0] = 1.0 * bounce; rot[:, 5, 0] = 1.0 * bounce     # knees
    rot[:, 7, 0] = -0.5 * bounce; rot[:, 8, 0] = -0.5 * bounce   # ankles
    rot[:, 3, 2] = 0.15 * np.sin(beat * t / 2 + ph[1])
    rot[:, 16, 2] = -0.9 + 0.8 * np.sin(beat * t / 2 + ph[2])    # arms
    rot[:, 17, 2] = 0.9 - 0.8 * np.sin(beat * t / 2 + ph[3])
    rot[:, 18, 1] = -1.0 * (0.5 + 0.5 * np.sin(beat * t + ph[4]))
    rot[:, 19, 1] = 1.0 * (0.5 + 0.5 * np.sin(beat * t + ph[5]))
    rot[:, 15, 1] = 0.3 * np.sin(beat * t / 2 + ph[6])
    hip_h = 0.92
    root = np.stack([0.2 * np.sin(beat * t / 8 + ph[7]), hip_h - 0.08 * bounce, np.zeros(T)], axis=1)
    pos, _ = forward_kinematics(rot, root, REST_OFFSETS)
    return {"smpl_poses": rot.reshape(T, 72), "smpl_trans": root, "full_pose": pos}

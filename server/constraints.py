"""Body constraints from words: "keep the left hand still", "arms crossed in the intro",
"no big leg kicks", "right arm raised at the drop".

Parsing is rule-based, so it always works, also when the LLM is offline or out of GPU
quota. Each constraint is {limb, mode, sections, label}:
  limb      left_arm, right_arm, arms, left_leg, right_leg, legs, head, torso
  mode      freeze (hold one pose), damp (keep movement small), raise (arm overhead),
            cross (arms crossed), down (arms by the sides)
  sections  plan section numbers it applies to, or [] for the whole dance

They are applied to the SMPL local joint rotations after the moves are stitched, with
smooth ramps in and out, so the rest of the body keeps dancing and the joins stay clean.
"""
from __future__ import annotations

import re

import numpy as np
from scipy.spatial.transform import Rotation as R

CHAINS = {
    "left_arm": [13, 16, 18, 20, 22], "right_arm": [14, 17, 19, 21, 23],
    "left_leg": [1, 4, 7, 10], "right_leg": [2, 5, 8, 11],
    "head": [12, 15], "torso": [3, 6, 9],
}
GROUPS = {"arms": ["left_arm", "right_arm"], "legs": ["left_leg", "right_leg"]}
NICE = {"left_arm": "left arm", "right_arm": "right arm", "arms": "both arms", "left_leg": "left leg",
        "right_leg": "right leg", "legs": "both legs", "head": "head", "torso": "torso"}
MODE_TEXT = {"freeze": "held still", "damp": "kept small", "raise": "raised", "cross": "crossed", "down": "down by the sides"}

# Target local rotations (axis-angle) for shoulder and elbow, solved by IK on the skeleton.
POSES = {
    "raise": {16: [0, 0, 1.2], 18: [0, 0, 0], 17: [0, 0, -1.2], 19: [0, 0, 0]},
    "down": {16: [0, 0, -1.25], 18: [0, 0, 0], 17: [0, 0, 1.25], 19: [0, 0, 0]},
    "cross": {16: [-0.046, -1.19, -1.078], 18: [-0.141, -1.254, -0.986],
              17: [-0.155, 1.131, 1.095], 19: [0.129, 1.224, 0.816]},
}

LIMB_RX = [
    ("left_arm", r"left (arm|hand|wrist|elbow|shoulder)"), ("right_arm", r"right (arm|hand|wrist|elbow|shoulder)"),
    ("left_leg", r"left (leg|foot|knee)"), ("right_leg", r"right (leg|foot|knee)"),
    ("arms", r"\b(arms?|hands|both arms|both hands|upper body)\b"), ("legs", r"\b(legs|feet|knees|lower body|kicks?)\b"),
    ("head", r"\b(head|neck)\b"), ("torso", r"\b(torso|chest|spine|body still)\b"),
]
SECTION_RX = {"last": r"\b(end|ending|last|outro|finale|final)\b", "first": r"\b(start|beginning|intro|opening|first)\b",
              "loud": r"\b(chorus|drop|peak|climax|loud part|hook)\b", "quiet": r"\b(quiet part|verse|soft part|calm part)\b"}


def _mode(clause: str, limb: str) -> str | None:
    c = clause
    if re.search(r"\bcross(ed)?\b|folded", c) and limb in ("arms", "left_arm", "right_arm"):
        return "cross"
    if re.search(r"\b(raised?|up high|in the air|overhead|above (the )?head|up)\b", c) and "arm" in limb or \
            re.search(r"\b(raised?|in the air|overhead)\b", c) and limb == "arms":
        return "raise"
    if re.search(r"\b(down|by (the|my|his|her) sides?|lowered)\b", c) and "arm" in limb:
        return "down"
    if re.search(r"\b(no|without|less|small|smaller|subtle|minimal|gentle|little|limit|reduce|soft)\b", c):
        return "damp"
    if re.search(r"\b(fixed|still|frozen|freeze|locked?|hold|holding|static|stay|stays|don'?t move|not move|steady|in place|in one place)\b", c):
        return "freeze"
    return None


def parse(text: str, sections: list[dict] | None = None) -> list[dict]:
    """Find body constraints in a request. Returns [] when there are none."""
    sections = sections or []
    out = []
    for clause in re.split(r"[.;,!?]| and (?=(?:keep|no|with|make|left|right|both|arms?|legs?|head|hands?|the)\b)| but ", text.lower()):
        for limb, rx in LIMB_RX:
            if not re.search(rx, clause):
                continue
            mode = _mode(clause, limb)
            if not mode:
                continue
            if limb in ("left_arm", "right_arm") and re.search(r"\bboth\b", clause):
                limb = "arms"
            secs = set()
            for key, srx in SECTION_RX.items():
                if re.search(srx, clause) and sections:
                    if key == "last":
                        secs.add(sections[-1]["section"])
                    elif key == "first":
                        secs.add(sections[0]["section"])
                    else:
                        secs |= {s["section"] for s in sections if s["music"] == key}
            where = "" if not secs else " in " + ", ".join(f"S{s}" for s in sorted(secs))
            out.append({"limb": limb, "mode": mode, "sections": sorted(secs),
                        "label": f"{NICE[limb].capitalize()} {MODE_TEXT[mode]}{where}"})
            break  # one limb per clause: "left hand" shouldn't also match "hands"
    # the last word on a limb wins
    seen, uniq = set(), []
    for c in reversed(out):
        if c["limb"] not in seen:
            uniq.append(c)
            seen.add(c["limb"])
    return list(reversed(uniq))


def clears(text: str) -> bool:
    return bool(re.search(r"\b(remove|clear|drop|no more|forget|reset) (the |all )?(constraints?|limits?|rules?)\b|\bmove freely\b|\bfree (the|my|his|her)? ?(arms?|legs?|body)\b", text.lower()))


def merge(old: list[dict], new: list[dict]) -> list[dict]:
    """Later rules are applied last, so they win where they overlap."""
    out = [c for c in old if c["limb"] not in {n["limb"] for n in new}]
    return out + list(new)


def _limbs(limb: str) -> list[str]:
    return GROUPS.get(limb, [limb])


def apply(rot: np.ndarray, constraints: list[dict], frame_ranges: dict[int, tuple[int, int]], fps: int = 30) -> np.ndarray:
    """rot (T,24,3) local axis-angle. frame_ranges: section -> (f0, f1). Returns a new array."""
    if not constraints:
        return rot
    T = rot.shape[0]
    out = rot.copy()
    ramp = int(0.35 * fps)
    for c in constraints:
        w = np.zeros(T)
        spans = [frame_ranges[s] for s in c["sections"] if s in frame_ranges] if c["sections"] else [(0, T)]
        for f0, f1 in spans:
            f0, f1 = max(0, f0), min(T, f1)
            if f1 <= f0:
                continue
            seg = np.ones(f1 - f0)
            r = min(ramp, (f1 - f0) // 3)
            if r > 0:
                if f0 > 0:
                    seg[:r] = np.linspace(0, 1, r)
                if f1 < T:
                    seg[-r:] = np.linspace(1, 0, r)
            w[f0:f1] = np.maximum(w[f0:f1], seg)
        if not w.any():
            continue
        for limb in _limbs(c["limb"]):
            for j in CHAINS[limb]:
                q = R.from_rotvec(out[:, j])
                if c["mode"] in POSES and j in POSES[c["mode"]]:
                    target = R.from_rotvec(np.tile(POSES[c["mode"]][j], (T, 1)))
                    strength = 1.0
                elif c["mode"] in POSES:  # wrists/hands of a posed arm: calm them down
                    target, strength = R.from_rotvec(np.tile(out[:, j].mean(0), (T, 1))), 0.8
                else:
                    active = w > 0.5
                    ref = out[active, j].mean(0) if active.any() else out[:, j].mean(0)
                    target = R.from_rotvec(np.tile(ref, (T, 1)))
                    strength = 0.95 if c["mode"] == "freeze" else 0.65  # damp keeps 35% of the motion
                # rotate part of the way from the current pose to the target, frame by frame
                rel = (target * q.inv()).as_rotvec() * (w * strength)[:, None]
                out[:, j] = (R.from_rotvec(rel) * q).as_rotvec()
    return out

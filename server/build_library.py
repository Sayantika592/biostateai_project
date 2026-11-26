"""Build the dance-move library from AIST++ SMPL motions.

Usage:
  curl -L -o motions.zip https://github.com/google/aistplusplus_dataset/releases/download/v1.0/motions.zip
  unzip motions.zip
  python build_library.py motions/ library.npz

Each AIST++ sequence was danced to a known track whose tempo is fixed by its
music id (e.g. mBR3 -> 110 BPM). We keep 30 fps local joint rotations and
root motion, and find the beat phase from the dancer's own movement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

import aist_io
import motion

# AIST Dance DB track tempos: index 0-5 -> BPM. House runs faster.
BPM = {g: [80, 90, 100, 110, 120, 130] for g in ["BR", "PO", "LO", "MH", "LH", "WA", "KR", "JS", "JB"]}
BPM["HO"] = [110, 115, 120, 125, 130, 135]
GENRES = {"BR": "Break", "PO": "Pop", "LO": "Lock", "MH": "Middle Hip-hop", "LH": "LA Hip-hop",
          "HO": "House", "WA": "Waack", "KR": "Krump", "JS": "Street Jazz", "JB": "Ballet Jazz"}


def kinetic_speed(pos: np.ndarray) -> np.ndarray:
    v = np.linalg.norm(np.diff(pos, axis=0), axis=-1).mean(axis=1) * motion.FPS
    return np.concatenate([v[:1], v])


def beat_phase(speed: np.ndarray, bpm: float) -> float:
    """Frame offset (0..period) that puts the most beats on the dancer's movement 'hits'.

    Hits are local minima of smoothed joint speed (a stop or change of direction),
    the same definition the AIST++ beat-alignment score uses.
    """
    period = motion.FPS * 60.0 / bpm
    v = gaussian_filter1d(speed.astype(np.float64), 2)
    hits = argrelextrema(v, np.less)[0].astype(np.float64)
    if len(hits) == 0:
        return 0.0
    sigma = 0.1 * motion.FPS
    best, best_p = -1.0, 0.0
    for p in np.linspace(0, period, 64, endpoint=False):
        grid = np.arange(p, len(speed) - 1, period)
        d = np.min(np.abs(grid[:, None] - hits[None]), axis=1)
        s = np.exp(-d ** 2 / (2 * sigma ** 2)).mean()
        if s > best:
            best, best_p = s, p
    return float(best_p)


def main(src: str, out: str):
    clips = []
    for f in sorted(Path(src).glob("*.pkl")):
        name = f.stem
        parts = name.split("_")
        genre, music = parts[0][1:], parts[4]
        bpm = BPM[genre][int(music[3])]
        d = aist_io.load(f)
        rot = np.asarray(d["smpl_poses"], np.float64).reshape(-1, 24, 3)[::2]  # 60 -> 30 fps
        trans = np.asarray(d["smpl_trans"], np.float64)[::2] / float(np.asarray(d["smpl_scaling"]).ravel()[0])
        if len(rot) < motion.FPS * 4:
            continue
        pos, _ = motion.forward_kinematics(rot, trans, motion.REST_OFFSETS)
        speed = kinetic_speed(pos)
        clips.append(dict(name=name, genre=genre, bpm=bpm, advanced=parts[1] == "sFM",
                          phase=beat_phase(speed, bpm),
                          rot=rot.astype(np.float32), trans=trans.astype(np.float32), speed=speed.astype(np.float32)))
    bpm = np.array([c["bpm"] for c in clips])
    offsets = np.cumsum([0] + [len(c["rot"]) for c in clips])
    np.savez_compressed(
        out,
        names=np.array([c["name"] for c in clips]), genres=np.array([c["genre"] for c in clips]),
        bpm=bpm.astype(np.float32), phase=np.array([c["phase"] for c in clips], np.float32),
        advanced=np.array([c["advanced"] for c in clips]), offsets=offsets,
        rot=np.concatenate([c["rot"] for c in clips]), trans=np.concatenate([c["trans"] for c in clips]),
        speed=np.concatenate([c["speed"] for c in clips]),
    )
    print("wrote", out, f"{offsets[-1]} frames")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

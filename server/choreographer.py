"""Music-to-dance by beat-matched retrieval and stitching of real AIST++ moves.

No neural-net training and no GPU. Pipeline:
  1. Track the song's beats and per-beat energy with librosa.
  2. Cut the AIST++ library into 8-beat phrases (2 bars), each from a real
     dancer whose track tempo is known.
  3. Pick phrases one after another, scoring each by how smoothly it joins
     the previous phrase (pose distance), how well its movement energy fits
     the music's energy, and its tempo match, while avoiding repeats.
  4. Time-warp every phrase beat-by-beat onto the song's actual beat times,
     then cross-fade the joins (quaternion blend) and keep the feet on the floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.spatial.transform import Rotation as R

import motion

FPS = motion.FPS
PHRASE = 8        # beats per phrase
BLEND = 8         # cross-fade frames (~0.27 s)
GENRES = {"BR": "Break", "PO": "Pop", "LO": "Lock", "MH": "Middle Hip-hop", "LH": "LA Hip-hop",
          "HO": "House", "WA": "Waack", "KR": "Krump", "JS": "Street Jazz", "JB": "Ballet Jazz"}


@dataclass
class Library:
    names: np.ndarray
    genres: np.ndarray
    bpm: np.ndarray
    phase: np.ndarray
    offsets: np.ndarray
    rot: np.ndarray
    trans: np.ndarray
    speed: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "Library":
        d = np.load(path)
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def clip(self, i: int):
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.rot[a:b], self.trans[a:b], self.speed[a:b]


_song_cache: dict = {}


def analyse_song(path: str | Path, seconds: float | None = None):
    p = Path(path)
    key = (str(p), seconds, p.stat().st_mtime if p.exists() else 0)
    if key not in _song_cache:
        if len(_song_cache) > 32:
            _song_cache.clear()
        _song_cache[key] = _analyse_song(path, seconds)
    return _song_cache[key]


def song_outline(path: str | Path, seconds: float | None = None) -> list[dict]:
    """Per 8-beat phrase of the song: time span and music energy (0-1 percentile and a word)."""
    song = analyse_song(path, seconds)
    dur = song["duration"]
    beats = _extend_beats(song["beats"], dur, song["tempo"])
    n_phr = int(np.ceil((len(beats) - 1) / PHRASE))
    e_beat = np.interp(beats, song["beats"], song["energy"]) if len(song["beats"]) else np.zeros(len(beats))
    e_phr = np.array([e_beat[j * PHRASE:(j + 1) * PHRASE].mean() for j in range(n_phr)])
    rel = (e_phr - e_phr.min()) / (np.ptp(e_phr) + 1e-8)
    out = []
    for j in range(n_phr):
        t0 = max(0.0, float(beats[min(j * PHRASE, len(beats) - 1)]))
        t1 = min(dur, float(beats[min((j + 1) * PHRASE, len(beats) - 1)]))
        if t0 >= dur:
            break
        out.append({"slot": j, "t0": round(t0, 2), "t1": round(t1, 2), "level": round(float(rel[j]), 2),
                    "music": "quiet" if rel[j] < 0.34 else "loud" if rel[j] > 0.66 else "medium",
                    "usable": t1 - t0 >= 0.25})
    return out


def _analyse_song(path: str | Path, seconds: float | None = None):
    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=seconds)
    hop = 512
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, units="time")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=np.float64)
    # The tracker sometimes locks onto off-beats (busy hi-hats). The kick drum lives
    # below ~150 Hz, so keep whichever grid, as tracked or shifted half a beat, lands
    # on more low-frequency onsets.
    if len(beats) > 4:
        low = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, fmax=150, n_mels=16)
        lt = librosa.frames_to_time(np.arange(len(low)), sr=sr, hop_length=hop)
        half = float(np.median(np.diff(beats))) / 2
        shifted = beats + half
        shifted = shifted[shifted < len(y) / sr]
        if np.interp(shifted, lt, low).mean() > 1.15 * np.interp(beats, lt, low).mean():
            beats = shifted
    # Bring the beat grid into the 75-140 BPM range the dancers used.
    if tempo > 140:
        beats, tempo = beats[::2], tempo / 2
    elif tempo < 75 and len(beats) > 1:
        mids = (beats[:-1] + beats[1:]) / 2
        beats, tempo = np.sort(np.concatenate([beats, mids])), tempo * 2
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    loud = np.interp(beats, times, rms) if len(beats) else np.zeros(0)
    on = np.interp(beats, librosa.frames_to_time(np.arange(len(onset)), sr=sr, hop_length=hop), onset) if len(beats) else np.zeros(0)
    energy = 0.6 * loud / (loud.max() + 1e-8) + 0.4 * on / (on.max() + 1e-8)
    return dict(tempo=tempo, beats=beats, energy=energy, duration=len(y) / sr)


def _extend_beats(beats: np.ndarray, duration: float, tempo: float) -> np.ndarray:
    """Pad the beat grid before the first and after the last tracked beat so the whole song is covered."""
    period = 60.0 / tempo
    if len(beats) < 2:
        beats = np.arange(0, duration + period, period)
    else:
        period = float(np.median(np.diff(beats)))
    pre = np.arange(beats[0] - period, -PHRASE * period, -period)[::-1]
    post = np.arange(beats[-1] + period, duration + (PHRASE + 2) * period, period)
    return np.concatenate([pre, beats, post])


def _pose_feat(rot_frames: np.ndarray) -> np.ndarray:
    """Pose descriptor: local rotation matrices of the body joints (root excluded)."""
    m = R.from_rotvec(rot_frames[..., 1:22, :].reshape(-1, 3)).as_matrix()
    return m.reshape(*rot_frames.shape[:-2], -1)


class Choreographer:
    def __init__(self, library: Library):
        self.lib = library
        self.phrases = self._index_phrases()

    def _index_phrases(self):
        out = []
        lib = self.lib
        for i in range(len(lib.names)):
            rot, trans, speed = lib.clip(i)
            period = FPS * 60.0 / float(lib.bpm[i])
            n = len(rot)
            k = 0
            while True:
                start = lib.phase[i] + k * period
                end = start + PHRASE * period
                if end + BLEND + 2 >= n:
                    break
                if start - 2 >= 0:
                    s, e = int(round(start)), int(round(end))
                    out.append(dict(pid=len(out), clip=i, start=start, period=period, genre=str(lib.genres[i]),
                                    bpm=float(lib.bpm[i]), energy=float(speed[s:e].mean()),
                                    f_in=_pose_feat(rot[s]), f_out=_pose_feat(rot[e])))
                k += 4  # bar-aligned starts
        # Energy as a percentile within each genre, so genres compare fairly.
        for g in set(p["genre"] for p in out):
            ps = [p for p in out if p["genre"] == g]
            ranks = np.argsort(np.argsort([p["energy"] for p in ps]))
            for p, r in zip(ps, ranks):
                p["energy_pct"] = r / max(len(ps) - 1, 1)
        return out

    def generate(self, audio: str | Path, seconds: float | None = None, genre: str = "auto", seed: int = 0,
                 energy: str = "match", fixed: dict | None = None, exclude: dict | None = None,
                 plan: list | None = None, constraints: list | None = None):
        """fixed: {phrase slot: pid} keeps those picks; exclude: {slot: set(pid)} bans picks
        (used to re-roll one phrase of an existing dance). plan: per phrase slot
        {"style": code, "energy": calm|match|intense}, written by the AI choreographer."""
        fixed, exclude = fixed or {}, exclude or {}
        rng = np.random.default_rng(seed)
        song = analyse_song(audio, seconds)
        dur = song["duration"]
        beats = _extend_beats(song["beats"], dur, song["tempo"])
        n_phr = int(np.ceil((len(beats) - 1) / PHRASE))

        if genre == "auto" or genre not in GENRES:
            genre = str(rng.choice(sorted(GENRES)))
        plan = list(plan or [])
        slot_style = [plan[min(j, len(plan) - 1)].get("style", genre) if plan else genre for j in range(n_phr)]
        slot_style = [g if g in GENRES else genre for g in slot_style]
        slot_energy = [plan[min(j, len(plan) - 1)].get("energy", energy) if plan else energy for j in range(n_phr)]
        pools = {g: [p for p in self.phrases if p["genre"] == g] for g in set(slot_style)}
        # Song energy per phrase, as a percentile across the song.
        e_beat = np.interp(beats, song["beats"], song["energy"]) if len(song["beats"]) else np.zeros(len(beats))
        e_phr = np.array([e_beat[j * PHRASE:(j + 1) * PHRASE].mean() for j in range(n_phr)])
        e_pct = np.argsort(np.argsort(e_phr)) / max(n_phr - 1, 1)
        # Energy preference shifts which moves count as a fit: calm favours the
        # gentler half of the library, intense the bigger half.
        e_pct = np.array([0.45 * e if en == "calm" else 0.55 + 0.45 * e if en == "intense" else e
                          for e, en in zip(e_pct, slot_energy)])

        chosen, recent = [], []
        for j in range(n_phr):
            if j in fixed:
                pick = self.phrases[fixed[j]]
                chosen.append(pick)
                recent = (recent + [pick["clip"]])[-3:]
                continue
            b0 = beats[min(j * PHRASE, len(beats) - 1)]
            b1 = beats[min((j + 1) * PHRASE, len(beats) - 1)]
            song_bpm = 60.0 * PHRASE / max(b1 - b0, 1e-3)
            pool = pools[slot_style[j]]
            cost = np.array([
                (0 if not chosen else np.linalg.norm(chosen[-1]["f_out"] - p["f_in"]) / 3.0)
                + 1.5 * abs(p["energy_pct"] - e_pct[j])
                + 2.0 * abs(np.log(song_bpm / p["bpm"]))
                + (1.5 if p["clip"] in recent else 0.0)
                + (1e6 if p["pid"] in exclude.get(j, ()) else 0.0)
                + (0.8 if j + 1 in fixed and np.linalg.norm(p["f_out"] - self.phrases[fixed[j + 1]]["f_in"]) > 6 else 0.0)
                for p in pool
            ])
            top = np.argsort(cost)[:8]
            w = np.exp(-(cost[top] - cost[top].min()) / 0.15)
            pick = pool[int(rng.choice(top, p=w / w.sum()))]
            chosen.append(pick)
            recent = (recent + [pick["clip"]])[-3:]

        rot, root = self._assemble(chosen, beats, dur)
        if constraints:  # body constraints ("left hand still in the intro"), per plan section
            import constraints as K
            ranges: dict[int, tuple[int, int]] = {}
            for j in range(n_phr):
                sec = (plan[min(j, len(plan) - 1)].get("section") if plan else None) or 0
                f0 = int(max(0.0, beats[min(j * PHRASE, len(beats) - 1)]) * FPS)
                f1 = int(max(0.0, beats[min((j + 1) * PHRASE, len(beats) - 1)]) * FPS)
                a0, a1 = ranges.get(sec, (f0, f1))
                ranges[sec] = (min(a0, f0), max(a1, f1))
            rot = K.apply(rot, constraints, ranges, FPS)
        pos, _ = motion.forward_kinematics(rot, root, motion.REST_OFFSETS)
        sample = {"smpl_poses": rot.reshape(len(rot), 72), "smpl_trans": root, "full_pose": pos}
        T = len(rot) / FPS
        timeline = []
        for j, p in enumerate(chosen):
            t0 = max(0.0, float(beats[min(j * PHRASE, len(beats) - 1)]))
            t1 = T if j == len(chosen) - 1 else min(T, float(beats[min((j + 1) * PHRASE, len(beats) - 1)]))
            if t0 >= T:
                break
            if t1 - t0 < 0.25:  # pre-roll phrase that ends before the song starts
                continue
            timeline.append({"slot": j, "pid": p["pid"], "t0": round(max(0.0, t0), 3), "t1": round(min(T, t1), 3),
                             "source": str(self.lib.names[p["clip"]]), "style": GENRES[p["genre"]],
                             "energy": round(float(p["energy_pct"]), 2)})
        song_beats = [round(float(b), 3) for b in song["beats"] if 0 <= b <= T]
        used = list(dict.fromkeys(slot_style[t["slot"]] for t in timeline)) or [genre]
        info = {"genre": " + ".join(GENRES[g] for g in used), "genre_code": used[0] if len(used) == 1 else genre, "tempo": round(song["tempo"], 1), "energy": energy,
                "phrases": [str(self.lib.names[p["clip"]]) for p in chosen], "picks": [p["pid"] for p in chosen],
                "timeline": timeline, "beats": song_beats}
        return sample, info

    def _segment(self, p, song_knots: np.ndarray, t_out: np.ndarray):
        """Sample a phrase at output times t_out, warping clip beats onto song beats."""
        rot, trans, _ = self.lib.clip(p["clip"])
        clip_knots = p["start"] + p["period"] * np.arange(len(song_knots))
        # piecewise-linear warp with linear extrapolation at the ends
        f = np.interp(t_out, song_knots, clip_knots)
        lo, hi = t_out < song_knots[0], t_out > song_knots[-1]
        r0 = (clip_knots[1] - clip_knots[0]) / (song_knots[1] - song_knots[0])
        r1 = (clip_knots[-1] - clip_knots[-2]) / (song_knots[-1] - song_knots[-2])
        f[lo] = clip_knots[0] + (t_out[lo] - song_knots[0]) * r0
        f[hi] = clip_knots[-1] + (t_out[hi] - song_knots[-1]) * r1
        f = np.clip(f * 1.0, 0, len(rot) - 1.001)
        i0 = np.floor(f).astype(int); a = (f - i0)[:, None]
        q0 = R.from_rotvec(rot[i0].reshape(-1, 3)).as_quat().reshape(len(f), 24, 4)
        q1 = R.from_rotvec(rot[i0 + 1].reshape(-1, 3)).as_quat().reshape(len(f), 24, 4)
        q = _nlerp(q0, q1, a[..., None])
        tr = trans[i0] * (1 - a) + trans[i0 + 1] * a
        return q, tr.astype(np.float64)

    def _assemble(self, chosen, beats, dur):
        T = int(round(dur * FPS))
        t_all = np.arange(T) / FPS
        out_q = np.zeros((T, 24, 4)); out_tr = np.zeros((T, 3))
        prev_end = None
        for j, p in enumerate(chosen):
            k0 = j * PHRASE
            knots = beats[k0:k0 + PHRASE + 1] * FPS
            if len(knots) < 2:
                break
            a = 0 if j == 0 else int(np.ceil(knots[0]))
            b = T if j == len(chosen) - 1 else min(T, int(np.ceil(knots[-1])))
            if a >= T:
                break
            b_ext = min(T, b + BLEND)
            q, tr = self._segment(p, knots, np.arange(a, b_ext, dtype=np.float64))
            # floor: lift each phrase so its lowest foot point sits at y=0
            pos, _ = motion.forward_kinematics(R.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(-1, 24, 3), tr, motion.REST_OFFSETS)
            tr[:, 1] -= pos[:, [7, 8, 10, 11], 1].min()
            # continue from where the previous phrase left the dancer (xz), drifting back to centre
            if prev_end is not None:
                shift = prev_end - tr[0, [0, 2]]
                tr[:, [0, 2]] += shift - 0.15 * (prev_end)
            else:
                tr[:, [0, 2]] -= tr[0, [0, 2]]
            n_own = b - a
            if j == 0:
                out_q[a:b_ext] = q; out_tr[a:b_ext] = tr
            else:
                # cross-fade over the overlap written by the previous phrase's extension
                nb = min(BLEND, b_ext - a)
                w = _smoothstep(np.linspace(0, 1, nb + 2)[1:-1])[:, None]
                out_q[a:a + nb] = _nlerp(out_q[a:a + nb], q[:nb], w[..., None])
                out_tr[a:a + nb] = out_tr[a:a + nb] * (1 - w) + tr[:nb] * w
                out_q[a + nb:b_ext] = q[nb:]; out_tr[a + nb:b_ext] = tr[nb:]
            prev_end = tr[min(n_own, len(tr)) - 1, [0, 2]].copy()
        rot = R.from_quat(out_q.reshape(-1, 4)).as_rotvec().reshape(T, 24, 3)
        return rot, out_tr


def _nlerp(q0: np.ndarray, q1: np.ndarray, w) -> np.ndarray:
    d = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    q = q0 * (1 - w) + q1 * w
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _smoothstep(x):
    return x * x * (3 - 2 * x)
